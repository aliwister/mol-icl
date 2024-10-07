import numpy as np
import pandas as pd
import torch
import time
from argparse import ArgumentParser
from datasets import load_dataset
from util.util import create_incontext_prompt2, get_random_samples, get_samples_new, get_samples_new_irl, get_samples_top
from util.dataset import GraphTextDataset
from util.scibert import get_batched_text_outputs, get_tokenizer
from model.mmcl_attr import MultiModalCLAttr, train as train_mmcl_attr
from model.gae_gcl_attr import train as train_gae_gcl_attr
from util.icl import ICL
import torch_geometric.utils.smiles as smiles
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from model.airl2 import get_reward_function


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def smiles2graph(smiles_str):
    data = smiles.from_smiles(smiles_str)
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()
    return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)

def create_input_file(args, prompts, refs, num_examples, label):
    data_dict = {
        'prompt': np.squeeze(prompts),
        'ref': refs
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(f"/home/ali.lawati/mol-incontext/input/{args.method}-{args.dataset}-{num_examples}-epochs{args.epochs}-{label}.csv", index=False)

def create_embed_file(args, embeds, test_pool, num_examples):
    np.savez_compressed(f"/home/ali.lawati/mol-incontext/input/embed/{args.method}-{args.dataset}-{num_examples}-epochs{args.epochs}-embeds.npz", embeds=embeds, test_pool=test_pool)

def create_meta_file(args, indices, num_examples, label):
    np.save(f"/home/ali.lawati/mol-incontext/input/embed/{args.method}-{args.dataset}-{num_examples}-epochs{args.epochs}-{label}-meta.npy", np.array(indices))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="chebi") # default="liupf/ChEBI-20-MM") 
    parser.add_argument('--num_examples', type=int, default=2)
    parser.add_argument('--method', type=str, default="mmcl_attr")
    parser.add_argument('--limit', type=int, default=100) 
    parser.add_argument('--epochs', type=int, default=300) 
    parser.add_argument('--gpus', type=int, default='1')
    parser.add_argument('--model_checkpoint', type=str, default='/home/ali.lawati/mol-incontext/checkpoints/mmcl-300.pt')
    parser.add_argument('--create_embeds', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default='16')
    
    args = parser.parse_args()
    print(f"Starting: method: {args.method}, limit: {args.limit}, epochs: {args.epochs}")

    dataset_name = args.dataset
    if (args.dataset == "chebi"):
        dataset_name = 'liupf/ChEBI-20-MM'

    dataset = load_dataset(dataset_name)
    df_train = dataset['train'].to_pandas()
    df_valid = dataset['validation'].to_pandas()
    df_test = dataset['test'].to_pandas()

    start_time = time.time()

    if(args.method[0:4] == "mmcl"):
        max_seq_len = 512
        batch_size = 128
        text_tokenizer, text_model = get_tokenizer()
        
        test_graphs = [smiles2graph(smiles) for smiles in df_test['SMILES']]
        val_graphs = [smiles2graph(smiles) for smiles in df_valid['SMILES']]
        train_graphs = [smiles2graph(smiles) for smiles in df_train['SMILES']]
        
        if args.model_checkpoint:
            model = MultiModalCLAttr(9, 32, 64, 9)  # Replace with your model class 
            model.load_state_dict(torch.load('/home/ali.lawati/mol-incontext/checkpoints/mmcl-300.pt', map_location=torch.device(device)))
            model.to(device)
        else:
            train_repr = get_batched_text_outputs(device, df_train['description'].to_numpy(), text_tokenizer, text_model, max_seq_len, batch_size=512)
            train_loader = DataLoader(GraphTextDataset(train_graphs, train_repr), batch_size=batch_size, shuffle=False)
            valid_repr = get_batched_text_outputs(device, df_valid['description'].to_numpy(), text_tokenizer, text_model, max_seq_len, batch_size=512)
            val_loader = DataLoader(GraphTextDataset(val_graphs, valid_repr), batch_size=batch_size, shuffle=False)                
            test_repr = get_batched_text_outputs(device, df_test['description'].to_numpy(), text_tokenizer, text_model, max_seq_len, batch_size=512)
            test_loader = DataLoader(GraphTextDataset(test_graphs, test_repr), batch_size=batch_size, shuffle=False)
            model = train_mmcl_attr(val_loader, val_loader, valid_repr, args.epochs, batch_size)
            torch.save(model.state_dict(), f"/home/ali.lawati/mol-incontext/checkpoints/mmcl-{args.epochs}.pt")
        model.eval()
        with torch.no_grad():
            train_batch = Batch.from_data_list(train_graphs).to(device)
            test_batch  = Batch.from_data_list(test_graphs).to(device)
            valid_batch  = Batch.from_data_list(val_graphs).to(device)
            train_pool = model(train_batch.x, train_batch.edge_index, train_batch.batch, train_batch.edge_attr)
            test_pool  = model(test_batch.x, test_batch.edge_index, test_batch.batch, test_batch.edge_attr)
            valid_pool = model(valid_batch.x, valid_batch.edge_index, valid_batch.batch, valid_batch.edge_attr)
            
    LEN = 5
    df = df_test
    sample_pool = test_pool
    refs = []
    prompts = [[] for i in range(0,5)]
    indices = [[] for i in range(0,5)]
    #irl = get_reward_function()

    prompts2, prompts3 = [], []
    for i in range(0, len(df)):
        prog = df.iloc[i]['SMILES']
        # Create Input files with different demonstration sizes
        for n in range(0, LEN):
            if(args.method == "mmcl-random"):
                samples, logits = get_random_samples(df_train, train_pool, n+1)
            else: 
                samples, selected_indices = get_samples_new(sample_pool[i], df_train, train_pool, n+1)
                #samples2 = get_samples_top(valid_pool[i], df_train, train_pool, n+1)
                #samples3, logits3 = get_samples_new_irl(irl, valid_pool[i], df_train, train_pool, n+1)
            prompts[n].append(create_incontext_prompt2(*samples + [prog]))
            indices[n].append(selected_indices)
            #prompts2.append(create_incontext_prompt2(*samples2 + [prog]))
            #prompts3.append(create_incontext_prompt2(*samples3 + [prog]))
        refs.append(df.iloc[i]['description'])
    [create_input_file(args, prompts[i], refs, i+1, 'new-test') for i in range(0, LEN)]
    #[create_input_file(args, prompts2, refs, i+1, 'top') for i in range(1, 2)]
    #[create_input_file(args, prompts3, refs, i+1, 'irl') for i in range(1, 2)]
    #if (args.create_embeds):
    [create_meta_file(args, indices[i], i+1, 'new-test') for i in range(0, LEN)]

    #end_time = time.time()
    #time_prompt = end_time - start_time
    #print("Prompt Selection time:", time_prompt)
    torch.cuda.empty_cache()


    """        if(args.method == "random"):
            p_args1 = get_samples(df_train, -1, args.num_examples) + [prog]
            prompt1 = create_incontext_prompt2(*p_args1)
        elif(args.method[0:4] == "mmcl" or args.method[0:3] == "gae"):


        elif(args.method=="gae-gcl"):
            p_args1 = get_samples_new(test_pool[i], df_train, train_pool, args.num_examples) + [prog]
            prompt1 = create_incontext_prompt2(*p_args1)
        el
        elif(args.method=="zero1"):
            p_args2 = (prog,)
            prompt1 = create_zeroshot_prompt1(*p_args2)
        elif(args.method=="zero2"):
            p_args2 = (prog,)
            prompt1 = create_zeroshot_prompt2(*p_args2)
        elif(args.method=="cot"):
            p_args2 = (prog,)
            prompt1 = create_cot_prompt(*p_args2)
        elif(args.method=="zero-gpt"):
            prompt1 = prog
        else:
            p_args1 = get_samples_new(test_pool[i], df_train, train_pool, args.num_examples) + [prog]
            prompt1 = create_incontext_prompt2(*p_args1)"""
