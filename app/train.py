import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time
from itertools import chain
from argparse import ArgumentParser
from datasets import load_dataset
from util.dataset import GraphTextDataset
from util.scibert import get_batched_text_outputs, get_tokenizer
from model.mmcl import train as train_mmcl
from model.mmcl_attr import MultiModalCLAttr, train as train_mmcl_attr
from model.gae_gcl import GAEWithPooling, train as train_gae_gcl
from model.gae_gcl_attr import GAEWithAttributes, train as train_gae_gcl_attr
from util.icl import ICL
#from util.chatgpt import run_chatgpt
from run_prompts import run_prompts
import torch_geometric.utils.smiles as smiles
from util.balanced_kmeans import balanced_kmeans
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from util.measure import measure
from model.airl2 import get_reward_function

def create_incontext_prompt2(*args):
    if len(args) % 2 == 0:
        raise ValueError("The number of arguments must be odd.")
    
    # Initialize an empty string to accumulate the formatted text
    formatted_text = ""
    # Iterate through pairs of arguments
    for i in range(0, len(args)-1, 2):
        input_str = args[i]
        output_str = args[i + 1]
        # Format the input and output into the desired format
        formatted_text += 'Input: {0}\nOutput: {1}\n###\n'.format(input_str, output_str)
    formatted_text += 'Input: {0}\nOutput:'.format(args[-1])
    return formatted_text

def smiles2graph(smiles_str):
    data = smiles.from_smiles(smiles_str)
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()
    return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)

def encode(self, batch, lang_model):
    x, edge_index, attention_mask = batch.x, batch.edge_index, batch.attention_mask
    inputs = lang_model(x, output_hidden_states=True, attention_mask=attention_mask)
    hidden_states = inputs.hidden_states
    last_hidden_state = hidden_states[-1]
    denom = torch.sum(attention_mask, -1, keepdim=True)
    feat = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / denom
    feat = feat.to(torch.float32)
    return feat

def get_random_samples(df_train, train_pool, num_examples):
    sampled_data = df_train.sample(n=num_examples)
    sampled_indices = sampled_data.index
    return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description']))), train_pool[sampled_indices]

def run_transformer(args, df, time_prompt):
    if args.limit > 0:
        df = df[0:args.limit]

    prompts_all = df['prompt1'].to_numpy().flatten()
    references = df['ref'].tolist()

    output_file = f"./output/{args.dataset}-{args.model_name}-{args.method}-{args.num_examples}.csv"
    if (args.model_name == "openai/chatgptxxxx"):
        #run_chatgpt(args.langmodel_name_model, prompts_all, output_file)
        exit(-1)
    else:
        run_prompts(args.model_name, prompts_all, output_file)
        
    measure(args, output_file, time_prompt, references)

B_ALPHA = 0.3
def get_samples_new(a_emb, df_train, train_pool, num_examples):
    alpha = B_ALPHA
    selected_indices = []
    combined_score = F.cosine_similarity(a_emb, train_pool)
    for _ in range(num_examples):
        top20 = torch.topk(combined_score, 20)[1]
        idx = next((i for i in top20 if i not in selected_indices), None).item()
        selected_indices.append(idx)
        
        b_emb = train_pool[idx]
        similarity_to_selected =  F.cosine_similarity(b_emb, train_pool)
        combined_score -= alpha * similarity_to_selected

    #print(selected_indices)
    sampled_data = df_train.iloc[selected_indices[::-1]]
    return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description']))), train_pool[selected_indices]

def get_samples_new_irl(irl, a_emb, df_train, train_pool, num_examples):
    
    alpha = B_ALPHA
    selected_indices = []
    combined_score = F.cosine_similarity(a_emb, train_pool)
    for _ in range(num_examples):
        top20 = torch.topk(combined_score, 20)[1]
        top20_embed = train_pool[torch.topk(combined_score, 20)[1]]
        
        idx = next((i for i in top20 if i not in selected_indices), None).item()
        scores = [irl(a_emb.unsqueeze(0), t.unsqueeze(0)) for t in top20_embed]
        while (a_emb.unsqueeze(0), top20_embed.unsqueeze(0)) < 0:
            idx = next((i for i in top20 if i not in [selected_indices, idx]), None).item()
        selected_indices.append(idx)
        a_emb = b_emb
        b_emb = train_pool[idx]
        similarity_to_selected =  F.cosine_similarity(b_emb, train_pool) 
        combined_score -= alpha * similarity_to_selected

    #print(selected_indices)
    sampled_data = df_train.iloc[selected_indices[::-1]]
    return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description']))), train_pool[selected_indices]

def get_samples_top(a_emb, df_train, train_pool, num_examples):
    b_idx = torch.topk(F.cosine_similarity(a_emb, train_pool), num_examples).indices.tolist()
    sampled_data = df_train.iloc[b_idx]
    return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description'])))


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_input_file(args, prompts, refs, num_examples):
    data_dict = {
        'prompt': np.squeeze(prompts),
        'ref': refs
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(f"/home/ali.lawati/mol-incontext/input/{args.method}-{args.dataset}-{num_examples}-epochs{args.epochs}-airl.csv", index=False)

def create_embed_file(args, embeds, test_pool, num_examples):
    np.savez_compressed(f"/home/ali.lawati/mol-incontext/input/embed/{args.method}-{args.dataset}-{num_examples}-epochs{args.epochs}-embeds.npz", embeds=embeds, test_pool=test_pool)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="chebi") # default="liupf/ChEBI-20-MM") 
    parser.add_argument('--num_examples', type=int, default=3)
    parser.add_argument('--method', type=str, default="mmcl_attr")
    parser.add_argument('--limit', type=int, default=100) 
    parser.add_argument('--epochs', type=int, default=300) 
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--encoder_type', type=str, default='GraphAutoencoder')
    parser.add_argument('--gpus', type=int, default='1')
    parser.add_argument('--model_checkpoint', type=str, default='/home/ali.lawati/mol-incontext/checkpoints/mmcl-300.pt')
    parser.add_argument('--create_embeds', type=bool, default=False)
    
    args = parser.parse_args()
    print(f"Starting: method: {args.method}, limit: {args.limit}, epochs: {args.epochs}")

    dataset_name = args.dataset
    if (args.dataset == "chebi"):
        dataset_name = 'liupf/ChEBI-20-MM'

    dataset = load_dataset(dataset_name)
    df_train = dataset['train'].to_pandas()
    df_valid = dataset['validation'].to_pandas()
    df_test = dataset['test'].to_pandas()

    time_gnn = 0
    batch_size = 16
    start_time = time.time()
    if(args.method[0:3] == "gae"):
        #df_train = df_train[0:1000]
        #df_valid = df_valid[0:500]
        #df_test = df_test[0:500]
        train_graphs = [smiles2graph(smiles) for smiles in df_train['SMILES']]
        train_loader = DataLoader(train_graphs, batch_size=16, shuffle=False)

        val_graphs = [smiles2graph(smiles) for smiles in df_valid['SMILES']]
        val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)

        test_graphs = [smiles2graph(smiles) for smiles in df_test['SMILES']]
        test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

        if(args.method == "gae_gcl_attr"):
            model = train_gae_gcl_attr(train_loader, val_loader, train_graphs, args.epochs)
            model.eval()
            torch.save(model.state_dict(), f"gae_gcl_attr-{args.epochs}.pt")
            with torch.no_grad():
                train_batch = Batch.from_data_list(train_graphs).to(device)
                test_batch  = Batch.from_data_list(test_graphs).to(device)
                _, _, _, train_pool = model(train_batch.x, train_batch.edge_index, train_batch.batch, train_batch.edge_attr)
                _, _, _, test_pool = model(test_batch.x, test_batch.edge_index, test_batch.batch, test_batch.edge_attr)
        else:
            if args.model_checkpoint:
                model = GAEWithPooling()  # Replace with your model class
                model.load_state_dict(torch.load(args.model_checkpoint))
            else:
                model = train_gae_gcl(train_loader, val_loader, train_graphs, args.epochs)
                model.eval()
                torch.save(model.state_dict(), f"gcl-gae-{args.epochs}.pt")
            model.eval()
            with torch.no_grad():
                train_batch = Batch.from_data_list(train_graphs).to(device)
                test_batch  = Batch.from_data_list(test_graphs).to(device)
                _, _, train_pool = model(train_batch.x, train_batch.edge_index, train_batch.batch)
                _, _, test_pool = model(test_batch.x, test_batch.edge_index, test_batch.batch)

    start_time = time.time()
    #if args.method == "ss":


    if(args.method[0:4] == "mmcl"):
        max_seq_len = 512
        batch_size = 128
        text_tokenizer, text_model = get_tokenizer()
        
        #df_train = df_train[0:100]
        #df_valid = df_valid[0:50]
        #df_test = df_test[0:50]
        test_graphs = [smiles2graph(smiles) for smiles in df_test['SMILES']]
        val_graphs = [smiles2graph(smiles) for smiles in df_valid['SMILES']]
        train_graphs = [smiles2graph(smiles) for smiles in df_train['SMILES']]


        #if args.model_checkpoint:
        #    model = MultiModalCLGAE(9, 32, 16, 9)  # Replace with your model class
        #    model.load_state_dict(torch.load(args.model_checkpoint))
        #else:
        if args.method == "mmcl":
            train_repr = get_batched_text_outputs(device, df_train['description'].to_numpy(), text_tokenizer, text_model, max_seq_len, batch_size=512)
            train_loader = DataLoader(GraphTextDataset(train_graphs, train_repr), batch_size=batch_size, shuffle=False)
            valid_repr = get_batched_text_outputs(device, df_valid['description'].to_numpy(), text_tokenizer, text_model, max_seq_len, batch_size=512)
            val_loader = DataLoader(GraphTextDataset(val_graphs, valid_repr), batch_size=batch_size, shuffle=False)                
            test_repr = get_batched_text_outputs(device, df_test['description'].to_numpy(), text_tokenizer, text_model, max_seq_len, batch_size=512)
            test_loader = DataLoader(GraphTextDataset(test_graphs, test_repr), batch_size=batch_size, shuffle=False)
            model = train_mmcl(val_loader, val_loader, valid_repr, args.epochs, batch_size)
            model.eval()
            with torch.no_grad():
                train_batch = Batch.from_data_list(train_graphs).to(device)
                test_batch  = Batch.from_data_list(test_graphs).to(device)
                train_pool = model(train_batch.x, train_batch.edge_index, train_batch.batch)
                test_pool = model(test_batch.x, test_batch.edge_index, test_batch.batch)
        else:
            if args.model_checkpoint:
                model = MultiModalCLAttr(9, 32, 64, 9)  # Replace with your model class 
                model.load_state_dict(torch.load('/home/ali.lawati/mol-incontext/checkpoints/mmcl-300.pt', map_location=torch.device('cpu')))
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
                test_pool = model(test_batch.x, test_batch.edge_index, test_batch.batch, test_batch.edge_attr)
                valid_pool = model(valid_batch.x, valid_batch.edge_index, valid_batch.batch, valid_batch.edge_attr)
            
    LEN = 5
    df = df_valid
    refs = []
    prompts = [[] for i in range(0,5)]
    embeds = [[] for i in range(0,5)]
    irl = get_reward_function()

    prompts2, prompts3 = [], []
    for i in range(0, len(df_valid)):
        prog = df.iloc[i]['SMILES']
        # Create Input files with different demonstration sizes
        for n in range(1,2):
            if(args.method == "mmcl-random"):
                samples, logits = get_random_samples(df_train, train_pool, n+1)
            else: 
                samples, logits = get_samples_new(valid_pool[i], df_train, train_pool, n+1)
                samples2 = get_samples_top(valid_pool[i], df_train, train_pool, n+1)
                samples3, logits3 = get_samples_new_irl(irl, valid_pool[i], df_train, train_pool, n+1)
            prompts[n].append(create_incontext_prompt2(*samples + [prog]))
            embeds[n].append(logits)
            prompts2.append(create_incontext_prompt2(*samples2 + [prog]))
            prompts3.append(create_incontext_prompt2(*samples3 + [prog]))
        refs.append(df.iloc[i]['description'])
    [create_input_file(args, prompts[i], refs, i+1, 'new') for i in range(1, 2)]
    [create_input_file(args, prompts2, refs, i+1, 'top') for i in range(1, 2)]
    [create_input_file(args, prompts3, refs, i+1, 'irl') for i in range(1, 2)]
    if (args.create_embeds):
        [create_embed_file(args, torch.stack(embeds[i]).cpu().numpy(), test_pool.cpu().numpy(), i+1) for i in range(0, 5)]


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

    end_time = time.time()
    time_prompt = end_time - start_time
    print("Prompt Selection time:", time_prompt)
    torch.cuda.empty_cache()

    #run_transformer(args, df, time_prompt)

