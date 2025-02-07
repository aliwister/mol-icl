import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser
from model.mmcl import MMCL
from util.util import create_incontext_prompt2, get_random_samples, smiles2graph, get_samples_top, generate_scaffolds, create_input_file, get_scaffold_samples, get_samples_new
from model.gae import GAE
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split

TASK = "yield"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def assemble(method, dataset, checkpoint, random_state = 42, llambda=0.3):
    if (method == "mmcl-nomorgan"):
        checkpoint = "./checkpoints/mmcl-morgan-exp1-301-chebi-scibert-768-morgan=False.pt"

    label_source = 'reaction' 
    label_target = 'yield'

    input_label = "Reaction"
    output_label = "High-Yield"

    if not args.model_checkpoint:
        raise Exception("Trained model required") 

    if dataset == "suzuki":
        suzuki = np.load('./data/yield_prediction/Suzuki.npz', allow_pickle=True)
        df_train = pd.DataFrame(suzuki['data_df'], columns = ['reaction', 'yield_num'])
    else:
        bh = np.load("./data/yield_prediction/BH_dataset .npz", allow_pickle=True)
        df_train = pd.DataFrame({'reaction': bh['data'][2]['rsmi'], 'yield_num': bh['data'][2]['yld']})


    df_train['smiles'] = df_train[label_source].str.split('>>').str[0]
    df_train['yield'] = df_train['yield_num'].apply(lambda x: "Yes" if x > 50 else "No")

    df_train, df_test = train_test_split(df_train, test_size=0.1, random_state=random_state)
    df_train, df_val = train_test_split(df_train, test_size=0.1111, random_state=random_state) 
    
    test_graphs, train_graphs = [], []

    train_graphs  = [smiles2graph(smiles) for smiles in df_train['smiles']]
    test_graphs  = [smiles2graph(smiles) for smiles in df_test['smiles']]
    
    
    
    if(method == "scaffold"):
        sc_train_pool = generate_scaffolds(df_train, 'smiles')
        sc_test_pool = generate_scaffolds(df_test, 'smiles')
    elif(method != "random"):
        if method == "mmcl":
            model = MMCL(9, 128, 768, 9)
        else:
            model = GAE(9, 32, 16, 9)
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device(device)))
        model.to(device)

        model.eval()
        with torch.no_grad():
            train_batch = Batch.from_data_list(train_graphs).to(device)
            test_batch  = Batch.from_data_list(test_graphs).to(device)
            _, _, train_pool = model(train_batch.x, train_batch.edge_index, train_batch.batch, train_batch.edge_attr)
            _, _, test_pool  = model(test_batch.x, test_batch.edge_index, test_batch.batch, test_batch.edge_attr)

    LEN = 8
    df = df_test
    refs = []
    prompts = [[] for i in range(0,LEN)]
    indices = [[] for i in range(0,LEN)]
    for i in range(0, len(df)):
        prog = df.iloc[i][label_source]
        # Create Input files with different demonstration sizes
        for n in range(0, LEN):
            if(method == "random"):
                samples = get_random_samples(df_train, n+1, label_source, label_target, True)
            elif(method == "scaffold"):
                samples = get_scaffold_samples(sc_test_pool[i], df_train, sc_train_pool, n+1, label_source, label_target)
            elif(method == "mmcl-top"):
                samples = get_samples_top(test_pool[i], df_train, train_pool, n+1, label_source, label_target)
            else: 
                samples, selected_indices = get_samples_new(test_pool[i], df_train, train_pool, n+1, label_source, label_target, llambda=llambda)

                indices[n].append(selected_indices)
            prompts[n].append(create_incontext_prompt2(*samples + [prog],  input_label=input_label,  output_label=output_label))

        refs.append(df.iloc[i][label_target])
    [create_input_file(prompts[i], refs, i+1, f"{method}-{dataset}-gae-jan3_lambda={llambda}-{random_state}", TASK) for i in [0,1,2,3,4]]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="./ChemLLMBench/data/yield_prediction/Suzuki.npz")
    parser.add_argument('--num_examples', type=int, default=2)
    parser.add_argument('--method', type=str, default="mmcl")
    parser.add_argument('--gpus', type=int, default='1')
    parser.add_argument('--model_checkpoint', type=str, default='./checkpoints/gae-301-chebi-scibert.pt')
    parser.add_argument('--create_embeds', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default='16')
    parser.add_argument('--experiment', type=str, default="suzuki-yield")
    parser.add_argument('--lamda', type=int, default='.3')
    
    args = parser.parse_args()
    checkpoint = args.model_checkpoint
    datasets = ['suzuki', 'bh'] #['tox21', 'clintox', 'BACE', 'BBBP']
    random_states = [3, 42, 53] 
    for rs in random_states:
        assemble(args.method, args.dataset, checkpoint, rs, args.lamda)