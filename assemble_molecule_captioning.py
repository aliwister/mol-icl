import pandas as pd
import torch
import time, os
from argparse import ArgumentParser
from model.gae import GAE
from model.mmcl import MMCL
#from test_kvplm import generate_kvplm
from util.util import create_incontext_prompt, create_incontext_prompt2, get_random_samples, get_samples_new, create_input_file, smiles2graph, get_scaffold_samples, generate_scaffolds, get_samples_top#, get_kvplm_samples
from util.dataset import PubChemDataset
from torch_geometric.data import Batch

TASK = "molecule-caption"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def assemble(method, dataset, checkpoint, random_state = 42):
    label_source = 'SMILES' 
    label_target = 'description'

    if dataset == "chebi-20":
        df_train = pd.read_csv('./data/chebi/train.txt', sep='\t')
        #df_train = df_train[:500]
        df_test = pd.read_csv('./data/chebi/test.txt', sep='\t')
        #df_test = df_test[:500]
        test_graphs = [smiles2graph(smiles) for smiles in df_test[label_source]]
        train_graphs = [smiles2graph(smiles) for smiles in df_train[label_source]]

    elif dataset == "pubchem":
        train_graphs = PubChemDataset('./data/reactXT/caption_data/train.pt')
        test_graphs = PubChemDataset('./data/reactXT/caption_data/test.pt')
        train_captions = [x['text'] for x in train_graphs]
        test_captions = [x['text'] for x in test_graphs]
        train_smiles = [x['smiles'] for x in train_graphs]
        test_smiles = [x['smiles'] for x in test_graphs]
        df_train = pd.DataFrame({
            'SMILES': train_smiles,
            'description': train_captions
        })
        df_test = pd.DataFrame({
            'SMILES': test_smiles,
            'description': test_captions
        })

    start_time = time.time()

    if(method == "scaffold"):
        sc_train_pool = generate_scaffolds(df_train, label_source)
        sc_test_pool = generate_scaffolds(df_test, label_source)
    #elif(method == "kvplm"):
    #    sc_train_pool = generate_kvplm(df_train, label_source)
    #    sc_test_pool = generate_kvplm(df_test, label_source)
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
            train_pool = model(train_batch.x, train_batch.edge_index, train_batch.batch, train_batch.edge_attr)
            test_pool  = model(test_batch.x, test_batch.edge_index, test_batch.batch, test_batch.edge_attr)
                
    LEN = 10
    refs = []
    prompts = [[] for i in range(0,LEN)]
    indices = [[] for i in range(0,LEN)]

    demo_counts = [1] 
    df = df_test
    for i in range(0, len(df)):
        prog = df.iloc[i][label_source]
        # Create Input files with different demonstration sizes
        if (method == "zero"):
            prompts[0].append(create_incontext_prompt(prog))
        else:
            for n in demo_counts:
                if(method == "random"):
                    samples = get_random_samples(df_train, n+1, label_source, label_target)
                elif(method == "scaffold"):
                    samples = get_scaffold_samples(sc_test_pool[i], df_train, sc_train_pool, n+1, label_source, label_target)
                #elif(method == "kvplm"):
                #    samples = get_kvplm_samples(sc_test_pool[i], df_train, sc_train_pool, n+1, label_source, label_target)
                elif(method == "mmcl_top"):
                    samples = get_samples_top(test_pool[i], df_train, train_pool, n+1, label_source, label_target)
                else: 
                    samples, selected_indices = get_samples_new(test_pool[i], df_train, train_pool, n+1, label_source, label_target)
                    indices[n].append(selected_indices)
                prompts[n].append(create_incontext_prompt2(*samples + [prog]))

        refs.append(df.iloc[i][label_target])
    
    cp = os.path.splitext(os.path.basename(checkpoint))[0]
    if (method == "zero"):
        create_input_file(prompts[0], refs, 0, method+"-"+dataset+"", TASK)
    else:
        [create_input_file(prompts[i], refs, i+1, f"{method}-{cp}-{dataset}", TASK) for i in demo_counts]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="chebi-20") # default="liupf/ChEBI-20-MM") 
    parser.add_argument('--checkpoint', type=str, default="./checkpoints/gae-301-chebi-scibert.pt")
    parser.add_argument('--method', type=str, default="mmcl")
    args = parser.parse_args()

    assemble(args.method, args.dataset, args.checkpoint)
    
    #datasets = ["chebi-20"] #, "pubchem"]
    #methods = ['mmcl']#'mmcl']
    #checkpoints = [
    #    './checkpoints/mmcl-morgan-exp1-301-chebi-biobert-768-morgan=True.pt',
    #    './checkpoints/mmcl-morgan-exp1-301-chebi-pubmedbert-768-morgan=True.pt',
    #]
    #for c in checkpoints:
    #    for dataset in datasets:
    #        for m in methods:
    #            assemble(m, dataset, c)