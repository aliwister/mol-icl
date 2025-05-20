import numpy as np
import pandas as pd
import torch
#print(torch.__version__)
import time
from argparse import ArgumentParser
from datasets import load_dataset
from util.dataset import GraphTextMorganDataset, PubChemDataset
from util.scibert import get_batched_text_outputs, get_tokenizer
from model.mmcl import train as train
from rdkit import Chem
from rdkit.Chem import AllChem
import torch_geometric.utils.smiles as smiles
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def smiles2graph(smiles_str, text):
    data = smiles.from_smiles(smiles_str)
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()
    return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, smiles=smiles_str, text=text)

# Function to convert a SMILES string to a Morgan fingerprint
def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle invalid SMILES
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def parse(delign):
    method = 'graph-bm25'
    if(delign[0]['M'] > 0):
        method = 'morgan'

    if(delign[1]['M'] > 0 and delign[1]['T'] > 0):
        method = method + "-text+morgan"
    elif(delign[1]['M'] > 0):
        method = method + "-morgan"
    elif(delign[1]['T'] > 0):
        method = method + "-text"

    return method

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="chebi") # default="liupf/ChEBI-20-MM") 
    parser.add_argument('--num_examples', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=301) 
    parser.add_argument('--gpus', type=int, default='1')
    parser.add_argument('--create_embeds', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default='16')
    parser.add_argument('--delign', type=object, default=[{'G': 1, 'M':0}, {'T':1, 'M':0, 'K':0}])
    parser.add_argument('--bert', type=str, default="pubmedbert")
    parser.add_argument('--is_morgan', type=bool, default=True)
    
    args = parser.parse_args()
    print(f"Start Training = loss: {args.delign}, epochs: {args.epochs}, batch_size: {args.batch_size}")

    if (args.dataset == "chebi"):
        df_train = pd.read_csv('./data/chebi/train.txt', sep='\t')
        df_valid = pd.read_csv('./data/chebi/validation.txt', sep='\t')
        #df_train = df_train[:1000]
        #df_valid = df_valid[:100]
        val_graphs = [smiles2graph(smiles, text) for (smiles, text) in zip(df_valid['SMILES'], df_valid['description'])]
        train_graphs = [smiles2graph(smiles, text) for (smiles, text) in zip(df_train['SMILES'], df_train['description'])]

        train_captions, val_captions  = df_train['description'].to_numpy(), df_valid['description'].to_numpy()
        train_smiles, val_smiles  = df_train['SMILES'].to_numpy(), df_valid['SMILES'].to_numpy()
    elif(args.dataset == 'pubchem'):
        df_train = pd.read_csv('./data/chebi/train.txt', sep='\t')
        df_valid = pd.read_csv('./data/chebi/validation.txt', sep='\t')
        val_graphs2 = [smiles2graph(smiles, text) for (smiles, text) in zip(df_valid['SMILES'], df_valid['description'])]
        train_graphs2 = [smiles2graph(smiles, text) for (smiles, text) in zip(df_train['SMILES'], df_train['description'])]
        val_graphs2 = [smiles2graph(smiles, text) for (smiles, text) in zip(df_valid['SMILES'], df_valid['description'])]
        train_graphs2 = [smiles2graph(smiles, text) for (smiles, text) in zip(df_train['SMILES'], df_train['description'])]
        train_captions2, val_captions2  = df_train['description'].to_numpy(), df_valid['description'].to_numpy()
        train_smiles2, val_smiles2  = df_train['SMILES'].to_numpy(), df_valid['SMILES'].to_numpy()

        train_graphs = PubChemDataset('./data/reactXT/caption_data/train.pt')
        val_graphs = PubChemDataset('./data/reactXT/caption_data/valid.pt')
        train_captions = [x['text'] for x in train_graphs]
        val_captions = [x['text'] for x in val_graphs]
        train_smiles = [x['smiles'] for x in train_graphs]
        val_smiles = [x['smiles'] for x in val_graphs]
    else:
        raise ValueError

    start_time = time.time()

    radius = 2
    n_bits = 2048
    val_morgan = [smiles_to_morgan(smiles, radius, n_bits) for smiles in val_smiles]
    train_morgan = [smiles_to_morgan(smiles, radius, n_bits) for smiles in train_smiles]
    max_seq_len = 512
    batch_size = 128
    text_tokenizer, text_model = get_tokenizer(args.bert)
    #get_pos_neg_sample(train_morgan[0], train_morgan, 1)
    val_morgan_torch = torch.tensor(list(val_morgan)).float()
    train_morgan_torch = torch.tensor(list(train_morgan)).float()

    BM25 = False
    is_morgan = args.is_morgan
    train_repr = get_batched_text_outputs(device, train_captions, text_tokenizer, text_model, max_seq_len, batch_size=512)
    train_loader = DataLoader(GraphTextMorganDataset(train_graphs, train_repr, train_morgan_torch), batch_size=batch_size, shuffle=False)
    valid_repr = get_batched_text_outputs(device, val_captions, text_tokenizer, text_model, max_seq_len, batch_size=512)
    val_loader = DataLoader(GraphTextMorganDataset(val_graphs, valid_repr, val_morgan_torch), batch_size=batch_size, shuffle=False)                
    model = train(train_loader, val_loader, train_repr, train_morgan, val_morgan, args.epochs, batch_size, is_morgan)
    torch.save(model.state_dict(), f"./checkpoints/mmcl-morgan-exp1-{args.epochs}-{args.dataset}-{args.bert}-768-morgan={is_morgan}.pt")