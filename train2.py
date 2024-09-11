from info_nce import InfoNCE
from util.contrastive_similarity import GraphContrastiveSimilarity, find_similar_to_A_different_from_B, train_model
import torch_geometric.utils.smiles as smiles

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from itertools import chain
import time
import argparse
from datasets import load_dataset

import GCL.augmentors as A

from util.gae_gcl import GAEWithPooling, smiles2graph
import random


class MyDataset:
    def __getitem__(self, idx):
        # Return a single graph as a Data object
        return Data(x=..., edge_index=..., y=...)

    def __len__(self):
        return len(self.data)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

info_loss = InfoNCE(negative_mode='unpaired')
aug = A.RandomChoice([ A.RWSampling(num_seeds=1000, walk_length=10),
                                A.NodeDropping(pn=0.1),
                                A.FeatureMasking(pf=0.1),
                                A.EdgeRemoving(pe=0.1)],
                                num_choices=1)


import pdb

def train(train_loader, val_loader, train_graphs, epochs=100):
    input_dim = 9  # From the smiles2graph function output
    hidden_dim = 32
    embedding_dim = 16
    
    
    model = GAEWithPooling(in_channels=input_dim, hidden_dim=hidden_dim, latent_dim=embedding_dim, out_channels=input_dim, train_graphs=train_graphs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    


    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        r_loss_cum = 0.0
        c_loss_cum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, z, g_embed  = model(batch.x, batch.edge_index, batch.batch)
            
            pos = aug(batch.x, batch.edge_index)
            x_p, z_p, g_embed_p = model(pos[0], pos[1], batch.batch)

            neg = Batch.from_data_list(random.sample(train_graphs, 16)).to(device)
            x_n, z_n, g_embed_n = model(neg.x, neg.edge_index, neg.batch)

            r_loss = F.mse_loss(x_hat, batch.x)
            c_loss = info_loss(g_embed, g_embed_p, g_embed_n)

            train_loss_epoch = r_loss + c_loss
            train_loss_epoch.backward()
            optimizer.step()
            train_loss += train_loss_epoch.item()

            r_loss_cum += r_loss
            c_loss_cum += c_loss
        
        train_loss /= len(train_loader)
        r_loss_cum /= len(train_loader)
        c_loss_cum /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_hat, z, g_embed  = model(batch.x, batch.edge_index, batch.batch)
                
                pos = aug(batch.x, batch.edge_index)
                x_p, z_p, g_embed_p = model(pos[0], pos[1], batch.batch)

                neg = Batch.from_data_list(random.sample(train_graphs, 16)).to(device)
                x_n, z_n, g_embed_n = model(neg.x, neg.edge_index, neg.batch)

                r_loss = F.mse_loss(x_hat, batch.x)
                c_loss = info_loss(g_embed, g_embed_p, g_embed_n)

                val_loss_epoch = r_loss + c_loss
                val_loss += val_loss_epoch.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train C Loss: {c_loss_cum:.4f}, Train R Loss {r_loss_cum:.4f}")
    
    #train_embeddings = extract_latent_representations(model, train_loader)
    #test_embeddings = extract_latent_representations(model, test_loader)
    
    end_time = time.time()
    time_gnn = end_time - start_time
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ICL model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--method", type=str, default="icl-new", help="Method to use")
    parser.add_argument("--limit", type=int, default=0, help="Limit for test data")
    args = parser.parse_args()

    # Load your data here (replace with your actual data loading code)
    dataset_name = 'liupf/ChEBI-20-MM'

    dataset = load_dataset(dataset_name)
    df_train = dataset['train'].to_pandas()
    df_valid = dataset['validation'].to_pandas()
    df_test = dataset['test'].to_pandas()

    train_graphs = [smiles2graph(smiles) for smiles in df_train['SMILES']]
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=False)

    val_graphs = [smiles2graph(smiles) for smiles in df_valid['SMILES']]
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)

    test_graphs = [smiles2graph(smiles) for smiles in df_test['SMILES']]
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    train(train_loader, val_loader, train_graphs)
    # You can add more code here to use the trained model, e.g.:
    # test_results = icl.get_samples(0, 5)
    # print(test_results)
