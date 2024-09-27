import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_scatter import scatter

import torch_geometric.utils.smiles as smiles
from info_nce import InfoNCE
from util.contrastive_similarity import GraphContrastiveSimilarity, find_similar_to_A_different_from_B, train_model
import torch_geometric.utils.smiles as smiles
import time
from datasets import load_dataset
from torch.utils.data import Dataset
import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
info_loss = InfoNCE(negative_mode='unpaired')
    

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_dim)
        self.conv2 = GATConv(hidden_dim, latent_dim)
        self.linear = torch.nn.Linear(latent_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        # Two GCN layers with ReLU activation
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.linear(x)
        return x  # Latent node embeddings


class MultiModalCLAttr(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, out_channels):
        super(MultiModalCLAttr, self).__init__()
        self.encoder = GATEncoder(in_channels, hidden_dim, latent_dim)
        #self.decoder = MLPDecoder(latent_dim, hidden_dim, out_channels)
        text_dim = 768
        self.text2latent = torch.nn.Linear(text_dim, latent_dim)


    def forward(self, x, edge_index, batch, edge_attr):
        # Encoder (GCN) step
        z = self.encoder(x, edge_index, edge_attr)
        #z = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Pool node embeddings to get graph-level embedding
        z = global_mean_pool(z, batch)
        
        # Decoder (MLP) step for node reconstruction
        #x_hat = self.decoder(z)
        return z  # Return reconstructed node features, latent embeddings, and graph embedding

def train(train_loader, val_loader, train_repr, epochs=1, batch_size=64):
    input_dim = 9  # From the smiles2graph function output
    hidden_dim = 32
    embedding_dim = 64
    
    model = MultiModalCLAttr(in_channels=input_dim, hidden_dim=hidden_dim, latent_dim=embedding_dim, out_channels=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    train_repr.to(device)
    for epoch in range(epochs):
        model.train() 
        model.text2latent.train()
        train_loss = 0.0

        for batch in train_loader:
            graph, repr = batch 
            graph.to(device)
            repr.to(device)
            optimizer.zero_grad()

            z  = model(graph.x, graph.edge_index, graph.batch, graph.edge_attr)
            pos = model.text2latent(repr)
            weights = torch.ones(train_repr.shape[0])
            indices = torch.multinomial(weights, batch_size, replacement=True)
            neg_raw = train_repr[indices]
            neg = model.text2latent(neg_raw)

            loss = info_loss(z, pos, neg) 

            train_loss_epoch = loss
            train_loss_epoch.backward()
            optimizer.step()
            train_loss += train_loss_epoch.item()

        
        train_loss /= len(train_loader)
        model.text2latent.eval()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                graph, repr = batch 
                graph.to(device)
                repr.to(device)#.to(device)
                z  = model(graph.x, graph.edge_index, graph.batch, graph.edge_attr)
                
                pos = model.text2latent(repr)

                weights = torch.ones(train_repr.shape[0])
                indices = torch.multinomial(weights, batch_size, replacement=True)
                neg_raw = train_repr[indices]
                neg = model.text2latent(neg_raw)

                loss = info_loss(z, pos, neg) / 3

                val_loss_epoch = loss
                val_loss += val_loss_epoch.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    end_time = time.time()
    time_gnn = end_time - start_time
    return model