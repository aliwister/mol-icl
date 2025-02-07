import torch, time, random
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from info_nce import InfoNCE
from torch_geometric.data import Data, Batch
from torch_scatter import scatter
import torch_geometric.utils.smiles as smiles


class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_dim)
        self.conv2 = GATConv(hidden_dim, latent_dim)
        self.linear = torch.nn.Linear(latent_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        # Two GCN layers with ReLU activation
        x = self.conv1(x.float(), edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.linear(x)
        return x  # Latent node embeddings


class MLPDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_channels):
        super(MLPDecoder, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, z):
        return self.mlp(z)  # Reconstructed node features

class GAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, out_channels):
        super(GAE, self).__init__()
        self.encoder = GATEncoder(in_channels, hidden_dim, latent_dim)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr):
        # Encoder (GCN) step
        z = self.encoder(x, edge_index, edge_attr)
        
        # Pool node embeddings to get graph-level embedding
        graph_embedding = global_mean_pool(z, batch)
        
        # Decoder (MLP) step for node reconstruction
        x_hat = self.decoder(z)
        return x_hat, z, graph_embedding  # Return reconstructed node features, latent embeddings, and graph embedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import pdb

def train(train_loader, val_loader, epochs=1):
    input_dim = 9  # From the smiles2graph function output
    hidden_dim = 32
    embedding_dim = 16
    
    model = GAE(in_channels=input_dim, hidden_dim=hidden_dim, latent_dim=embedding_dim, out_channels=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, z, g_embed  = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = F.mse_loss(x_hat, batch.x)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_hat, z, g_embed  = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                loss = F.mse_loss(x_hat, batch.x)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    
    end_time = time.time()
    time_gnn = end_time - start_time
    return model