import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_scatter import scatter
import lightning as L
import torch_geometric.utils.smiles as smiles
import GCL.augmentors as A


def smiles2graph(smiles_str):
    data = smiles.from_smiles(smiles_str)
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()
    return Data(x=data.x, edge_index=data.edge_index)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        # Two GCN layers with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
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

class GAEWithPooling(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, out_channels, train_graphs):
        super(GAEWithPooling, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_dim, latent_dim)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, out_channels)
        self.train_graphs = train_graphs


    def forward(self, x, edge_index, batch):
        # Encoder (GCN) step
        z = self.encoder(x, edge_index)
        
        # Pool node embeddings to get graph-level embedding
        graph_embedding = global_mean_pool(z, batch)
        
        # Decoder (MLP) step for node reconstruction
        x_hat = self.decoder(z)
        return x_hat, z, graph_embedding  # Return reconstructed node features, latent embeddings, and graph embedding
