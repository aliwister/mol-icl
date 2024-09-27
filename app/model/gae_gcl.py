import torch, time, random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from info_nce import InfoNCE
from torch_geometric.data import Data, Batch
from torch_scatter import scatter
import lightning as L
import torch_geometric.utils.smiles as smiles
import GCL.augmentors as A


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
        #z = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Pool node embeddings to get graph-level embedding
        graph_embedding = global_mean_pool(z, batch)
        
        # Decoder (MLP) step for node reconstruction
        x_hat = self.decoder(z)
        return x_hat, z, graph_embedding  # Return reconstructed node features, latent embeddings, and graph embedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'

info_loss = InfoNCE(negative_mode='unpaired')
aug = A.RandomChoice([ A.RWSampling(num_seeds=1000, walk_length=10),
                                A.NodeDropping(pn=0.1),
                                A.FeatureMasking(pf=0.1),
                                A.EdgeRemoving(pe=0.1)],
                                num_choices=1)


import pdb

def train(train_loader, val_loader, train_graphs, epochs=1):
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
            c_loss = info_loss(g_embed, g_embed_p, g_embed_n) / 3

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
                c_loss = info_loss(g_embed, g_embed_p, g_embed_n) / 3

                val_loss_epoch = r_loss + c_loss
                val_loss += val_loss_epoch.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train C Loss: {c_loss_cum:.4f}, Train R Loss {r_loss_cum:.4f}")
    
    #train_embeddings = extract_latent_representations(model, train_loader)
    #test_embeddings = extract_latent_representations(model, test_loader)
    
    end_time = time.time()
    time_gnn = end_time - start_time
    return model