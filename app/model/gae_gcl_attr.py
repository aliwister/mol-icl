import torch, time, random
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from info_nce import InfoNCE
from torch_geometric.data import Data, Batch
from torch_scatter import scatter
import lightning as L
import torch_geometric.utils.smiles as smiles
import GCL.augmentors as A


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_dim)
        self.conv2 = GATConv(hidden_dim, latent_dim)
        self.linear = torch.nn.Linear(latent_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        # Two GCN layers with ReLU activation
        #x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.linear(x)
        return x  # Latent node embeddings

class MLPDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_channels):
        super(MLPDecoder, self).__init__()
        self.x_mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_channels)
        )
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 3)
        )

    def forward(self, z):
        return self.x_mlp(z), self.edge_mlp(z)  # Reconstructed node features

class GAEWithAttributes(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, out_channels, train_graphs):
        super(GAEWithAttributes, self).__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, out_channels)
        self.train_graphs = train_graphs


    def forward(self, x, edge_index, batch, edge_attr):
        # Encoder (GCN) step
        z = self.encoder(x, edge_index, edge_attr)
        
        
        # Pool node embeddings to get graph-level embedding
        graph_embedding = global_mean_pool(z, batch)
        
        # Decoder (MLP) step for node reconstruction
        x_hat, e_hat = self.decoder(z)

        return x_hat, e_hat, z, graph_embedding  # Return reconstructed node features, latent embeddings, and graph embedding

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
    embedding_dim = 128
    
    
    model = GAEWithAttributes(in_channels=input_dim, hidden_dim=hidden_dim, latent_dim=embedding_dim, out_channels=input_dim, train_graphs=train_graphs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    


    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        x_loss_cum = 0.0
        c_loss_cum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, e_hat, z, g_embed  = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            
            pos = aug(batch.x, batch.edge_index, batch.edge_attr)
            x_p, e_p, z_p, g_embed_p = model(pos[0], pos[1], batch.batch, pos[2])

            neg = Batch.from_data_list(random.sample(train_graphs, 16)).to(device)
            x_n, e_n, z_n, g_embed_n = model(neg.x, neg.edge_index, neg.batch, neg.edge_attr)

            x_loss = F.mse_loss(x_hat, batch.x)
            #e_loss = F.mse_loss(e_hat, batch.edge_attr) 
            c_loss = info_loss(g_embed, g_embed_p, g_embed_n) / 4

            train_loss_epoch = x_loss + c_loss #+ e_loss
            train_loss_epoch.backward()
            optimizer.step()
            train_loss += train_loss_epoch.item()

            x_loss_cum += x_loss
            c_loss_cum += c_loss
            #e_loss_cum += e_loss
        
        train_loss /= len(train_loader)
        x_loss_cum /= len(train_loader)
        c_loss_cum /= len(train_loader)
        #e_loss_cum /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_hat, e_hat, z, g_embed  = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                
                pos = aug(batch.x, batch.edge_index)
                x_p, e_p, z_p, g_embed_p = model(pos[0], pos[1], batch.batch, pos[2])

                neg = Batch.from_data_list(random.sample(train_graphs, 16)).to(device)
                x_n, e_n, z_n, g_embed_n = model(neg.x, neg.edge_index, neg.batch, neg.edge_attr)

                x_loss = F.mse_loss(x_hat, batch.x)
                #e_loss = F.mse_loss(e_hat, batch.edge_attr) 
                c_loss = info_loss(g_embed, g_embed_p, g_embed_n) / 4

                val_loss_epoch = x_loss + c_loss #+ e_loss
                val_loss += val_loss_epoch.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train C Loss: {c_loss_cum:.4f}, Train x Loss {x_loss_cum:.4f}")#, , Train e Loss {e_loss_cum:.4f}")
    
    #train_embeddings = extract_latent_representations(model, train_loader)
    #test_embeddings = extract_latent_representations(model, test_loader)
    
    end_time = time.time()
    time_gnn = end_time - start_time
    return model