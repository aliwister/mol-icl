import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter

import torch

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder_conv1 = GCNConv(input_dim, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, latent_dim)
        self.decoder_conv1 = GCNConv(latent_dim, hidden_dim)
        self.decoder_conv2 = GCNConv(hidden_dim, input_dim)
    
    def encode(self, x, edge_index):
        x = F.relu(self.encoder_conv1(x, edge_index))
        x = self.encoder_conv2(x, edge_index)
        return x
    
    def decode(self, x, edge_index):
        x = F.relu(self.decoder_conv1(x, edge_index))
        x = self.decoder_conv2(x, edge_index)
        return x
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encode(x, edge_index)
        agg = scatter(z, data.batch, dim=0, reduce="mean")
        x_hat = self.decode(z, edge_index)
        return x_hat, z, agg

def train_autoencoder(model, train_loader, val_loader, epochs, lr=0.01, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x_hat, z, _ = model(batch.to(device))
            loss = F.mse_loss(x_hat, batch.x)  # Reconstruction loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        #print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_hat, z, _ = model(batch.to(device))
                loss = F.mse_loss(x_hat, batch.x)  # Reconstruction loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
import pdb
def extract_latent_representations(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    latent_representations = []
    #pdb.set_trace()
    with torch.no_grad():
        for batch in loader:
            _, _, agg = model(batch.to(device)) # Get the latent representations
            latent_representations.append(agg)
    
    latent_representations = torch.cat(latent_representations, dim=0)
    return latent_representations

def extract_latent_representation(model, data, device):
    model.eval()  # Set the model to evaluation mode
    latent_representations = []
    #pdb.set_trace()
    with torch.no_grad():
        _, _, agg = model(data.to(device))  # Get the latent representations
        latent_representations.append(agg)
    
    #latent_representations = torch.cat(latent_representations[0], dim=0)
    return latent_representations[0]