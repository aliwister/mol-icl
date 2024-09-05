import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter
import lightning as L
import torch_geometric.utils.smiles as smiles
from torch_geometric.data import Data


def smiles2graph(smiles_str):
    data = smiles.from_smiles(smiles_str)
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()
    return Data(x=data.x, edge_index=data.edge_index)

class GraphAutoencoder(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, latent_dim, lr=0.01):
        super(GraphAutoencoder, self).__init__()
        self.encoder_conv1 = GCNConv(input_dim, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, latent_dim)
        self.decoder_conv1 = GCNConv(latent_dim, hidden_dim)
        self.decoder_conv2 = GCNConv(hidden_dim, input_dim)
        self.lr = lr
    
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

    def training_step(self, batch, batch_idx):
        x_hat, _, _ = self(batch)
        loss = F.mse_loss(x_hat, batch.x)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_hat, _, _ = self(batch)
        loss = F.mse_loss(x_hat, batch.x)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def extract_latent_representations(model, loader):
    model.eval()
    latent_representations = []
    with torch.no_grad():
        for batch in loader:
            _, _, agg = model(batch)
            latent_representations.append(agg)
    
    latent_representations = torch.cat(latent_representations, dim=0)
    return latent_representations

def extract_latent_representation(model, data):
    model.eval()
    latent_representations = []
    with torch.no_grad():
        _, _, agg = model(data)
        latent_representations.append(agg)
    
    return latent_representations[0]