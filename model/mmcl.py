import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit.Chem import DataStructs
from info_nce import InfoNCE
import time
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
info_loss = InfoNCE(negative_mode='paired')

def calculate_similarity(pool, train_pool):
    max = 10
    n_items = len(pool)
    pos = np.zeros((n_items, max), dtype=int)
    pscores = np.zeros((n_items, max), dtype=float)
    neg = np.zeros((n_items, 1000), dtype=int)

    for i, a in enumerate(pool):
        # Compute Tanimoto similarity to all other items
        similarities = np.array([DataStructs.TanimotoSimilarity(t, a) for t in train_pool])
        sorted_indices = np.argsort(-similarities)  # Sort in descending order
        similarities = similarities[sorted_indices]
        pos[i] = sorted_indices[:max]  # Exclude self (index 0)
        pscores[i] = similarities[:max] / similarities[:max].sum()
        neg[i] = sorted_indices[-1000:]
    return pos, pscores, neg

def get_pos_neg_sample_np(pos, pscores, neg):
    pos_idx = np.random.choice(len(pscores), p=pscores)
    neg_idxs = np.random.choice(len(neg), size=40, replace=False)
    return pos[pos_idx], neg[neg_idxs]

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
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.linear(x)
        return x  # Latent node embeddings


class MMCL(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, out_channels):
        super(MMCL, self).__init__()
        self.encoder = GATEncoder(in_channels, hidden_dim, latent_dim)
        #self.decoder = MLPDecoder(latent_dim, hidden_dim, out_channels)
        text_dim = 768
        morgan_dim = 2048
        h_dim = 1024
        self.text2latent = torch.nn.Sequential(
            torch.nn.Linear(text_dim, h_dim),
            torch.nn.ReLU(),  # Activation function between layers
            torch.nn.Linear(h_dim, latent_dim)
        )

    def forward(self, x, edge_index, batch, edge_attr):
        # Encoder (GCN) step
        z = self.encoder(x, edge_index, edge_attr)
        z = global_mean_pool(z, batch)
        return z  # Return reconstructed node features, latent embeddings, and graph embedding

def train(train_loader, val_loader, train_repr, train_morgan, val_morgan, epochs=1, batch_size=64, is_morgan = True):
    input_dim = 9  # From the smiles2graph function output
    hidden_dim = 128
    embedding_dim = 768

    model = MMCL(in_channels=input_dim, hidden_dim=hidden_dim, latent_dim=embedding_dim, out_channels=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if (is_morgan):
        pos_idxs, pos_scores, neg_idxs = calculate_similarity(train_morgan, train_morgan)
        vpos_idxs, vpos_scores, vneg_idxs = calculate_similarity(val_morgan, train_morgan)

    start_time = time.time()
    train_repr.to(device)
    #train_morgan= train_morgan.to(device)
    for epoch in range(epochs):
        model.train() 
        model.text2latent.train()

        train_loss = 0.0
        loss_txt_cum = 0.0
        loss_morgan_cum = 0.0

        for batch in train_loader:
            idx, graph, repr, morgan = batch 
            graph.to(device)
            repr.to(device)
            morgan = morgan.to(device)
            #query_vec = query_vec.to(device)

            optimizer.zero_grad()
            z  = model(graph.x, graph.edge_index, graph.batch, graph.edge_attr)

            if(is_morgan):
                pos_idx, neg_idx = zip(*[get_pos_neg_sample_np(pos_idxs[i], pos_scores[i], neg_idxs[i]) for i in idx])
                pos_idx = list(pos_idx)
                neg_idx = list(neg_idx)
            else:
                weights = torch.ones(len(idx), train_repr.shape[0])
                neg_idx = torch.multinomial(weights, 40, replacement=True)
                pos_idx = idx

            pos = train_repr[pos_idx].squeeze(1)
            neg = train_repr[neg_idx].squeeze(1)
    
            loss_txt = info_loss(z, pos, neg) 
            train_loss_epoch = loss_txt #+ morgan_w*loss_morgan
            
            train_loss_epoch.backward()
            optimizer.step()
            train_loss += train_loss_epoch.item()
            loss_txt_cum += loss_txt.item()
            #loss_morgan_cum += loss_morgan.item()
        
        train_loss /= len(train_loader)
        model.text2latent.eval()

        model.eval()
        val_loss = 0.0
        val_loss_txt_cum = 0.0
        val_loss_morgan_cum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                idx, graph, repr, morgan = batch 
                graph.to(device)
                repr.to(device)#.to(device)
                morgan = morgan.to(device)
                z  = model(graph.x, graph.edge_index, graph.batch, graph.edge_attr)


                if(is_morgan):
                    pos_idx, neg_idx = zip(*[get_pos_neg_sample_np(vpos_idxs[i], vpos_scores[i], vneg_idxs[i]) for i in idx])
                    pos_idx = list(pos_idx)
                    neg_idx = list(neg_idx)
                else:
                    weights = torch.ones(len(idx), train_repr.shape[0])
                    neg_idx = torch.multinomial(weights, 40, replacement=True)
                    pos_idx = idx

                pos = train_repr[pos_idx].squeeze(1)
                neg = train_repr[neg_idx].squeeze(1)
                #mneg = model.morgan2latent_r(mneg_raw).squeeze(1)
                loss_txt = info_loss(z, pos, neg) 
                #loss_morgan = info_loss(z, mpos, mneg) 

                val_loss_epoch =  loss_txt #+ morgan_w*loss_morgan
                val_loss += val_loss_epoch.item()
                val_loss_txt_cum += loss_txt.item()
                #val_loss_morgan_cum += loss_morgan.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f} ({loss_txt_cum:.4f}, {loss_morgan_cum:.4f}), Val Loss: {val_loss:.4f} ({val_loss_txt_cum:.4f}, {val_loss_morgan_cum:.4f})")
    
    end_time = time.time()
    time_gnn = end_time - start_time
    return model