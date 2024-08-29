import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.cluster import SpectralClustering
import numpy as np

class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

def graph_based_clustering(graphs, n_clusters):
    """
    Perform graph-based clustering on a list of graphs.
    
    Args:
    graphs (list): List of torch_geometric.data.Data objects
    n_clusters (int): Number of clusters to create
    
    Returns:
    list: Cluster assignments for each graph
    """
    # Create a GraphEncoder model
    input_dim = graphs[0].num_node_features
    hidden_dim = 16
    output_dim = 8
    model = GraphEncoder(input_dim, hidden_dim, output_dim)
    
    # Get graph embeddings
    embeddings = []
    for graph in graphs:
        with torch.no_grad():
            embedding = model(Batch.from_data_list([graph]))
        embeddings.append(embedding.numpy())
    
    # Perform spectral clustering on the embeddings
    embeddings = np.vstack(embeddings)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)
    cluster_labels = clustering.fit_predict(embeddings)
    
    return cluster_labels

def balanced_contrastive_similarity(features, graphs, temperature=0.5, base_temperature=0.07, n_clusters=5):
    """
    Compute contrastive similarity with balanced clustering based on graph structure.
    
    Args:
    features (torch.Tensor): Feature embeddings of shape (batch_size, feature_dim)
    graphs (list): List of torch_geometric.data.Data objects representing the structure of each example
    temperature (float): Temperature parameter for similarity scaling
    base_temperature (float): Base temperature for loss calculation
    n_clusters (int): Number of clusters to create
    
    Returns:
    torch.Tensor: Contrastive loss
    """
    device = features.device
    batch_size = features.shape[0]
    
    # Normalize features
    features = F.normalize(features, p=2, dim=1)
    
    # Compute pairwise similarity
    sim_matrix = torch.matmul(features, features.T)
    
    # Get balanced cluster assignments based on graph structure
    cluster_assignments = graph_based_clustering(graphs, n_clusters)
    
    # Convert cluster assignments to mask
    mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    for i in range(batch_size):
        for j in range(batch_size):
            if cluster_assignments[i] == cluster_assignments[j]:
                mask[i][j] = True
    
    # Compute positive and negative similarity
    pos_sim = torch.exp(sim_matrix / temperature)
    neg_sim = torch.exp(sim_matrix / temperature)
    
    # Apply mask to positive similarity
    pos_sim = torch.where(mask, pos_sim, torch.zeros_like(pos_sim))
    
    # Compute loss
    loss = -torch.log(pos_sim.sum(dim=1) / (neg_sim.sum(dim=1) - neg_sim.diag()))
    loss = (temperature / base_temperature) * loss.mean()
    
    return loss

# Example usage:
# features = torch.randn(32, 128)  # 32 examples with 128-dimensional features
# graphs = [Data(...), Data(...), ...]  # List of 32 torch_geometric.data.Data objects
# loss = balanced_contrastive_similarity(features, graphs)