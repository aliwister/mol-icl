import networkx as nx
from sklearn.cluster import SpectralClustering
import numpy as np

def graph_based_clustering(graphs, n_clusters):
    """
    Perform graph-based clustering on a list of graphs.
    
    Args:
    graphs (list): List of networkx graphs
    n_clusters (int): Number of clusters to create
    
    Returns:
    list: Cluster assignments for each graph
    """
    # Convert graphs to adjacency matrices
    adj_matrices = [nx.to_numpy_array(g) for g in graphs]
    
    # Combine adjacency matrices into a block diagonal matrix
    combined_adj = np.block_diag(*adj_matrices)
    
    # Perform spectral clustering
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    cluster_labels = clustering.fit_predict(combined_adj)
    
    # Split the labels back into separate assignments for each graph
    start = 0
    assignments = []
    for g in graphs:
        end = start + g.number_of_nodes()
        assignments.append(cluster_labels[start:end])
        start = end
    
    return assignments

def get_balanced_clusters(graphs, n_clusters):
    """
    Get balanced clusters based on graph structure.
    
    Args:
    graphs (list): List of networkx graphs
    n_clusters (int): Number of clusters to create
    
    Returns:
    list: Cluster assignments for each graph
    """
    cluster_assignments = graph_based_clustering(graphs, n_clusters)
    
    # Count the number of nodes in each cluster
    cluster_sizes = [sum(len(assign) for assign in cluster_assignments if i in assign) for i in range(n_clusters)]
    
    # Rebalance clusters if necessary
    while max(cluster_sizes) > 1.5 * min(cluster_sizes):
        # Find the largest and smallest clusters
        largest_cluster = cluster_sizes.index(max(cluster_sizes))
        smallest_cluster = cluster_sizes.index(min(cluster_sizes))
        
        # Move a node from the largest to the smallest cluster
        for i, assigns in enumerate(cluster_assignments):
            if largest_cluster in assigns:
                largest_idx = assigns.index(largest_cluster)
                assigns[largest_idx] = smallest_cluster
                cluster_sizes[largest_cluster] -= 1
                cluster_sizes[smallest_cluster] += 1
                break
    
    return cluster_assignments

# Example usage:
# graphs = [list of networkx graphs]
# n_clusters = 5
# balanced_clusters = get_balanced_clusters(graphs, n_clusters)