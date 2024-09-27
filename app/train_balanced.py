import numpy as np
import torch
import torch.nn as nn
from itertools import chain
from argparse import ArgumentParser
import pandas as pd
from torch_geometric.loader import DataLoader
from datasets import load_dataset

from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sentence_transformers import SentenceTransformer, util

from util.graph import smiles2graph
from util.model import GraphAutoencoder, extract_latent_representations, train_autoencoder
from balanced_contrastive_similarity import balanced_contrastive_similarity, graph_based_clustering

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_cluster_sizes(cluster_assignments):
    unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
    print("\nCluster sizes:")
    for cluster, count in zip(unique_clusters, counts):
        print(f"Cluster {cluster}: {count} items")
    print(f"Total clusters: {len(unique_clusters)}")
    print(f"Average cluster size: {np.mean(counts):.2f}")
    print(f"Std dev of cluster sizes: {np.std(counts):.2f}")

def run_transformer(args, train_graphs, test_graphs):
    model = GraphAutoencoder(input_dim=9, hidden_dim=16, latent_dim=8)
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    # Train the model
    train_autoencoder(model, train_loader, test_loader, epochs=args.epochs)

    # Extract latent representations
    train_features = extract_latent_representations(model, train_loader)
    test_features = extract_latent_representations(model, test_loader)

    # Perform graph-based clustering
    cluster_assignments = graph_based_clustering(test_graphs, args.num_clusters)
    
    # Print cluster sizes
    print_cluster_sizes(cluster_assignments)

    # Compute balanced contrastive similarity loss
    loss = balanced_contrastive_similarity(test_features, test_graphs, n_clusters=args.num_clusters)

    print(f"\nBalanced Contrastive Similarity Loss: {loss.item()}")

    # You can add more evaluation metrics here

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="liupf/ChEBI-20-MM")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_clusters', type=int, default=5)
    
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()

    train_graphs = df_train['SMILES'].apply(smiles2graph).tolist()
    test_graphs = df_test['SMILES'].apply(smiles2graph).tolist()

    run_transformer(args, train_graphs, test_graphs)