from util.contrastive_similarity import GraphContrastiveSimilarity, find_similar_to_A_different_from_B, train_model
from util.graph import mol_to_graph, smiles2graph
from util.model import GraphAutoencoder, extract_latent_representations, train_autoencoder
from util.balanced_kmeans import balanced_kmeans
from torch_geometric.data import Data, Batch
from util.sql_tree import parse_query
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from itertools import chain
import time
import pdb

class ICL:
    def __init__(self, df_train, df_valid, df_test, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.args = args

        self.train_graphs = [smiles2graph(smiles) for smiles in df_train['SMILES']]
        self.train_loader = DataLoader(self.train_graphs, batch_size=16, shuffle=False)

        self.val_graphs = [smiles2graph(smiles) for smiles in df_valid['SMILES']]
        self.val_loader = DataLoader(self.val_graphs, batch_size=16, shuffle=False)

        self.test_graphs = [smiles2graph(smiles) for smiles in df_test['SMILES']]
        self.test_loader = DataLoader(self.test_graphs, batch_size=16, shuffle=False)

        self._initialize_and_train_model()

    def _initialize_and_train_model(self):
        input_dim = 9  # From the smiles2graph function output
        hidden_dim = 32
        embedding_dim = 16
        #pdb.set_trace()
        self.encoder = GraphAutoencoder(input_dim, hidden_dim, embedding_dim).to(self.device)
        start_time = time.time()
        train_autoencoder(self.encoder, self.train_loader, self.val_loader, epochs=self.args.epochs, device=self.device)
        self.train_embeddings = extract_latent_representations(self.encoder, self.train_loader, self.device).cpu()
        self.test_embeddings = extract_latent_representations(self.encoder, self.test_loader, self.device).cpu()
        
        self.cluster_assignments, self.kmeans = balanced_kmeans(self.train_embeddings, 50)
        self.test_labels = self.kmeans.predict(self.test_embeddings)

        end_time = time.time()
        self.time_gnn = end_time - start_time
    
    def get_samples(self, test_index, num):
        idxs = self.get_ae_samples(test_index, num)
        sampled_data = self.df_train.iloc[idxs]
        return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description'])))

    def get_samples_new(self, test_index, num):
        print(test_index)
        idx1 = self.get_ae_samples(test_index, 1)
        idx2 = self.get_samples_sim_A_diff_B(test_index, idx1[0])
        sampled_data = self.df_train.iloc[idx1 + [idx2.item()]]
        return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description'])))

    def get_ae_samples(self, test_index, num):
        test_embedding = self.test_embeddings[test_index]
        test_cluster = self.test_labels[test_index]

        # Filter train_embeddings to only include graphs from the same cluster
        cluster_mask = self.cluster_assignments == test_cluster
        cluster_embeddings = self.train_embeddings[cluster_mask]

        # For the first example (B), use cosine similarity
        similarities = F.cosine_similarity(test_embedding.unsqueeze(0), self.train_embeddings)
        top_indices = torch.topk(similarities, num).indices.tolist()
        return top_indices

    def get_samples_sim_A_diff_B(self, a, b):
        test_cluster = self.test_labels[a]

        # Filter train_embeddings to only include graphs from the same cluster
        cluster_mask = self.cluster_assignments == test_cluster

        # For the second example, use GraphContrastiveSimilarity within the cluster
        model = GraphContrastiveSimilarity(9, 32, 9).to(self.device)
        
        B = self.train_embeddings[b]
        examples = [self.train_graphs[i] for i in range(len(self.train_graphs)) if cluster_mask[i]]
            
        # Train the model
        model = train_model(model, examples, cluster_mask, self.test_graphs[a], self.train_graphs[b], 50, 4, self.device)
        similar_to_A_different_from_B = find_similar_to_A_different_from_B(model, self.test_graphs[a], self.train_graphs[b], examples, self.device, 1)

        return similar_to_A_different_from_B

