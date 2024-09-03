from util.contrastive_similarity import GraphContrastiveSimilarity, find_similar_to_A_different_from_B, train_model
from util.graph import mol_to_graph, smiles2graph
from util.model import GraphAutoencoder, extract_latent_representations, train_autoencoder
from util.balanced_kmeans import balanced_kmeans
from torch_geometric.data import Data, Batch
from util.sql_tree import parse_query
from torch_geometric.loader import DataLoader
import torch
import lightning as L
import torch.nn.functional as F
from itertools import chain
import time
import pdb
import random

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

        if(args.method == "icl-new"):
            self.model = self._train_icl_new(args.limit)
    

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

    def _train_icl_new(self, limit, devices=1):
        if (limit > 0):
            df_test_limited = self.df_test[:limit]
        else:
            df_test_limited = self.df_test

        training_set = []
        for i in range(0, len(df_test_limited)):
            a = i
            b = self.get_ae_samples(a, 1)[0]

            test_cluster = self.test_labels[a]

            # Filter train_embeddings to only include graphs from the same cluster
            cluster_mask = self.cluster_assignments == test_cluster
            examples = [self.train_graphs[i] for i in range(len(self.train_graphs)) if cluster_mask[i]]
            for i in range(20):
                training_set.append((self.test_graphs[a], self.train_graphs[b], random.sample(examples,1)[0]))

        # Create train_loader from training_set
        def collate_fn(batch):
            test_graphs, train_graphs, example = zip(*batch)
            return Batch.from_data_list(test_graphs), Batch.from_data_list(train_graphs), Batch.from_data_list(example)

        train_loader = DataLoader(training_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
        
        model = GraphContrastiveSimilarity()
        trainer = L.Trainer(
            max_epochs=100,
            devices=devices
        )
        trainer.fit(model, train_loader)
        return model
    
    def get_samples(self, test_index, num):
        idxs = self.get_ae_samples(test_index, num)
        sampled_data = self.df_train.iloc[idxs]
        return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description'])))

    def get_samples_new(self, test_index, num):
        idx1 = self.get_ae_samples(test_index, 1)
        idx2 = self.get_samples_sim_A_diff_B(test_index, idx1[0])
        sampled_data = self.df_train.iloc[idx1 + [idx2.item()]]
        return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description'])))

    def get_ae_samples(self, test_index, num):
        test_embedding = self.test_embeddings[test_index]
        test_cluster = self.test_labels[test_index]

        # Filter train_embeddings to only include graphs from the same cluster
        cluster_mask = self.cluster_assignments == test_cluster

        # For the first example (B), use cosine similarity
        similarities = F.cosine_similarity(test_embedding.unsqueeze(0), self.train_embeddings)
        top_indices = torch.topk(similarities, num).indices.tolist()
        return top_indices

    def get_samples_sim_A_diff_B(self, a, b):
        test_cluster = self.test_labels[a]

        # Filter train_embeddings to only include graphs from the same cluster
        cluster_mask = self.cluster_assignments == test_cluster
        examples = [self.train_graphs[i] for i in range(len(self.train_graphs)) if cluster_mask[i]]
        similar_to_A_different_from_B = find_similar_to_A_different_from_B(self.model, self.test_graphs[a], self.train_graphs[b], examples, 1)
        return similar_to_A_different_from_B
