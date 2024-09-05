from util.contrastive_similarity import GraphContrastiveSimilarity, find_similar_to_A_different_from_B, train_model
import torch_geometric.utils.smiles as smiles
from util.model import GraphAutoencoder, extract_latent_representations, smiles2graph

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch
import lightning as L
import torch.nn.functional as F
from itertools import chain
import time

class MyDataset:
    def __getitem__(self, idx):
        # Return a single graph as a Data object
        return Data(x=..., edge_index=..., y=...)

    def __len__(self):
        return len(self.data)


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

        self.autoencoder = self._initialize_and_train_model(args.gpus, args.epochs)

        if(args.method == "icl-new"):
            self.gcl_model = self._train_icl_new(args.limit, args.gpus, args.epochs)
        

    def _initialize_and_train_model(self, devices=1, epochs=100):
        input_dim = 9  # From the smiles2graph function output
        hidden_dim = 32
        embedding_dim = 16
        #pdb.set_trace()
        model = GraphAutoencoder(input_dim, hidden_dim, embedding_dim).to(self.device)
        start_time = time.time()
        #train_autoencoder(self.encoder, self.train_loader, self.val_loader, epochs=self.args.epochs, device=self.device)
        
        trainer = L.Trainer(max_epochs=epochs, devices=devices)
        trainer.fit(model, self.train_loader, self.val_loader)
        
        self.train_embeddings = extract_latent_representations(model, self.train_loader)
        self.test_embeddings = extract_latent_representations(model, self.test_loader)
        
        # self.cluster_assignments, self.kmeans = balanced_kmeans(self.train_embeddings, 50)
        # self.test_labels = self.kmeans.predict(self.test_embeddings)

        end_time = time.time()
        self.time_gnn = end_time - start_time
        return model

    def _train_icl_new(self, limit, devices=1, epochs=100):
        if (limit > 0):
            df_test_limited = self.df_test[:limit]
        else:
            df_test_limited = self.df_test

        training_set = []
        for i in range(0, len(df_test_limited)):
            a = i
            b = self.get_ae_samples(a, 1)[0]

            # test_cluster = self.test_labels[a]

            # Filter train_embeddings to only include graphs from the same cluster
            # cluster_mask = self.cluster_assignments == test_cluster
            # examples = [self.train_graphs[i] for i in range(len(self.train_graphs)) if cluster_mask[i]]
            for i in range(20):
                training_set.append((self.test_graphs[a], self.train_graphs[b]))

        # Create train_loader from training_set
        def collate_fn(batch):
            test_graphs, train_graphs, example = zip(*batch)
            return Batch.from_data_list(test_graphs), Batch.from_data_list(train_graphs) # Batch.from_data_list(example)

        train_loader = DataLoader(training_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
        
        model = GraphContrastiveSimilarity()
        trainer = L.Trainer(
            max_epochs=epochs,
            devices=devices
        )
        trainer.fit(model, train_loader)

        self.examples_embeddings = model(Batch.from_data_list(self.train_graphs))
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

        similarities = F.cosine_similarity(test_embedding.unsqueeze(0), self.train_embeddings)
        top_indices = torch.topk(similarities, num).indices.tolist()
        return top_indices

    def get_samples_sim_A_diff_B(self, a, b):
        similar_to_A_different_from_B = find_similar_to_A_different_from_B(self.gcl_model, self.test_graphs[a], self.train_graphs[b], self.examples_embeddings, 1)
        return similar_to_A_different_from_B
