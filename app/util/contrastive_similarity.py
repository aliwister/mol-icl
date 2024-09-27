import torch

import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
import lightning as L
import GCL.augmentors as A

from datasets import load_dataset


from util.model import smiles2graph

class GraphContrastiveSimilarity(L.LightningModule):
    def __init__(self, input_dim=9, hidden_dim=32, embedding_dim=9, learning_rate=0.001):
        super(GraphContrastiveSimilarity, self).__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.learning_rate = learning_rate
        self.batch_size = 16

        # Define augmentations: node dropout and edge masking
        self.aug = A.RandomChoice([ A.RWSampling(num_seeds=1000, walk_length=10),
                                    A.NodeDropping(pn=0.1),
                                    A.FeatureMasking(pf=0.1),
                                    A.EdgeRemoving(pe=0.1)],
                                    num_choices=1)

    def forward(self, data, augment = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if (augment):
            x, edge_index, _ = self.aug(x, edge_index)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)  # Global mean pooling

    def training_step(self, batch, batch_idx):
        anchor, negative = batch
        anchor_embedding = self(anchor)
        # positive_embedding = self(positive)

        positive_embedding = self(anchor, True)
        negative_embedding = self(negative)
        loss = self.contrastive_loss(anchor_embedding, positive_embedding, negative_embedding)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def contrastive_loss(anchor, positive, negative, margin=1.0):
        distance_positive = torch.sum((anchor - positive) ** 2, dim=1)
        distance_negative = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.mean(torch.clamp(distance_positive - distance_negative + margin, min=0))
        return loss

B_ALPHA = 0.3

def find_similar_to_A_different_from_B(model, A, B, examples_embedding, top_k=5):
    model.eval()
    with torch.no_grad():
        A_embedding = model(Batch.from_data_list([A]))
        B_embedding = model(Batch.from_data_list([B]))
        #examples_embedding = model(Batch.from_data_list(examples))

        similarity_to_A = torch.sum((examples_embedding - A_embedding) ** 2, dim=1)
        similarity_to_B = torch.sum((examples_embedding - B_embedding) ** 2, dim=1)

        # Combine the two criteria: similarity to A and difference from B
        # ADD hypterparameters here.. similarity vs. diversity. We can try tuning this

        combined_score = similarity_to_A - B_ALPHA*similarity_to_B

        # Find the top_k examples with the lowest combined score
        # (most similar to A and most different from B)
        top_indices = torch.argsort(combined_score)[:top_k]

    return top_indices

def load_chebi_dataset(num_examples):
    dataset = load_dataset('liupf/ChEBI-20-MM')
    df_train = dataset['train'].to_pandas()
    df_train = df_train.head(num_examples)
    graphs = [smiles2graph(smiles) for smiles in df_train['SMILES']]
    return graphs, df_train['description'].tolist()

def train_model(model, train_dataset, val_dataset, num_epochs, batch_size, num_gpus):
    trainer = L.Trainer(
        max_epochs=num_epochs,
        devices=num_gpus,
        strategy='ddp' if num_gpus > 1 else None,
    )
    trainer.fit(model, train_dataset, val_dataset)
    return model
