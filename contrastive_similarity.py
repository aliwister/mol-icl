import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
import random
from datasets import load_dataset
from util.graph import smiles2graph

class GraphContrastiveSimilarity(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GraphContrastiveSimilarity, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)  # Global mean pooling

def contrastive_loss(anchor, positive, negative, margin=1.0):
    distance_positive = torch.sum((anchor - positive) ** 2, dim=1)
    distance_negative = torch.sum((anchor - negative) ** 2, dim=1)
    loss = torch.mean(torch.clamp(distance_positive - distance_negative + margin, min=0))
    return loss

def find_similar_to_A_different_from_B(model, A, B, examples, top_k=5):
    model.eval()
    with torch.no_grad():
        A_embedding = model(Batch.from_data_list([A]))
        B_embedding = model(Batch.from_data_list([B]))
        examples_embedding = model(Batch.from_data_list(examples))

        similarity_to_A = torch.sum((examples_embedding - A_embedding) ** 2, dim=1)
        similarity_to_B = torch.sum((examples_embedding - B_embedding) ** 2, dim=1)

        # Combine the two criteria: similarity to A and difference from B
        combined_score = similarity_to_A - similarity_to_B

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

def main():
    # Hyperparameters
    input_dim = 9  # From the smiles2graph function output
    hidden_dim = 32
    embedding_dim = 16
    num_examples = 1000
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 32

    # Load ChEBI dataset
    examples, descriptions = load_chebi_dataset(num_examples)
    A, B = examples[0], examples[1]  # Use the first two molecules as A and B

    # Initialize model and optimizer
    model = GraphContrastiveSimilarity(input_dim, hidden_dim, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for _ in range(num_examples // batch_size):
            optimizer.zero_grad()

            # Sample a batch of examples
            batch_indices = random.sample(range(len(examples)), batch_size)
            batch = [examples[i] for i in batch_indices]

            # Forward pass
            batch_data = Batch.from_data_list(batch)
            batch_embeddings = model(batch_data)

            A_embedding = model(Batch.from_data_list([A]))
            B_embedding = model(Batch.from_data_list([B]))

            # Compute loss
            anchor = A_embedding.repeat(batch_size, 1)
            positive = batch_embeddings
            negative = torch.cat([B_embedding, batch_embeddings[:-1]])
            loss = contrastive_loss(anchor, positive, negative)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (num_examples // batch_size)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    # Find examples similar to A and different from B
    similar_to_A_different_from_B = find_similar_to_A_different_from_B(model, A, B, examples)

    print("A: ", descriptions[0])
    print("B: ", descriptions[1])

    print("\nIndices of examples most similar to A and most different from B:")
    print(similar_to_A_different_from_B)

    print("\nDescriptions of selected molecules:")
    for idx in similar_to_A_different_from_B:
        print(descriptions[idx])

if __name__ == "__main__":
    main()