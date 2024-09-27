import torch
import torch.nn as nn
import torch.optim as optim

class ContrastiveSimilarity(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(ContrastiveSimilarity, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.embedding(x)

def contrastive_loss(anchor, positive, negative, margin=1.0):
    distance_positive = torch.sum((anchor - positive) ** 2, dim=1)
    distance_negative = torch.sum((anchor - negative) ** 2, dim=1)
    loss = torch.mean(torch.clamp(distance_positive - distance_negative + margin, min=0))
    return loss

def find_similar_different(model, A, B, examples, top_k=5):
    with torch.no_grad():
        A_embedding = model(A)
        B_embedding = model(B)
        examples_embedding = model(examples)

        similarity_to_A = torch.sum((examples_embedding - A_embedding) ** 2, dim=1)
        similarity_to_B = torch.sum((examples_embedding - B_embedding) ** 2, dim=1)

        most_similar_to_A = torch.argsort(similarity_to_A)[:top_k]
        most_different_to_B = torch.argsort(similarity_to_B, descending=True)[:top_k]

    return most_similar_to_A, most_different_to_B

def main():
    # Hyperparameters
    input_dim = 10
    embedding_dim = 32
    num_examples = 1000
    num_epochs = 100
    learning_rate = 0.001

    # Generate random data for demonstration
    A = torch.randn(1, input_dim)
    B = torch.randn(1, input_dim)
    examples = torch.randn(num_examples, input_dim)

    # Initialize model and optimizer
    model = ContrastiveSimilarity(input_dim, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        A_embedding = model(A)
        B_embedding = model(B)
        examples_embedding = model(examples)

        # Compute loss
        positive_samples = examples[torch.randint(0, num_examples, (1,))]
        negative_samples = examples[torch.randint(0, num_examples, (1,))]
        loss = contrastive_loss(A_embedding, model(positive_samples), model(negative_samples))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Find similar and different examples
    similar_to_A, different_to_B = find_similar_different(model, A, B, examples)

    print("\nIndices of examples most similar to A:")
    print(similar_to_A)
    print("\nIndices of examples most different to B:")
    print(different_to_B)

if __name__ == "__main__":
    main()