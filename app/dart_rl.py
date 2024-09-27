import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# Define a simple policy network to select in-context examples
class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# Function to compute the reward based on task performance
def get_reward(selected_examples, target):
    # Simulate a simple reward function (accuracy or task performance)
    # In a real scenario, you'd compute this based on the model's task performance after using selected examples.
    # For now, we'll use a placeholder reward based on how many selected examples match the target.
    correct = random(1,3) #sum([1 if example.item() == target.item() else 0 for example in selected_examples])
    return correct / len(selected_examples)

# Training loop for DART (combines supervised and RL)
def train_dart(policy_net, optimizer, suboptimal_demonstrations, targets, num_epochs=100, rl_start_epoch=50):
    policy_net.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_reward = 0
        
        for i, (demonstrations, target) in enumerate(zip(suboptimal_demonstrations, targets)):
            # Convert demonstrations to a tensor (input features to the policy network)
            demonstrations_tensor = torch.tensor(demonstrations, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)
            
            # Policy network output: probabilities of selecting each in-context example
            probs = policy_net(demonstrations_tensor)
            
            # Sample from the policy to get selected examples
            selected_indices = torch.multinomial(probs, num_samples=3, replacement=False)
            selected_examples = demonstrations_tensor[selected_indices]
            
            # Compute the reward based on task performance (e.g., classification accuracy)
            reward = get_reward(selected_examples, target)
            total_reward += reward

            if epoch < rl_start_epoch:
                # Supervised learning phase: minimize task error
                # Placeholder task loss: Mean squared error with the target
                supervised_loss = F.mse_loss(selected_examples, target_tensor.repeat(selected_examples.shape[0], 1))
                total_loss += supervised_loss.item()

                # Backpropagation
                optimizer.zero_grad()
                supervised_loss.backward()
                optimizer.step()
            else:
                # Reinforcement learning phase: maximize reward
                # Policy gradient (REINFORCE)
                log_probs = torch.log(probs[selected_indices])
                rl_loss = -reward * log_probs.mean()
                total_loss += rl_loss.item()

                # Backpropagation
                optimizer.zero_grad()
                rl_loss.backward()
                optimizer.step()

        avg_loss = total_loss / len(suboptimal_demonstrations)
        avg_reward = total_reward / len(suboptimal_demonstrations)

        if epoch < rl_start_epoch:
            print(f'Epoch [{epoch+1}/{num_epochs}] - Supervised Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}] - RL Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}')

# Example usage
input_dim = 10  # Number of features per example
hidden_dim = 20
output_dim = input_dim  # Same as input_dim, representing the number of in-context examples to choose from

# Instantiate the policy network and optimizer
policy_net = PolicyNet(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

# Example suboptimal demonstrations (randomly generated for simplicity)
suboptimal_demonstrations = [torch.rand(input_dim) for _ in range(100)]  # 100 samples of suboptimal demonstrations
targets = [torch.rand(input_dim) for _ in range(100)]  # Corresponding targets

# Train the DART model
train_dart(policy_net, optimizer, suboptimal_demonstrations, targets, num_epochs=100, rl_start_epoch=50)
