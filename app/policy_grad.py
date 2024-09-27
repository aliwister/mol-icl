import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from util.graph import mol_to_graph, smiles2graph_arr
from util.model import GraphAutoencoder, extract_latent_representation, extract_latent_representations, train_autoencoder
from torch_geometric.loader import DataLoader

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output a score for each example

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def extract_state(data):
    return extract_latent_representation(model, data)

def extract_features(data):
    return

def compute_reward(selected_examples, data):
    train_pool
    return

def update_state(state, action_features):
    # Get the features of the selected example
    #selected_example_features = extract_features(data[action])
    
    # Append these features to the current state
    updated_state = torch.cat((state, action_features), dim=0)
    
    return updated_state


# Define the policy gradient method
def policy_gradient(policy_net, optimizer, states, actions, rewards, gamma=0.99):
    policy_net.train()
    optimizer.zero_grad()

    # Convert actions and rewards to tensors
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    
    # Compute discounted rewards
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
    
    # Normalize rewards
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

    # Compute the policy gradient loss
    log_probs = []
    for i in range(len(actions)):
        log_prob = torch.log(policy_net(states[i])[actions[i]])
        log_probs.append(log_prob)
    log_probs = torch.stack(log_probs)
    loss = -torch.sum(log_probs * discounted_rewards)
    
    # Perform backpropagation
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
def train_policy_gradient(policy_net, optimizer, training_data, num_epochs=1000, num_in_context=5):
    for epoch in range(num_epochs):
        states, actions, rewards = [], [], []

        # Sample a batch of data
        for data in training_data:
            state = extract_state(data)  # Define this function based on your data
            
            selected_examples = []
            selected_examples_idx = []
            for _ in range(num_in_context):
                # Get action probabilities for all examples
                action_scores = policy_net(state)
                action_probs = torch.softmax(action_scores.squeeze(-1), dim=-1)
                
                # Sample an action (select an example)
                action = torch.multinomial(action_probs, num_samples=1)
                
                # Add the selected example to the list and update state
                selected_examples.append(train_pool[action].squeeze(1))
                selected_examples_idx.append(action)
                state = update_state(state, train_pool[action].squeeze(1))  # Define this function

                # Store state, action, and a placeholder for the reward
                states.append(state)
                actions.append(action)
                
            # Compute reward for the entire selection
            reward = compute_reward(selected_examples_idx, data)  # Define this function
            
            # Assign the same reward to all actions in the sequence
            rewards.extend([reward] * num_in_context)

        # Update policy network
        loss = policy_gradient(policy_net, optimizer, states, actions, rewards)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')



dataset_name = 'liupf/ChEBI-20-MM'
dataset = load_dataset(dataset_name)
df_train = dataset['train'].to_pandas()
df_valid = dataset['validation'].to_pandas()
df_test = dataset['test'].to_pandas()
train_graphs = df_train.apply(smiles2graph_arr, axis=1)
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=False)

val_graphs = df_valid.apply(smiles2graph_arr, axis=1)
val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)

test_graphs = df_test.apply(smiles2graph_arr, axis=1)
test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

model = GraphAutoencoder(input_dim=9, hidden_dim=16, latent_dim=8)

train_autoencoder(model, train_loader, val_loader, epochs=1)
train_pool = extract_latent_representations(model, train_loader)
test_pool  = extract_latent_representations(model, test_loader)

# Initialize the policy network, optimizer, and training data
input_size = 8  # Example input size, change based on your data
hidden_size = 50
output_size = train_pool.shape[0]

policy_net = PolicyNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

# Example training data (replace with your actual data)
training_data = train_loader

# Train the policy network
train_policy_gradient(policy_net, optimizer, training_data)
