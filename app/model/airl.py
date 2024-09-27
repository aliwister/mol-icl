import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RewardNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        return self.net(state)

class Discriminator(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state):
        return self.net(state)

class AIRL:
    def __init__(self, state_dim, learning_rate=3e-4):
        self.state_dim = state_dim
        
        self.reward_net = RewardNetwork(state_dim)
        self.discriminator = Discriminator(state_dim)
        
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=learning_rate)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
    
    def train(self, expert_trajectories, policy_trajectories, num_iterations, batch_size=128):
        for iteration in range(num_iterations):
            # Sample a batch of expert states
            expert_states = self.sample_batch(expert_trajectories, batch_size)
            
            # Generate random states for comparison
            policy_states = self.sample_batch(policy_trajectories, batch_size)
            
            # Update discriminator
            d_loss = self.update_discriminator(expert_states, policy_states)
            
            # Update reward function
            r_loss = self.update_reward(expert_states, policy_states)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: D_loss: {d_loss:.4f}, R_loss: {r_loss:.4f}")

                with torch.no_grad():
                    expert_reward = self.reward_net(expert_states).mean().item()
                    random_reward = self.reward_net(policy_states).mean().item()
                    print(f"Mean Expert Reward: {expert_reward:.4f}, Mean Random Reward: {random_reward:.4f}")
    
    def update_discriminator(self, expert_states, random_states):
        expert_preds = self.discriminator(expert_states)
        random_preds = self.discriminator(random_states)
        
        loss = -torch.mean(torch.log(expert_preds + 1e-8) + torch.log(1 - random_preds + 1e-8))
        
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()
        
        return loss.item()
    
    def update_reward(self, expert_states, random_states):
        expert_rewards = self.reward_net(expert_states)
        random_rewards = self.reward_net(random_states)
        
        loss = -torch.mean(expert_rewards) + torch.mean(random_rewards)
        
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()
        
        return loss.item()
    
    def sample_batch(self, trajectories, batch_size):
        # Randomly sample states from the expert trajectories
        all_states = trajectories.reshape(-1,64)
        indices = torch.randint(0, all_states.shape[0], (batch_size,))
        return all_states[indices]

    def get_reward(self, state):
        with torch.no_grad():
            return self.reward_net(state).item()

# Usage
state_dim = 64
airl = AIRL(state_dim)

# Load your expert trajectories
# expert_trajectories = [torch.randn(100, state_dim) for _ in range(10)]  # Replace with your actual data
def create_expert_trajs(refs, seqs):
    combined = torch.cat([refs, seqs], dim=1)  # Shape: [batch, L, dim]
    cumulative_sum = torch.cumsum(combined, dim=1)  # Shape: [batch, L, dim]
    return cumulative_sum

seqs = []
refs = []
scores = []
for f in range(1, 4):
    scores1 = np.load(f"/home/ali.lawati/mol-incontext/input/embed/mmcl_attr-chebi-{f}-epochs300-loop.mistral-7B.scores.npy")
    print(len(scores1))
    print(scores1[7], scores1[9], scores1[18], scores1[19], scores1[21], scores1[29], scores1[66])
    embeds1 = np.load(f"/home/ali.lawati/mol-incontext/input/embed/mmcl_attr-chebi-{f}-epochs300-embeds.npz")
    seqs1 = torch.tensor(embeds1['embeds']) #.flip(dims=[1])] #torch.cat([torch.zeros(3297, 5-f, 64), torch.tensor(embeds1['embeds']).flip(dims=[1])], dim=1)
    #print(embeds1['embeds'][7])
    refs1 = np.reshape(embeds1['test_pool'], (embeds1['test_pool'].shape[0],1,-1))
    seqs.append(seqs1)
    refs.append(torch.tensor(refs1))
    scores.append(scores1)

idx = np.where((scores[2] > scores[1]) & (scores[2] > scores[0]) & (scores[2] > .7))[0]
expert_trajs = create_expert_trajs(refs[2][idx], seqs[2][idx])
policy_trajs = create_expert_trajs(refs[2][~idx], seqs[2][~idx])

airl.train(expert_trajs, policy_trajs, num_iterations=10000, batch_size=128)

# After training, you can use the learned reward function
new_state = torch.randn(1, state_dim)
reward = airl.get_reward(new_state)
print(f"Reward for new state: {reward}")