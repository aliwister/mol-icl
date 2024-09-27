import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(RewardNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state.squeeze(1), action], dim=-1)
        return self.network(x)

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.reward = RewardNetwork(state_dim, action_dim, hidden_dim)
        self.v = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, next_state, action, gamma):
        r = self.reward(state, action)
        v_s = self.v(state)
        v_ns = self.v(next_state)
        return r + gamma * v_ns - v_s

class AIRL:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3):
        self.discriminator = Discriminator(state_dim, action_dim)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.gamma = gamma
    
    def train(self, expert_states, expert_actions, expert_next_states, 
              policy_states, policy_actions, policy_next_states):
        # Expert
        expert_inputs = (expert_states, expert_next_states, expert_actions)
        expert_outputs = self.discriminator(*expert_inputs, self.gamma)
        expert_loss = -torch.mean(torch.log(torch.sigmoid(expert_outputs) + 1e-8))
        
        # Policy
        policy_inputs = (policy_states, policy_next_states, policy_actions)
        policy_outputs = self.discriminator(*policy_inputs, self.gamma)
        policy_loss = -torch.mean(torch.log(1 - torch.sigmoid(policy_outputs) + 1e-8))
        
        # Total loss
        loss = expert_loss + policy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_reward(self, state, action):
        with torch.no_grad():
            return self.discriminator.reward(state, action).item()

# Example usage
state_dim = 64
action_dim = 64

airl = AIRL(state_dim, action_dim)


def sample_batch(current_states, actions, next_states, batch_size=32):
    # Total number of available transitions
    total_transitions = len(current_states)
    
    # Randomly sample indices for the batch
    sampled_indices = random.sample(range(total_transitions), batch_size)
    
    # Select the corresponding current states, actions, and next states
    batch_current_states = current_states[sampled_indices]
    batch_actions = actions[sampled_indices]
    batch_next_states = next_states[sampled_indices]

    return batch_current_states, batch_actions, batch_next_states


def create_policy_traj(states, actions, B=32, ML=3):
    actions = actions.reshape(-1 ,64)
    states = states.reshape(-1 ,64)
    
    actions_B = actions.unsqueeze(0).expand(B, -1, 64)
    states_B = states.unsqueeze(0).expand(B, -1, 64)

    sample_sizes = torch.randint(1, ML, (B,))
    mask = (torch.arange(ML-1).expand(B, -1) < sample_sizes.unsqueeze(1))
    probabilities = torch.ones(B, actions.shape[0])

    indices = torch.multinomial(probabilities, ML-1, replacement=False)
    actions_s = actions_B[torch.arange(32).unsqueeze(1), indices]
    result = actions_s*mask.unsqueeze(-1).float()
    
    action_add = result.sum(dim=1)
    indices = torch.multinomial(torch.ones(B, actions.shape[0]), 1, replacement=False)
    p_action = actions_B[torch.arange(32).unsqueeze(1), indices].squeeze(1)
    indices = torch.multinomial(torch.ones(B, states.shape[0]), 1, replacement=False)
    p_state = states_B[torch.arange(32).unsqueeze(1), indices].squeeze(1)
    
    p_state = p_state + action_add
    p_next_state = p_action + p_state

    return p_state, p_action, p_next_state


def create_expert_states(refs, seqs):
    combined = torch.cat([refs, seqs], dim=1)  # Shape: [batch, L, dim]
    cumulative_sum = torch.cumsum(combined, dim=1)  # Shape: [batch, L, dim]
    return cumulative_sum

def create_triplets(states, actions):
    next_states = states[:, 1:, :]  # Shape: [BATCH, 3, 64]
    current_states = states[:, :3, :]  # Shape: [BATCH, 3, 64]
    #triplets = torch.stack([current_states, actions, next_states], dim=2)
    #return triplets
    return current_states, actions, next_states

def get_reward_function():
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
    actions = seqs[2][idx]
    states = create_expert_states(refs[2][idx], seqs[2][idx])
    expert_trajs = create_triplets(states, actions)

    actions = seqs[2]
    states = refs[2]
    #policy_trajs = create_policy_traj(states, actions.reshape(-1,64))

    # Training loop (you would need to implement data loading and policy training)
    for epoch in range(1000):
        expert_states, expert_actions, expert_next_states = sample_batch(expert_trajs[0], expert_trajs[1], expert_trajs[2])
        policy_states, policy_actions, policy_next_states = create_policy_traj(states, actions.reshape(-1,64))
        
        loss = airl.train(expert_states, expert_actions, expert_next_states,
                        policy_states, policy_actions, policy_next_states)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
            with torch.no_grad():
                expert_reward = airl.discriminator.reward(expert_states, expert_actions).mean().item()
                random_reward = airl.discriminator.reward(policy_states, policy_actions).mean().item()
                print(f"Mean Expert Reward: {expert_reward:.4f}, Mean Random Reward: {random_reward:.4f}")

    # Get reward for a new state-action pair

    #new_state = torch.randn(1, state_dim)
    #new_action = torch.randn(1, action_dim)
    #reward = airl.get_reward(new_state, new_action)
    #print(f"Reward for new state-action pair: {reward}")
    return airl.get_reward