import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal


import sys


from airl2 import AIRL

# Previous AIRL implementation (RewardNetwork, Discriminator, AIRL classes)
# ... [Keep the previous implementation as is] ...

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        mean = self.network(state)
        std = self.log_std.exp()
        return Normal(mean, std)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
    
    def get_action(self, state):
        with torch.no_grad():
            dist = self.actor(state)
            action = dist.sample()
        return action
    
    def update(self, states, actions, rewards, next_states, dones):
        # Convert to tensor
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(-1)
        
        # Compute advantage
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            advantages = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Get old action probabilities
        old_dist = self.actor(states)
        old_log_probs = old_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        for _ in range(self.epochs):
            # Actor loss
            dist = self.actor(states)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            value_pred = self.critic(states)
            value_loss = nn.MSELoss()(value_pred, rewards + self.gamma * next_values * (1 - dones))
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

class AIRLPPO:
    def __init__(self, state_dim, action_dim, gamma=0.99, airl_lr=1e-3, ppo_lr=3e-4):
        self.airl = AIRL(state_dim, action_dim, gamma, airl_lr)
        self.ppo = PPO(state_dim, action_dim, ppo_lr, gamma)
    
    def train(self, expert_states, expert_actions, expert_next_states, env, num_episodes=1000):
        for episode in range(num_episodes):
            states, actions, rewards, next_states, dones = [], [], [], [], []
            state = env.reset()
            done = False
            
            while not done:
                action = self.ppo.get_action(torch.FloatTensor(state))
                next_state, _, done, _ = env.step(action.numpy())
                reward = self.airl.get_reward(torch.FloatTensor(state), action)
                
                states.append(state)
                actions.append(action.numpy())
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                state = next_state
            
            # Update AIRL
            policy_states = torch.FloatTensor(states)
            policy_actions = torch.FloatTensor(actions)
            policy_next_states = torch.FloatTensor(next_states)
            airl_loss = self.airl.train(expert_states, expert_actions, expert_next_states,
                                        policy_states, policy_actions, policy_next_states)
            
            # Update PPO
            self.ppo.update(states, actions, rewards, next_states, dones)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, AIRL Loss: {airl_loss}")

# Example usage
state_dim = 4
action_dim = 2

airlppo = AIRLPPO(state_dim, action_dim)

# You would need to implement your environment and data loading
class DummyEnv:
    def reset(self):
        return np.random.randn(state_dim)
    
    def step(self, action):
        return np.random.randn(state_dim), 0, np.random.rand() > 0.99, {}

env = DummyEnv()

# Load expert data (replace with your actual data loading)
expert_states = torch.randn(1000, state_dim)
expert_actions = torch.randn(1000, action_dim)
expert_next_states = torch.randn(1000, state_dim)

# Train
airlppo.train(expert_states, expert_actions, expert_next_states, env)