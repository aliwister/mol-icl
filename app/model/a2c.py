import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(RewardModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output the predicted reward
        )
    
    def forward(self, x):
        return self.network(x)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Common layers for both actor and critic
        self.common = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Output probabilities for each action
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)  # Output state value
        )
    
    def forward(self, x):
        common_out = self.common(x)
        policy = self.actor(common_out)
        value = self.critic(common_out)
        return policy, value


class A2CAgent:
    def __init__(self, input_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma  # Discount factor
        
        self.model = ActorCritic(input_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        policy, _ = self.model(state)
        action_probs = policy.detach().numpy().squeeze()
        action = np.random.choice(len(action_probs), p=action_probs)
        return action
    
    def compute_returns(self, rewards, next_value, done):
        returns = []
        R = next_value if not done else 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns
    
    def update(self, states, actions, rewards, dones, next_state):
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        
        # Compute the next value (value of the next state)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        _, next_value = self.model(next_state)
        next_value = next_value.squeeze().detach()
        
        # Compute the returns
        returns = self.compute_returns(rewards, next_value, dones[-1])
        returns = torch.FloatTensor(returns)
        
        # Update actor-critic network
        self.optimizer.zero_grad()
        
        # Compute loss
        policy, values = self.model(states)
        values = values.squeeze()
        advantages = returns - values
        
        # Actor loss (policy gradient)
        log_probs = torch.log(policy.gather(1, actions.unsqueeze(1)).squeeze())
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = self.criterion(values, returns)
        
        # Total loss
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()



# Example: Training the reward model
def train_reward_model(samples, reward_model, num_epochs=1000, lr=1e-3):
    optimizer = optim.Adam(reward_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for state, reward in samples:  # Assume samples is a list of (state, reward)
            state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            reward = torch.FloatTensor([reward])
            
            optimizer.zero_grad()
            predicted_reward = reward_model(state)
            loss = criterion(predicted_reward, reward)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(samples)}")

def train_a2c(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        states, actions, rewards, dones = [], [], [], []
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
        
        # Update the A2C model at the end of the episode
        agent.update(states, actions, rewards, dones, next_state)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards)}")

class CustomEnvWithRewardModel:
    def __init__(self, vectors, starting_vector, reward_model):
        self.vectors = vectors
        self.starting_vector = starting_vector
        self.reward_model = reward_model
        self.current_state = starting_vector
        self.selected_vectors = []
    
    def reset(self):
        self.current_state = self.starting_vector
        self.selected_vectors = []
        return self.current_state
    
    def step(self, action):
        selected_vector = self.vectors[action]
        self.selected_vectors.append(selected_vector)
        
        # Update state (e.g., concatenate or modify based on selected vectors)
        self.current_state = np.concatenate([self.starting_vector] + self.selected_vectors, axis=-1)
        
        # Predict reward using the trained reward model
        reward = self.calculate_reward()
        done = len(self.selected_vectors) >= 5  # Episode ends after selecting 5 vectors
        return self.current_state, reward, done
    
    def calculate_reward(self):
        # Use the reward model to predict the reward
        state = torch.FloatTensor(self.current_state).unsqueeze(0)  # Add batch dimension
        predicted_reward = self.reward_model(state).item()
        return predicted_reward


# Function to generate random vectors (e.g., starting and selectable vectors)
def generate_vectors(num_vectors, vector_size):
    return torch.randn(num_vectors, vector_size)

# Function to simulate reward for a given combination of vectors (random for this example)
def simulate_reward(selected_vectors):
    return np.random.uniform(0, 1)  # Reward in the range [0, 1]

# Function to generate a batch of sample data
def generate_samples(batch_size, num_vectors, vector_size, max_selections=5):
    samples = []
    starting_vectors = generate_vectors(batch_size, vector_size)  # Batch of starting vectors
    selectable_vectors = generate_vectors(num_vectors, vector_size)  # Pool of vectors to select from
    
    for i in range(batch_size):
        # Randomly select between 1 and max_selections vectors
        num_selected = 1 #np.random.randint(1, 1)
        selected_indices = np.random.choice(num_vectors, num_selected, replace=False)
        selected_vectors = selectable_vectors[selected_indices]
        
        # Concatenate the starting vector with the selected vectors
        state = torch.cat([starting_vectors[i]] + [v for v in selected_vectors], dim=0)
        
        # Simulate a reward for the selected combination
        reward = simulate_reward(selected_vectors)
        
        # Append the sample (state, reward) to the list
        samples.append((state, reward))
    
    return samples

# Generate a batch of sample data
batch_size = 10
num_vectors = 50  # Number of vectors to choose from
vector_size = 64  # Dimensionality of each vector
samples = generate_samples(batch_size, num_vectors, vector_size)

# Display a sample
for idx, (state, reward) in enumerate(samples):
    print(f"Sample {idx + 1} - State shape: {state.shape}, Reward: {reward}")

# Example environment setup (dummy vectors and starting vector)
num_vectors = 50
vector_size = 64
vectors = np.random.randn(num_vectors, vector_size)
starting_vector = np.random.randn(vector_size)

# Initialize environment and agent
input_dim = vector_size * 2  # assuming we concatenate state and selected vector
action_dim = num_vectors
# Assuming vectors, starting_vector, and samples are defined
reward_model = RewardModel(input_dim=vector_size * 2)  # Input size depends on state representation
train_reward_model(samples, reward_model)  # Train the reward model on the samples

# Update the environment to use the trained reward model
env = CustomEnvWithRewardModel(vectors, starting_vector, reward_model)

# Initialize the A2C agent
agent = A2CAgent(input_dim, action_dim)

# Train the agent using the A2C algorithm
train_a2c(env, agent, num_episodes=1000)


