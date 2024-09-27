import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Tanh for action space
        return action

class DiscriminatorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscriminatorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        validity = torch.sigmoid(self.fc3(x))  # Binary classification
        return validity

def generate_trajectories(policy, env, num_trajectories=10):
    trajectories = []
    for _ in range(num_trajectories):
        state = env.reset()
        done = False
        trajectory = []
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = policy(state_tensor).detach().numpy()[0]
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward, next_state, done))
            state = next_state
        trajectories.append(trajectory)
    return trajectories

def train_gail(policy, discriminator, expert_trajectories, policy_optimizer, discriminator_optimizer, env, epochs=1000):
    for epoch in range(epochs):
        # Generate policy trajectories
        policy_trajectories = generate_trajectories(policy, env)

        # Extract state-action pairs from expert and policy trajectories
        expert_states = torch.FloatTensor([transition[0] for traj in expert_trajectories for transition in traj])
        expert_actions = torch.FloatTensor([transition[1] for traj in expert_trajectories for transition in traj])

        policy_states = torch.FloatTensor([transition[0] for traj in policy_trajectories for transition in traj])
        policy_actions = torch.FloatTensor([transition[1] for traj in policy_trajectories for transition in traj])

        # Train the discriminator
        expert_labels = torch.ones(expert_states.size(0), 1)
        policy_labels = torch.zeros(policy_states.size(0), 1)

        expert_predictions = discriminator(expert_states, expert_actions)
        policy_predictions = discriminator(policy_states, policy_actions)

        loss_discriminator = F.binary_cross_entropy(expert_predictions, expert_labels) + \
                             F.binary_cross_entropy(policy_predictions, policy_labels)

        discriminator_optimizer.zero_grad()
        loss_discriminator.backward()
        discriminator_optimizer.step()

        # Train the policy (maximize log(D(policy)))
        policy_predictions = discriminator(policy_states, policy_actions)
        loss_policy = -torch.log(policy_predictions).mean()

        policy_optimizer.zero_grad()
        loss_policy.backward()
        policy_optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Discriminator Loss = {loss_discriminator.item()}, Policy Loss = {loss_policy.item()}')

class CustomEnvironment:
    def __init__(self, state_dim, action_dim, initial_states):
        self.initial_states = initial_states
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_state = self.reset()
        self.step_count = 0 

    def reset(self):
        # Reset the environment (return initial state)
        self.current_state = self.initial_states[np.random.randint(len(self.initial_states))]
        self.step_count = 0
        return self.current_state

    def step(self, action):
        # Perform action and return next state, reward, and done flag
        next_state = self.current_state + action 
        self.current_state = next_state # Calculate next state based on the action
        self.step_count += 1
        reward = np.random.uniform(0, 1) # Calculate reward
        done = False
        if self.step_count == 3:
            done = False  # Return False if the trajectory length is 3
        elif self.step_count > 3:
            done = True  # End the episode after length exceeds 3
        return next_state, reward, done

    def render(self):
        # Optional: render the environment (e.g., visualization)
        pass



def run():
    state_dim = 64  # Assuming 64-dimensional state
    action_dim = 64  # Assuming 64-dimensional action

    policy = PolicyNetwork(state_dim, action_dim)
    discriminator = DiscriminatorNetwork(state_dim, action_dim)

    policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=3e-4)



    # Assuming you have a custom or pre-defined environment
    embeds1 = np.load(f"/home/ali.lawati/mol-incontext/input/embed/mmcl_attr-chebi-{1}-epochs300-embeds.npz")
    env = CustomEnvironment(64, 64, embeds1['test_pool'])








    # Expert trajectories (collected from expert demonstrations)
    expert_trajectories = generate_trajectories(policy, env)

    # Train the GAIL agent
    train_gail(policy, discriminator, expert_trajectories, policy_optimizer, discriminator_optimizer, env, epochs=1000)


state_dim = 64  # Assuming 64-dimensional state
action_dim = 64  # Assuming 64-dimensional action

policy = PolicyNetwork(state_dim, action_dim)
discriminator = DiscriminatorNetwork(state_dim, action_dim)

policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=3e-4)



run()
seqs = []
refs = []
scores = []
for f in range(1, 4):
    scores1 = np.load(f"/home/ali.lawati/mol-incontext/input/embed/mmcl_attr-chebi-{f}-epochs300-loop.mistral-7B.scores.npy")
    embeds1 = np.load(f"/home/ali.lawati/mol-incontext/input/embed/mmcl_attr-chebi-{f}-epochs300-embeds.npz")
    seqs1 = torch.cat([torch.zeros(3297, 5-f, 64), torch.tensor(embeds1['embeds']).flip(dims=[1])], dim=1)
    refs1 = np.reshape(embeds1['test_pool'], (embeds1['test_pool'].shape[0],1,-1))
    seqs.append(seqs1)
    refs.append(refs1)
    scores.append(scores1)


seqs_tensor = torch.cat(seqs)
scores_tensor = torch.tensor(np.concatenate(scores))
refs_tensor = torch.tensor(np.concatenate(refs))