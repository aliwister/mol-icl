
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch_geometric.utils.smiles as smiles

import random
from torch_geometric.data import Data, Batch
from util.scibert import get_batched_text_outputs
from model.mmcl_attr import MultiModalCLAttr
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def smiles2graph(smiles_str):
    data = smiles.from_smiles(smiles_str)
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()
    return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)

def padarray(A, size, value=0):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values = value)

def preprocess_each_sentence(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(
        sentence, truncation=True, max_length=max_seq_len,
        padding='max_length', return_tensors='np')
    
    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()

    sentence_tokens_ids = padarray(input_ids, max_seq_len)
    sentence_masks = padarray(attention_mask, max_seq_len)
    return [sentence_tokens_ids, sentence_masks]

def embed_text(text2latent, text_model, text_tokenizer, text_arr):
    description_tokens_ids, description_masks = prepare_text_tokens(device, text_arr, text_tokenizer, 500) 
    description_output = text_model(input_ids=description_tokens_ids, attention_mask=description_masks)
    description_repr = description_output["pooler_output"]
    description_repr = text2latent(description_repr)
    return description_repr

# This is for BERT
def prepare_text_tokens(device, description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = [o[0] for o in tokens_outputs]
    masks = [o[1] for o in tokens_outputs]
    tokens_ids = torch.Tensor(tokens_ids).long().to(device)
    masks = torch.Tensor(masks).bool().to(device)
    return tokens_ids, masks

def load_local_dataset(dataset_name = 'liupf/ChEBI-20-MM'):
    dataset = load_dataset(dataset_name)
    df_train = dataset['train'].to_pandas()
    df_valid = dataset['validation'].to_pandas()
    df_test = dataset['test'].to_pandas()
    return df_train, df_valid, df_test

def load_model(model_checkpoint = '/home/ali.lawati/mol-incontext/checkpoints/mmcl-300.pt'):
    model = MultiModalCLAttr(9, 32, 64, 9)  # Replace with your model class 
    model.load_state_dict(torch.load('/home/ali.lawati/mol-incontext/checkpoints/mmcl-300.pt', map_location=torch.device(device)))
    model.to(device)
    
    model.requires_grad = False
    model.text2latent.requires_grad = False
    return model, model.text2latent

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
        x = torch.cat([state.squeeze(1), action.squeeze(1)], dim=-1)
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

class AIRL():
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3):
        self.discriminator = Discriminator(state_dim, action_dim)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.gamma = gamma
    
    def train(self, expert_states, expert_actions, expert_next_states, 
              policy_states, policy_actions, policy_next_states):
        
        self.optimizer.zero_grad()
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
        #pdb.set_trace()
        
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        return loss.item()
    
    def get_reward(self, state, action):
        with torch.no_grad():
            return self.discriminator.reward(state, action).item()


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

LAMBDA = .1
def create_expert_trajs(text_model, text_tokenizer, text2latent, states_all, actions_all, init_states_all, init_states_smiles_all, samples, demos, B=32, ML=2):
    
    actions_B = actions_all.unsqueeze(0).expand(B, -1, 64)
    traj_actions = actions_B[torch.arange(B).unsqueeze(1), demos]

    init_states = init_states_all[samples].T.reshape(B,-1)

    action_states = states_all[demos]

    
    action_states = np.char.add.accumulate(np.core.defchararray.add(action_states, ' # '), axis=1)
    action_states[:,0] = init_states[:, 0]

    embed_states =  get_batched_text_outputs(device, np.reshape(action_states, -1), text_tokenizer, text_model, 500, batch_size=512)
    embed_states = text2latent(embed_states).reshape(B,-1, 64)
    embed_states = np.concatenate((init_states, embed_states + LAMBDA * init_states), axis=1)

    #get_batched_text_outputs(np.reshape(action_states, -1)) # torch.tensor(embed_text(text2latent, text_model, text_tokenizer, np.reshape(action_states, -1))).reshape(B,-1, 64)

    # Create triplets using tensor indexing
    triplet_states = embed_states[:,:-1,:]  # Select all states except the last
    triplet_actions = traj_actions       # Actions are the same
    triplet_next_states = embed_states[:,1:,:]  # Select all states except the first

    # Stack the triplets into a single tensor (optional)
    triplets = torch.stack((triplet_states, triplet_actions, triplet_next_states), dim=2)
    triplets = triplets.reshape(-1,3,64)
    triplets = triplets[torch.multinomial(torch.ones(triplets.shape[0]), B, replacement=False)]
    return torch.split(triplets,1, dim=1)

def create_policy_traj(text_model, text_tokenizer, text2latent, states_all, actions_all, init_states_all, indexes=None, B=32, ML=2):
    action_indices = torch.multinomial(torch.ones(B, actions_all.shape[0]), ML-1, replacement=False)
    actions_B = actions_all.unsqueeze(0).expand(B, -1, 64)
    traj_actions = actions_B[torch.arange(B).unsqueeze(1), action_indices]

    init_states = init_states_all[np.random.choice(len(init_states_all), B)].T.reshape(B,-1)

    action_states = states_all[action_indices]

    action_states = np.concatenate((init_states, action_states), axis=1)
    action_states = np.char.add.accumulate(np.core.defchararray.add(action_states, ' # '), axis=1)
    action_states[:,0] = init_states[:, 0]

    embed_states =  torch.tensor(embed_text(text2latent, text_model, text_tokenizer, np.reshape(action_states, -1))).reshape(B,-1, 64)

    # Create triplets using tensor indexing
    triplet_states = embed_states[:,:-1,:]  # Select all states except the last
    triplet_actions = traj_actions       # Actions are the same
    triplet_next_states = embed_states[:,1:,:]  # Select all states except the first

    # Stack the triplets into a single tensor (optional)
    triplets = torch.stack((triplet_states, triplet_actions, triplet_next_states), dim=2)
    triplets = triplets.reshape(-1,3,64).detach().clone()
    triplets = triplets[torch.multinomial(torch.ones(triplets.shape[0]), B, replacement=False)]
    return torch.split(triplets,1, dim=1)

def get_reward_function():

    cache_dir = '/home/ali.lawati/mol-incontext/data/pretrained_SciBERT'
    text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=cache_dir)
    text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=cache_dir).to(device)

    # Example usage
    state_dim = 64
    action_dim = 64

    airl = AIRL(state_dim, action_dim)
    airl.discriminator.to(device)

    model, text2latent = load_model()
    df_train, df_valid, df_test = load_local_dataset()
    val_graphs = [smiles2graph(smiles) for smiles in df_valid['SMILES']]
    train_graphs = [smiles2graph(smiles) for smiles in df_train['SMILES']]
    train_batch = Batch.from_data_list(train_graphs).to(device)
    valid_batch  = Batch.from_data_list(val_graphs).to(device)
    train_pool = model(train_batch.x, train_batch.edge_index, train_batch.batch, train_batch.edge_attr).detach().clone()
    valid_pool = model(valid_batch.x, valid_batch.edge_index, valid_batch.batch, valid_batch.edge_attr).detach().clone()

    actions_all = torch.cat((train_pool, valid_pool)).detach().clone()
    init_states_all = torch.cat((valid_pool, train_pool)).detach().clone() 
    init_states_smiles_all = np.concatenate((df_valid['SMILES'].values, df_train['SMILES'].values)) # I'm doing this opposite since the prompt files use valid dataset, and so i can index it directly
    states_all = np.concatenate((df_train['description'].values, df_valid['description'].values))

    metas = []
    scores = []
    for f in range(2, 3):
        scores1 = np.load(f"/home/ali.lawati/mol-incontext/input/embed/mmcl_attr-chebi-{f}-epochs300-new.mistral-7B.scores.npy")
        meta = np.load('/home/ali.lawati/mol-incontext/input/embed/mmcl_attr-chebi-2-epochs300-meta.npy')
        metas.append(meta)
        scores.append(scores1)

    samples = np.where((scores[0] > .8))[0]
    demos = metas[0][samples]
    expert_trajs = create_expert_trajs(text_model, text_tokenizer, text2latent, states_all, actions_all, init_states_all, init_states_smiles_all, samples, demos, len(samples))
    #expert_trajs.detach().clone()
    #

    # Training loop (you would need to implement data loading and policy training)
    for epoch in range(1000):
        expert_states, expert_actions, expert_next_states = sample_batch(expert_trajs[0], expert_trajs[1], expert_trajs[2])
        policy_states, policy_actions, policy_next_states = create_policy_traj(text_model, text_tokenizer, text2latent, states_all, actions_all, init_states_all)

        expert_states = expert_states.to(device)
        expert_actions = expert_actions.to(device)
        expert_next_states = expert_next_states.to(device)

        policy_states = policy_states.to(device)
        policy_actions = policy_actions.to(device)
        policy_next_states = policy_next_states.to(device)
        
        loss = airl.train(expert_states, expert_actions, expert_next_states,
                        policy_states, policy_actions, policy_next_states)
        
        if epoch % 20 == 0:
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

    for f in range(2, 3):
        #scores1 = np.load(f"/home/ali.lawati/mol-incontext/input/embed/mmcl_attr-chebi-{f}-epochs300-new-test.mistral-7B.scores.npy")
        meta = np.load('/home/ali.lawati/mol-incontext/input/embed/mmcl_attr-chebi-2-epochs300-new-test-meta.npy')
        metas.append(meta)
        #scores.append(scores1)

    samples = np.arange(0, 10000)
    demos = metas[0][samples]
    expert_trajs = create_expert_trajs(text_model, text_tokenizer, text2latent, states_all, actions_all, init_states_all, samples, demos, len(samples))
    x,y,z  = expert_trajs
    rewards = [airl.get_reward(a,b) for (a,b) in zip(x,y)]
    print(len(rewards))
    np.save(f"/home/ali.lawati/mol-incontext/input/embed/mmcl_attr-chebi-2-epochs300-new-test.mistral-7B.rewards.npy", np.array(rewards))
    #pdb.set_trace()

    return airl

airl = get_reward_function()
torch.save(airl.discriminator.state_dict(), f"/home/ali.lawati/mol-incontext/checkpoints/airl.pt")
new_state = torch.randn(1, 64)
new_action = torch.randn(1, 64)
reward = airl.get_reward(new_state, new_action)

metas = []
scores = []
