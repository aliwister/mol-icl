import torch
import torch.nn.functional as F
from itertools import chain


def create_incontext_prompt2(*args):
    if len(args) % 2 == 0:
        raise ValueError("The number of arguments must be odd.")    
    # Initialize an empty string to accumulate the formatted text
    formatted_text = ""
    # Iterate through pairs of arguments
    for i in range(0, len(args)-1, 2):
        input_str = args[i]
        output_str = args[i + 1]
        # Format the input and output into the desired format
        formatted_text += 'Input: {0}\nOutput: {1}\n###\n'.format(input_str, output_str)
    formatted_text += 'Input: {0}\nOutput:'.format(args[-1])
    return formatted_text


def encode(self, batch, lang_model):
    x, edge_index, attention_mask = batch.x, batch.edge_index, batch.attention_mask
    inputs = lang_model(x, output_hidden_states=True, attention_mask=attention_mask)
    hidden_states = inputs.hidden_states
    last_hidden_state = hidden_states[-1]
    denom = torch.sum(attention_mask, -1, keepdim=True)
    feat = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / denom
    feat = feat.to(torch.float32)
    return feat


def get_random_samples(df_train, train_pool, num_examples):
    sampled_data = df_train.sample(n=num_examples)
    sampled_indices = sampled_data.index
    return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description']))), train_pool[sampled_indices]


B_ALPHA = 0.3
def get_samples_new(a_emb, df_train, train_pool, num_examples):
    alpha = B_ALPHA
    selected_indices = []
    combined_score = F.cosine_similarity(a_emb, train_pool)
    for _ in range(num_examples):
        top20 = torch.topk(combined_score, 20)[1]
        idx = next((i for i in top20 if i not in selected_indices), None).item()
        selected_indices.append(idx)
        
        b_emb = train_pool[idx]
        similarity_to_selected =  F.cosine_similarity(b_emb, train_pool)
        combined_score -= alpha * similarity_to_selected

    #print(selected_indices)
    sampled_data = df_train.iloc[selected_indices[::-1]] # here we flip so that best demonstration is concatenated last

    # Trajectories: (Txt[Mol_t], G[Mol_1], Txt[Desc1 + Mol_t], G[Mol_2], Txt[Desc2 + Desc1 + Mol_t], ....)
    # Policy: starting states = Txt[Mol_x], actions = G[Mol_x]  s.t.  Mol_x \in {train, val}, 
    # I need the description of Mol_x if I use it as an action, and have to embed a concatenated string. (Maybe separate with hashes)
    #states = df_train.iloc[selected_indices]['description']

    return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description']))), selected_indices


def get_samples_new_irl(irl, a_emb, df_train, train_pool, num_examples):
    alpha = B_ALPHA
    selected_indices = []
    combined_score = F.cosine_similarity(a_emb, train_pool)
    for _ in range(num_examples):
        top20 = torch.topk(combined_score, 20)[1]
        top20_embed = train_pool[torch.topk(combined_score, 20)[1]]
        
        idx = next((i for i in top20 if i not in selected_indices), None).item()
        scores = [irl(a_emb.unsqueeze(0), t.unsqueeze(0)) for t in top20_embed]
        while (a_emb.unsqueeze(0), top20_embed.unsqueeze(0)) < 0:
            idx = next((i for i in top20 if i not in [selected_indices, idx]), None).item()
        selected_indices.append(idx)
        a_emb = b_emb
        b_emb = train_pool[idx]
        similarity_to_selected =  F.cosine_similarity(b_emb, train_pool) 
        combined_score -= alpha * similarity_to_selected

    #print(selected_indices)
    sampled_data = df_train.iloc[selected_indices[::-1]]
    return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description']))), train_pool[selected_indices]


def get_samples_top(a_emb, df_train, train_pool, num_examples):
    b_idx = torch.topk(F.cosine_similarity(a_emb, train_pool), num_examples).indices.tolist()
    sampled_data = df_train.iloc[b_idx]
    return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description'])))

