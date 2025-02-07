import torch, os
import torch.nn.functional as F
from itertools import chain
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch_geometric.utils.smiles as smiles
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors

def smiles2graph(smiles_str):
    data = smiles.from_smiles(smiles_str)
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()
    return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)

def create_input_file(prompts, refs, num_examples, method, task_name):
    data_dict = {
        'prompt': np.squeeze(prompts),
        'ref': refs
    }
    df = pd.DataFrame(data_dict)
    output_dir = f"./input/{task_name}/{method}"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/{method}-{num_examples}.csv", index=False)

def create_embed_file(args, embeds, test_pool, num_examples):
    np.savez_compressed(f"./input/embed/{args.method}-{args.dataset}-{num_examples}-epochs{args.epochs}-embeds.npz", embeds=embeds, test_pool=test_pool)

def create_meta_file(args, indices, num_examples, label):
    np.save(f"./input/embed/{args.method}-{args.dataset}-{num_examples}-epochs{args.epochs}-{label}-meta.npy", np.array(indices))

def parse_method(method, checkpoint):
    if (len(checkpoint) > 0):
        if(checkpoint.find("[[{'G': 0, 'M': 1}, {'T': 1, 'M': 0, 'K': 0}]]")):
            method = "morgan-text"
        elif(checkpoint.find("[[{'G': 1, 'M': 0}, {'T': 1, 'M': 0, 'K': 0}]]")):
            method = "graph-text"
        elif(checkpoint.find("[[{'G': 1, 'M': 0}, {'T': 1, 'M': 1, 'K': 0}]]")):
            method = "graph-morgan+text"

        if(checkpoint.find("chebi")):
            method = method + "-chebi"
        if(checkpoint.find("scibert")):
            method = method+"-scibert"
    return method

def create_incontext_prompt(*args, input_label = "Input", output_label="Output", predict="Caption"):
    if len(args) % 2 == 0:
        raise ValueError("The number of arguments must be odd.")    
    # Initialize an empty string to accumulate the formatted text
    formatted_text = f"You are an expert chemist. Given the molecular SMILES, your task is to predict the {predict} using your experienced molecular knowledge.\n###\n"
    # Iterate through pairs of arguments
    for i in range(0, len(args)-1, 2):
        input_str = args[i]
        output_str = args[i + 1]
        # Format the input and output into the desired format
        formatted_text += f"{input_label}: {input_str}\n{output_label}: {output_str}\n###\n"
    formatted_text += f"{input_label}: {args[-1]}\n{output_label}:"
    return formatted_text

def create_incontext_prompt2(*args, input_label = "Input", output_label="Output"):
    if len(args) % 2 == 0:
        raise ValueError("The number of arguments must be odd.")    
    # Initialize an empty string to accumulate the formatted text
    formatted_text = ""
    # Iterate through pairs of arguments
    for i in range(0, len(args)-1, 2):
        input_str = args[i]
        output_str = args[i + 1]
        # Format the input and output into the desired format
        formatted_text += f"{input_label}: {input_str}\n{output_label}: {output_str}\n###\n"
    formatted_text += f"{input_label}: {args[-1]}\n{output_label}:"
    return formatted_text

def create_incontext_prompt_binary(*args, input_label = "Input", output_label="Output", description="Caption"):
    if len(args) % 2 == 0:
        raise ValueError("The number of arguments must be odd.")    
    # Initialize an empty string to accumulate the formatted text
    formatted_text = f"You are an expert chemist. Given the molecular SMILES, your task is to predict if the molecule {description} using your experienced molecular knowledge. Responsd with Yes or No only.\n###\n"
    # Iterate through pairs of arguments
    for i in range(0, len(args)-1, 2):
        input_str = args[i]
        output_str = args[i + 1]
        # Format the input and output into the desired format
        formatted_text += f"{input_label}: {input_str}\n{output_label}: {output_str}\n###\n"
    formatted_text += f"{input_label}: {args[-1]}\n{output_label}:"
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
def generate_scaffolds(df, smiles_label):
    target_smiles = df[smiles_label].tolist()
    target_mol = [Chem.MolFromSmiles(smile) for smile in target_smiles]
    target_mol = list(filter(None, target_mol))
    target_scaffold = [MurckoScaffold.GetScaffoldForMol(mol) for mol in target_mol]
    target_fp = [rdMolDescriptors.GetMorganFingerprint(mol, 2) for mol in target_scaffold] 
    return target_fp

def get_random_samples(df_train, num_examples, label_source, label_target, bool2YesNo=False):
    sampled_data = df_train.sample(n=num_examples)
    sampled_indices = sampled_data.index
    class_label = sampled_data[label_target]
    if bool2YesNo:
        class_label = ["Yes" if i == 1 else "No" for i in class_label]
    return list(chain.from_iterable(zip(sampled_data[label_source], class_label)))

def get_scaffold_samples(a_emb, df_train, train_pool, num_examples, label_source, label_target, bool2YesNo=False):
    numbers = [DataStructs.TanimotoSimilarity(t, a_emb) for t in train_pool]
    selected_indices = sorted(range(len(numbers)), key=lambda i: numbers[i], reverse=True)[:num_examples]
    sampled_data = df_train.iloc[selected_indices[::-1]]
    class_label = sampled_data[label_target] 
    if bool2YesNo:
        class_label = ["Yes" if i == 1 else "No" for i in class_label]
    return list(chain.from_iterable(zip(sampled_data[label_source], class_label)))

def get_scaffold_samples2(a_emb1, a_emb2, df_train, train_pool1, train_pool2, num_examples, label_source='SMILES', label_target='description', bool2YesNo=True):
    
    numbers1 = [DataStructs.TanimotoSimilarity(t, a_emb1) for t in train_pool1]
    numbers2 = [DataStructs.TanimotoSimilarity(t, a_emb2) for t in train_pool2]

    selected_indices = sorted(range(len(numbers1)), key=lambda i: numbers1[i]+numbers2[i], reverse=True)[:num_examples]
    sampled_data = df_train.iloc[selected_indices[::-1]]
    class_label = sampled_data[label_target] 
    if bool2YesNo:
        class_label = ["Yes" if i == 1 else "No" for i in class_label]
    return list(chain.from_iterable(zip(sampled_data[label_source], class_label)))

def get_random_samples_no_logits(df_train, num_examples, label_source, label_target):
    sampled_data = df_train.sample(n=num_examples)
    return list(chain.from_iterable(zip(sampled_data[label_source], sampled_data[label_target])))

def get_samples_new(a_emb, df_train, train_pool, num_examples, label_source = 'SMILES', label_target = 'description', bool2YesNo=False, llambda=0.3):

    selected_indices = []
    combined_score = F.cosine_similarity(a_emb, train_pool)
    for _ in range(num_examples):
        top20 = torch.topk(combined_score, 20)[1]
        idx = next((i for i in top20 if i not in selected_indices), None).item()
        selected_indices.append(idx)
        
        b_emb = train_pool[idx]
        similarity_to_selected =  F.cosine_similarity(b_emb, train_pool)
        combined_score -= llambda * similarity_to_selected

    #print(selected_indices)
    selected_indices.sort(key=lambda i: combined_score[i].item(), reverse=True)
    sampled_data = df_train.iloc[selected_indices[::-1]] # here we flip so that best demonstration is concatenated last

    class_label = sampled_data[label_target] 
    if bool2YesNo:
        class_label = ["Yes" if i == 1 else "No" for i in class_label]
    return list(chain.from_iterable(zip(sampled_data[label_source], class_label))), selected_indices

def get_samples_new2(a_emb1, a_emb2, df_train, train_pool1, train_pool2, num_examples, label_source='SMILES', label_target='description', bool2YesNo=True, llambda=0.3):
    selected_indices = []
    
    # Compute cosine similarity scores separately for each embedding and pool
    combined_score1 = F.cosine_similarity(a_emb1, train_pool1)
    combined_score2 = F.cosine_similarity(a_emb2, train_pool2)
    
    # Combine scores by summing them (or other method depending on desired weighting)
    combined_score = combined_score1 + combined_score2
    
    for _ in range(num_examples):
        top20 = torch.topk(combined_score, 20)[1]
        idx = next((i for i in top20 if i not in selected_indices), None).item()
        selected_indices.append(idx)
        
        # Update scores to penalize previously selected indices
        b_emb1 = train_pool1[idx]
        b_emb2 = train_pool2[idx]
        
        similarity_to_selected1 = F.cosine_similarity(b_emb1, train_pool1)
        similarity_to_selected2 = F.cosine_similarity(b_emb2, train_pool2)
        
        # Adjust combined score based on similarity to selected indices
        combined_score -= llambda * (similarity_to_selected1 + similarity_to_selected2)

    # Select data samples based on the indices
    sampled_data = df_train.iloc[selected_indices[::-1]]  # Reverse order for best demonstration last
    class_label = sampled_data[label_target] 
    if bool2YesNo:
        class_label = ["Yes" if i == 1 else "No" for i in class_label]
    return list(chain.from_iterable(zip(sampled_data[label_source], class_label))), selected_indices


def get_samples_top(a_emb, df_train, train_pool, num_examples, label_source='SMILES', label_target='description', bool2YesNo=False):
    selected_indices = torch.topk(F.cosine_similarity(a_emb, train_pool), num_examples).indices.tolist()
    sampled_data = df_train.iloc[selected_indices[::-1]]

    class_label = sampled_data[label_target] 
    if bool2YesNo:
        class_label = ["Yes" if i == 1 else "No" for i in class_label]
    return list(chain.from_iterable(zip(sampled_data[label_source], class_label)))

