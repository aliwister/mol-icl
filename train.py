import numpy as np
import torch
import torch.nn as nn
from itertools import chain
from argparse import ArgumentParser
import pandas as pd
from torch_geometric.loader import DataLoader
from datasets import load_dataset

from math import ceil
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

import evaluate
from run_prompts import run_prompts

from util.graph import mol_to_graph, smiles2graph
from util.model import GraphAutoencoder, extract_latent_representations, train_autoencoder
from util.prompt import create_cot_prompt, create_incontext_prompt2, create_justcode_prompt, create_zeroshot_prompt1, create_zeroshot_prompt2, get_answer
import pdb

from torch_geometric.data import Data, Batch
from util.sql_tree import parse_query
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string
import time

# Import the balanced contrastive similarity components
from balanced_contrastive_similarity import balanced_contrastive_similarity, graph_based_clustering, GraphEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_string_to_array(string):
    stripped_string = string.strip('[]')
    array = np.array([float(num) for num in stripped_string.split()])
    return array

def encode(self, batch, lang_model):
    x, edge_index, attention_mask = batch.x, batch.edge_index, batch.attention_mask
    inputs = lang_model(x, output_hidden_states=True, attention_mask=attention_mask)
    hidden_states = inputs.hidden_states
    last_hidden_state = hidden_states[-1]
    denom = torch.sum(attention_mask, -1, keepdim=True)
    feat = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / denom
    feat = feat.to(torch.float32)
    return feat

def get_samples(df, cluster, num):
    fdf = df
    if cluster > -1:
        fdf = df[df['label'] == cluster]
    sampled_data = fdf.sample(n=num)
    return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description'])))

def get_samples_top(df, num, model, train_graphs, test_graph, cluster_assignments):
    model.eval()
    with torch.no_grad():
        train_batch = Batch.from_data_list(train_graphs)
        test_batch = Batch.from_data_list([test_graph])
        
        train_embeddings = model(train_batch)
        test_embedding = model(test_batch)

        # Get the cluster of the test graph
        test_cluster = graph_based_clustering([test_graph], len(set(cluster_assignments)))[0]

        # Filter train_embeddings to only include graphs from the same cluster
        cluster_mask = torch.tensor(cluster_assignments) == test_cluster
        cluster_embeddings = train_embeddings[cluster_mask]

        # For the first example (B), use the original method
        similarities = F.cosine_similarity(test_embedding, train_embeddings)
        first_index = torch.argmax(similarities)

        # For the second example, use GraphContrastiveSimilarity within the cluster
        cluster_similarities = F.cosine_similarity(test_embedding, cluster_embeddings)
        second_index = torch.argmax(cluster_similarities)

        # Convert the second index back to the original index in the full dataset
        full_indices = torch.where(cluster_mask)[0]
        second_index = full_indices[second_index]

    # Ensure we don't select the same example twice
    if first_index == second_index:
        second_index = torch.argsort(similarities, descending=True)[1]

    selected_indices = [first_index.item(), second_index.item()]
    sampled_data = df.iloc[selected_indices]
    return list(chain.from_iterable(zip(sampled_data['SMILES'], sampled_data['description'])))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    return tokens

def get_samples_bm25(df, cluster, num, bm25, test):
    tokenized_query = preprocess(test)

    doc_scores = bm25.get_scores(tokenized_query)
    x = np.argpartition(doc_scores, -num)[-num:]
    sampled_data = df.iloc[x]
    return list(chain.from_iterable(zip(sampled_data['query'], sampled_data['utterance'])))

def run_transformer(args, prompts1, references, start_time):
    data_dict = {
        'prompt1': np.squeeze(prompts1),
        'ref': references
    }
    end_time = time.time()
    time_prompt = end_time - start_time

    df = pd.DataFrame(data_dict)

    if args.limit > 0:
        df = df[0:args.limit]
        references = references[0:args.limit]

    prompts_all = df['prompt1'].to_numpy().flatten()
    if (args.lang_model == "openai/chatgptxxxx"):
        client = OpenAI(
            api_key="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",
        )   
        def get_response_zero(prompt):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an intelligent SQL Code assistant who effectively translates the intent and logic of the SQL queries into natural language that is easy to understand."},
                    {"role": "user", "content": """Convert the given SQL query into a clear and concise natural language query limited to 1 sentence.  Ensure that the request accurately represents the actions specified in the SQL query and is easy to understand for someone without technical knowledge of SQL.
                    Input: """ + prompt},
                ],
                temperature=1,
                max_tokens=150,
                top_p=1
            )
            return response.choices[0].message.content

        def get_response(prompt):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
                max_tokens=150,
                top_p=1
            )
            return response.choices[0].message.content

        responses = []
        for prompt in prompts_all:
            if (args.method == "zero-gpt"):
                response = get_response_zero(prompt)
            else:
                response = get_response(prompt)
            print(f"Prompt: {prompt}\nResponse: {response}\n")
            responses.append(response)
        df = pd.DataFrame(responses)
        df.to_csv(f"./output/{args.dataset}-{args.lang_model.split('/')[1]}-{args.method}-{args.num_examples}.csv", index=False)
    else:
        run_prompts(args.lang_model, prompts_all, f"./output/{args.dataset}-{args.lang_model.split('/')[1]}-{args.method}-{args.num_examples}.csv")
        df.to_csv(f"./input/{args.dataset}-{args.method}-{args.num_examples}-{args.limit}.csv", index=False)
    print(f"./input/{args.dataset}-{args.method}-{args.num_examples}-{args.limit}.csv")
    measure(args, time_prompt, references)

def measure(args, time_prompt, references):
    st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    df = pd.read_csv(f"./output/{args.dataset}-{args.lang_model.split('/')[1]}-{args.method}-{args.num_examples}.csv")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    res = np.squeeze(df.values)
    emb_res = st_model.encode(res, convert_to_tensor=True)
    emb_ref = st_model.encode(references, convert_to_tensor=True)
    score1 = F.cosine_similarity(emb_res, emb_ref, dim=1).mean().item()

    sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    emb_res = sbert_model.encode(res, convert_to_tensor=True)
    emb_ref = sbert_model.encode(references, convert_to_tensor=True)
    score2 = F.cosine_similarity(emb_res, emb_ref, dim=1).mean().item()

    bleu_metric = evaluate.load("bleu")
    bleu1 = bleu_metric.compute(predictions=res, references=references)
    bleu2 = bleu_metric.compute(predictions=res, references=references, max_order=2)

    file_path = 'EXPERIMENTS.txt'
    with open(file_path, 'a') as file:
        file.write(f"{args.limit}-{args.dataset}-{args.lang_model.split('/')[1]}-{args.method}-{args.num_examples}-{args.epochs}.csv, {score1}, {score2}, {bleu1['bleu']}, {bleu2['bleu']}, {time_prompt}" + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="liupf/ChEBI-20-MM") 
    parser.add_argument('--num_examples', type=int, default=2)
    parser.add_argument('--method', type=str, default="icl")
    parser.add_argument('--output_csv', type=str, default='/home/ali.lawati/gnn-incontext/cosql_processed2.csv.gptj.incontext')
    parser.add_argument('--lang_model', type=str, default="EleutherAI/gpt-j-6b") 
    parser.add_argument('--limit', type=int, default=-1) 
    parser.add_argument('--epochs', type=int, default=1) 
    parser.add_argument('--input_csv', type=int, default=0)
    parser.add_argument('--n_clusters', type=int, default=5)
    
    args = parser.parse_args()

    if(args.input_csv):
        print("Manual Input CSV")
        df = pd.read_csv('/home/ali.lawati/mol-incontext/input/debug.csv')
        prompts = df['prompt1'].values
        ref = df['ref'].values
        run_transformer(args, prompts, ref, 0)
    else:
        dataset_name = args.dataset
        if (args.dataset == "chebi"):
            dataset_name = 'liupf/ChEBI-20-MM'

        dataset = load_dataset(dataset_name)
        df_train = dataset['train'].to_pandas()
        df_valid = dataset['validation'].to_pandas()
        df_test = dataset['test'].to_pandas()

        time_gnn = 0
        if (args.method == "icl"):
            train_graphs = [smiles2graph(smiles) for smiles in df_train['SMILES']]
            train_loader = DataLoader(train_graphs, batch_size=16, shuffle=False)

            val_graphs = [smiles2graph(smiles) for smiles in df_valid['SMILES']]
            val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)

            test_graphs = [smiles2graph(smiles) for smiles in df_test['SMILES']]
            test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

            # Initialize and train the GraphAutoencoder model
            input_dim = 9  # From the smiles2graph function output
            hidden_dim = 32
            embedding_dim = 16
            autoencoder = GraphAutoencoder(input_dim, hidden_dim, embedding_dim).to(device)

            start_time = time.time()
            train_autoencoder(autoencoder, train_loader, val_loader, epochs=args.epochs)
            
            # Extract latent representations
            train_embeddings = extract_latent_representations(autoencoder, train_loader)
            
            # Perform balanced contrastive clustering
            cluster_assignments = graph_based_clustering(train_graphs, args.n_clusters)
            
            # Initialize the GraphEncoder for contrastive similarity
            graph_encoder = GraphEncoder(input_dim, hidden_dim, embedding_dim).to(device)
            
            # Train the GraphEncoder using balanced contrastive similarity
            optimizer = torch.optim.Adam(graph_encoder.parameters(), lr=0.001)
            for epoch in range(args.epochs):
                graph_encoder.train()
                total_loss = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    embeddings = graph_encoder(batch)
                    loss = balanced_contrastive_similarity(embeddings, [g.to(device) for g in train_graphs], n_clusters=args.n_clusters)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}")

            end_time = time.time()
            time_gnn = end_time - start_time

        predictions1, predictions2, predictions3, references = [], [], [], []
        prompts1, prompts2, prompts3, prompts4 = [], [], [], []

        batch_size = 16
        start_time = time.time()

        for i in range(0, len(df_test)):
            prog = df_test.iloc[i]['SMILES']
            
            if(args.method == "random"):
                p_args1 = get_samples(df_train, -1, args.num_examples) + [prog]
                prompt1 = create_incontext_prompt2(*p_args1)
            elif(args.method=="icl"):
                p_args1 = get_samples_top(df_train, args.num_examples, graph_encoder, train_graphs, test_graphs[i], cluster_assignments) + [prog]
                prompt1 = create_incontext_prompt2(*p_args1)
            elif(args.method=="zero1"):
                p_args2 = (prog,)
                prompt1 = create_zeroshot_prompt1(*p_args2)
            elif(args.method=="zero2"):
                p_args2 = (prog,)
                prompt1 = create_zeroshot_prompt2(*p_args2)
            elif(args.method=="cot"):
                p_args2 = (prog,)
                prompt1 = create_cot_prompt(*p_args2)
            elif(args.method=="zero-gpt"):
                prompt1 = prog

            prompts1.append(prompt1)
            references.append(df_test.iloc[i]['description'])
        run_transformer(args, prompts1, references, start_time)
