import numpy as np
import pandas as pd
import torch
import time
from itertools import chain
from argparse import ArgumentParser
from datasets import load_dataset

#from util.chatgpt import run_chatgpt
from run_prompts import run_prompts
from util.balanced_kmeans import balanced_kmeans
from util.icl import ICL
from util.measure import measure
from util.prompt import create_cot_prompt, create_incontext_prompt2, create_justcode_prompt, create_zeroshot_prompt1, create_zeroshot_prompt2, get_answer

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

def run_transformer(args, df, time_prompt):
    if args.limit > 0:
        df = df[0:args.limit]

    prompts_all = df['prompt1'].to_numpy().flatten()
    references = df['ref'].tolist()

    output_file = f"./output/{args.dataset}-{args.model_name}-{args.method}-{args.num_examples}.csv"
    if (args.model_name == "openai/chatgptxxxx"):
        #run_chatgpt(args.langmodel_name_model, prompts_all, output_file)
        exit(-1)
    else:
        run_prompts(args.model_name, prompts_all, output_file)
        
    measure(args, output_file, time_prompt, references)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="liupf/ChEBI-20-MM") 
    parser.add_argument('--num_examples', type=int, default=2)
    parser.add_argument('--method', type=str, default="icl-new")
    parser.add_argument('--output_csv', type=str, default='/home/ali.lawati/gnn-incontext/cosql_processed2.csv.gptj.incontext')
    parser.add_argument('--model_name', type=str, default="gemma-7b") 
    parser.add_argument('--limit', type=int, default=100) 
    parser.add_argument('--epochs', type=int, default=1) 
    parser.add_argument('--input_csv', type=str, default=None)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--encoder_type', type=str, default='GraphAutoencoder')
    parser.add_argument('--gpus', type=int, default='3')
    
    args = parser.parse_args()
    print(f"Starting: method: {args.method}, limit: {args.limit}, epochs: {args.epochs}")
    if(args.input_csv):
        print("Manual Input CSV")
        df = pd.read_csv(args.input_csv)
        prompts = df['prompt1'].values
        ref = df['ref'].values
        run_transformer(args, df, 0)
    else:
        dataset_name = args.dataset
        if (args.dataset == "chebi"):
            dataset_name = 'liupf/ChEBI-20-MM'

        dataset = load_dataset(dataset_name)
        df_train = dataset['train'].to_pandas()
        df_valid = dataset['validation'].to_pandas()
        df_test = dataset['test'].to_pandas()

        time_gnn = 0

        predictions1, predictions2, predictions3, references = [], [], [], []
        prompts1, prompts2, prompts3, prompts4 = [], [], [], []

        batch_size = 16
        start_time = time.time()
        if(args.method[0:3] == "icl"):
            icl = ICL(df_train, df_valid, df_test, args)

        for i in range(0, len(df_test)):
            prog = df_test.iloc[i]['SMILES']
            
            if(args.method == "random"):
                p_args1 = get_samples(df_train, -1, args.num_examples) + [prog]
                prompt1 = create_incontext_prompt2(*p_args1)
            elif(args.method=="icl"):
                p_args1 = icl.get_samples(i, args.num_examples) + [prog]
                prompt1 = create_incontext_prompt2(*p_args1)
            elif(args.method=="icl-new"):
                p_args1 = icl.get_samples_new(i, args.num_examples) + [prog]
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

        data_dict = {
            'prompt1': np.squeeze(prompts1),
            'ref': references
        }
        end_time = time.time()
        time_prompt = end_time - start_time
        print("Prompt Selection time:", time_prompt)
        df = pd.DataFrame(data_dict)

        if (args.limit > 0):
            df = df[:args.limit]

        df.to_csv(f"./input/GCL-{args.dataset}-{args.method}-{args.num_examples}-{args.limit}-epochs{args.epochs}.csv", index=False)
        run_transformer(args, df, time_prompt)
