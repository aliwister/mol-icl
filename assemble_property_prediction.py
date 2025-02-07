import pandas as pd
import torch
from argparse import ArgumentParser
from util.util import create_incontext_prompt2, get_random_samples, get_samples_new, get_samples_top, get_scaffold_samples, generate_scaffolds, create_input_file, create_incontext_prompt_binary
from model.mmcl import MMCL
from model.gae import GAE
import torch_geometric.utils.smiles as smiles
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from rdkit import Chem

TASK = "property-prediction"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def smiles2graph(smiles_str):
    data = smiles.from_smiles(smiles_str)
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()
    return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)

def assemble(method, dataset, checkpoint, random_state = 42, llambda=0.3):
    if (method == "mmcl-nomorgan"):
        checkpoint = "./checkpoints/mmcl-morgan-exp1-301-chebi-scibert-768-morgan=False.pt"

    dataset = dataset
    test_size=0.2
    input_label="Molecule"
    if(dataset == "BBBP"):
        dataset_file = "./data/property_prediction/bbbp.csv"
        label_source = 'smiles' 
        label_target = 'BBBP'
        output_label="BBBP"    
        description="is able to penetrate the blood-brain barrier"
    elif(dataset == "BACE"):
        dataset_file = "./data/property_prediction/bace.csv"
        label_source = 'smiles' 
        label_target = 'Class'        
        output_label = 'Class'    
        description = 'is a beta-secretase 1 inhibitor'           
    elif(dataset == "clintox"):
        dataset_file = "./data/property_prediction/clintox.csv"
        label_source = 'smiles' 
        label_target = 'CT_TOX'    
        output_label="CT_TOX"    
        description="has Cellular Toxicity"  
    elif(dataset == "HIV"):
        dataset_file = "./data/property_prediction/hiv.csv"
        label_source = 'smiles' 
        label_target = 'HIV_active'
        output_label = 'HIV Active'         
        description = 'is classified activity against the HIV'   
    elif(dataset == "tox21"):
        dataset_file = "./data/property_prediction/tox21.csv"
        label_source = 'smiles' 
        label_target = 'NR-ER'     
        output_label = 'NR-ER'
        description = "is a Nuclear Receptor Estrogen Receptor"     

    df = pd.read_csv(dataset_file) #, sep='\t')
    df_filtered = df[df[label_source].apply(lambda x: Chem.MolFromSmiles(x) is not None)]

    df_train, df_test = train_test_split(df_filtered, test_size=0.1, random_state=random_state)
    df_train, df_val = train_test_split(df_train, test_size=0.1111, random_state=random_state) 

    test_graphs = [smiles2graph(smiles) for smiles in df_test[label_source]]
    train_graphs = [smiles2graph(smiles) for smiles in df_train[label_source]]
    
    method = method
    import pdb
    if(method == "scaffold"):
        sc_train_pool = generate_scaffolds(df_train, label_source)
        sc_test_pool = generate_scaffolds(df_test, label_source)
    elif(method != "random" and method != "zero"):
        if method == "mmcl":
            model = MMCL(9, 128, 768, 9)
        else:
            model = GAE(9, 32, 16, 9)

        model.load_state_dict(torch.load(checkpoint, map_location=torch.device(device)))
        model.to(device)

        model.eval()
        with torch.no_grad():
            train_batch = Batch.from_data_list(train_graphs).to(device)
            test_batch  = Batch.from_data_list(test_graphs).to(device)
            _, _, train_pool = model(train_batch.x, train_batch.edge_index, train_batch.batch, train_batch.edge_attr)
            _, _, test_pool  = model(test_batch.x, test_batch.edge_index, test_batch.batch, test_batch.edge_attr)
            
    LEN = 8
    df = df_test
    refs = []
    prompts = [[] for i in range(0,LEN)]
    indices = [[] for i in range(0,LEN)]

    for i in range(0, len(df)):
        prog = df.iloc[i][label_source]
        # Create Input files with different demonstration sizes
        if (method == "zero"):
            prompts[0].append(create_incontext_prompt_binary(prog, input_label=input_label, output_label=output_label, description=description))
        else:
            for n in range(0, LEN):
                if method.startswith("random"):
                    samples = get_random_samples(df_train, n+1, label_source, label_target, True)
                elif(method == "scaffold"):
                    samples = get_scaffold_samples(sc_test_pool[i], df_train, sc_train_pool, n+1, label_source, label_target, True)               
                elif(method == "mmcl-top"):
                    samples = get_samples_top(test_pool[i], df_train, train_pool, n+1, label_source, label_target, True)
                else: 
                    samples, selected_indices = get_samples_new(test_pool[i], df_train, train_pool, n+1, label_source, label_target, True)
                    indices[n].append(selected_indices)
                prompts[n].append(create_incontext_prompt2(*samples + [prog], input_label=input_label, output_label=output_label))


        refs.append("Yes" if df.iloc[i][label_target] == 1 else "No")
    if (method == "zero"):
        create_input_file(prompts[0], refs, 0, f"{method}-{dataset}-{random_state}", TASK)
    else:
        [create_input_file(prompts[i], refs, i+1, f"{method}-{dataset}-GAE-{random_state}", TASK) for i in [0,1,2,3]]
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="tox21")
    parser.add_argument('--num_examples', type=int, default=2)
    parser.add_argument('--method', type=str, default="scaffold")
    parser.add_argument('--gpus', type=int, default='1')
    parser.add_argument('--model_checkpoint', type=str, default='./checkpoints/mmcl-morgan-exp1-301-chebi-scibert-768.pt')
    parser.add_argument('--create_embeds', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default='16')
    parser.add_argument('--lamda', type=int, default='.3')
    
    args = parser.parse_args()
    checkpoint = args.model_checkpoint
    random_states = [3, 42, 53]
    for rs in random_states:
        assemble(args.method, args.dataset, checkpoint, rs, args.lamda)