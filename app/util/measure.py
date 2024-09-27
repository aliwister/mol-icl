import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util
import evaluate
import torch.nn.functional as F
import pdb
from pathlib import Path


def create_scores_file(file, blue):
    np.save(f"/home/ali.lawati/mol-incontext/input/embed/{get_filename(file)}.scores.npy", blue)


def get_filename(input_csv):
    # Get the file name without directory and extension
    filename = Path(input_csv).stem
    return filename

def measure(output_file, time_prompt):
    st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    df = pd.read_csv(output_file)
    res = df['pred'].tolist()
    references = df['ref'].tolist()

    emb_res = st_model.encode(res, convert_to_tensor=True)
    emb_ref = st_model.encode(references, convert_to_tensor=True)
    score1 = F.cosine_similarity(emb_res, emb_ref, dim=1).mean().item()

    sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    emb_res = sbert_model.encode(res, convert_to_tensor=True)
    emb_ref = sbert_model.encode(references, convert_to_tensor=True)
    score2 = F.cosine_similarity(emb_res, emb_ref, dim=1).mean().item()

    bleu_metric = evaluate.load("bleu")


    # Initialize lists to store individual scores
    bleu4_scores = []
    bleu2_scores = []

    # Compute BLEU scores for each example
    for pred, ref in zip(res, references):
        # Each ref should be a list of reference texts, as BLEU expects multiple references for each prediction
        if not isinstance(ref, list):
            ref = [ref]
            
        # Compute BLEU-4 score
        bleu4 = bleu_metric.compute(predictions=[pred], references=[ref])
        bleu4_scores.append(bleu4['bleu'])
        
        # Compute BLEU-2 score (up to n-gram order 2)
        bleu2 = bleu_metric.compute(predictions=[pred], references=[ref], max_order=2)
        bleu2_scores.append(bleu2['bleu'])

    #lowest_values = sorted(enumerate(bleu4_scores), key=lambda x: x[1])[:5]
    #lowest_indexes, lowest_scores = zip(*lowest_values)
    #print(bleu4_scores)

    #lowest_values = sorted(enumerate(bleu2_scores), key=lambda x: x[1])[:5]
    #lowest_indexes, lowest_scores = zip(*lowest_values)
    #print(bleu2_scores)

    create_scores_file(output_file, np.array(bleu4_scores))
    bleu4 = bleu_metric.compute(predictions=res, references=references)
    bleu2 = bleu_metric.compute(predictions=res, references=references, max_order=2)

    rouge_metric = evaluate.load('rouge')
    rouge = rouge_metric.compute(predictions=res,
                      references=references)
    meteor_metric = evaluate.load('meteor')
    meteor = meteor_metric.compute(predictions=res, references=references)

    file_path = 'EXPERIMENTS.txt'
    print(f"{get_filename(output_file)}, {len(res)}, {bleu2['bleu']}, {bleu4['bleu']}, {rouge['rouge1']}, {rouge['rouge2']}, {rouge['rougeL']}, {meteor['meteor']}, {score1}, {score2}, {time_prompt}" + '\n')
    with open(file_path, 'a') as file:
        file.write(f"{get_filename(output_file)}, {len(res)}, {bleu2['bleu']}, {bleu4['bleu']}, {rouge['rouge1']}, {rouge['rouge2']}, {rouge['rougeL']}, {meteor['meteor']}, {score1}, {score2}, {time_prompt}" + '\n')
