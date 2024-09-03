import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util
import evaluate
import torch.nn.functional as F
import pdb

def measure(args, output_file, time_prompt, references):
    st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    df = pd.read_csv(output_file)
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
        file.write(f"{args.limit}-{args.dataset}-{args.model_name}-{args.method}-{args.num_examples}-{args.epochs}.csv, {score1}, {score2}, {bleu1['bleu']}, {bleu2['bleu']}, {time_prompt}" + '\n')
