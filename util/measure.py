import pandas as pd
import evaluate
from pathlib import Path


def get_filename(input_csv):
    filename = Path(input_csv).stem
    return filename

def get_filename_w_parents(input_csv):
    path = Path(input_csv)
    filename = path.stem
    dir = "/".join(path.parts[-3:-1])
    print(dir)
    return f"{dir}/{filename}"

def measure(output_file, time_prompt):
    df = pd.read_csv(output_file)
    res = df['pred'].tolist()
    references = df['ref'].tolist()
    bleu_metric = evaluate.load("bleu")

    bleu4 = bleu_metric.compute(predictions=res, references=references)
    bleu2 = bleu_metric.compute(predictions=res, references=references, max_order=2)

    rouge_metric = evaluate.load('rouge')
    rouge = rouge_metric.compute(predictions=res,
                      references=references)
    meteor_metric = evaluate.load('meteor')
    meteor = meteor_metric.compute(predictions=res, references=references)

    accuracy_metric = evaluate.load("exact_match")
    accuracy = accuracy_metric.compute(predictions=res, references=references)
    
    file_path = 'EXPERIMENTS.txt'
    print(f"{get_filename(output_file)}, {len(res)}, {accuracy['exact_match']}, {bleu2['bleu']}, {bleu4['bleu']}, {rouge['rouge1']}, {rouge['rouge2']}, {rouge['rougeL']}, {meteor['meteor']}, {time_prompt}" + '\n')
    with open(file_path, 'a') as file:
        file.write(f"{get_filename(output_file)}, {len(res)}, {accuracy['exact_match']}, {bleu2['bleu']}, {bleu4['bleu']}, {rouge['rouge1']}, {rouge['rouge2']}, {rouge['rougeL']}, {meteor['meteor']}, {time_prompt}" + '\n')
