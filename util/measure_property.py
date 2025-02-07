import pandas as pd

import evaluate
from pathlib import Path

def get_filename(input_csv):
    filename = Path(input_csv).stem
    return filename

def measure(output_file, time_prompt):
    df = pd.read_csv(output_file)
    res = df['pred'].tolist()
    references = df['ref'].tolist()

    res = [1 if str(val).strip().lower() == 'yes' else 0 for val in res]
    references = [1 if str(val).strip().lower() == 'yes' else 0 for val in references]

    accuracy_metric = evaluate.load("accuracy")
    accuracy = accuracy_metric.compute(predictions=res, references=references)

    f1_metric = evaluate.load("f1")
    f1 = f1_metric.compute(predictions=res, references=references)
    
    file_path = 'EXPERIMENTS_PROP.txt'
    print(f"{get_filename(output_file)}, {len(res)}, {accuracy['accuracy']}, {f1['f1']}, {time_prompt}" + '\n')
    with open(file_path, 'a') as file:
        file.write(f"{get_filename(output_file)}, {len(res)}, {accuracy['accuracy']}, {f1['f1']}" + '\n')
