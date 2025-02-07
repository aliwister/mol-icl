from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from statistics import mean
import torch, time, json
from tqdm import tqdm
import pandas as pd
from util.measure import measure, get_filename_w_parents
from util.measure_property import measure as measure_prop
import os

import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "llama-3-8B": ("meta-llama/Meta-Llama-3-8B-Instruct", 8192),
    "zephy-7B": ("HuggingFaceH4/zephyr-7b-beta", 32768, 'auto'),
    "gemma-7B": ("google/gemma-7b", 8192, 'auto'),
    "Qwen-2-7B": ("Qwen/Qwen2.5-7B", 32768, 'auto'),
    "mistral-7B": ("mistralai/Mistral-7B-v0.3", 2048, 'auto'),
    "openchat-8B": ("openchat/openchat-3.6-8b-20240522", 8192, 'auto'),
    "WizardLM-2-7B": ("lucyknada/microsoft_WizardLM-2-7B", 32768, 'auto'),
    "gpt-j-6b": ("EleutherAI/gpt-j-6b", 2048),
    "mol-t5-l": ('laituan245/molt5-large-smiles2caption', 512, 't5', False),
}


def create_folder(path):
    os.makedirs(path, exist_ok=True)

def write_pretty_json(file_path, data):
    import json
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)

def prepare_prompts(prompts, tokenizer, batch_size=4):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok

BATCH_SIZE = 32
def main(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    accelerator = Accelerator()

    df = pd.read_csv(args.input_csv)
    if  args.limit > 0:
        df = df[0:args.limit]
    prompts_all = df['prompt'].tolist()
    references = df['ref'].tolist()

    # load a base model and tokenizer
    model_path=models[args.model_name][0]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
        cache_dir=f"../llm/{args.model_name}"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path) #, cache_dir=f"../data/{model_name}")   
    tokenizer.pad_token = tokenizer.eos_token

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    
    start=time.time()
    if accelerator.is_main_process:
        pbar=tqdm(total=len(prompts_all))    
    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        print (len(prompts))
        results=dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=BATCH_SIZE)
        for prompts_tokenized in prompt_batches:

            outputs_tokenized=model.generate(
                **prompts_tokenized, 
                temperature=1,       # Control the randomness of the output
                top_p=1,             # Use nucleus sampling
                top_k=50,              # Use top-k sampling
                num_return_sequences=1,
                max_new_tokens=350,
                pad_token_id=tokenizer.eos_token_id)
            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

            # count and decode gen. tokens 
            num_tokens=sum([ len(t) for t in outputs_tokenized ])
            #pdb.set_trace()
            outputs=tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
            processed_outputs = [output.split('###')[0].replace('"', '').strip() for output in outputs]
            #print(processed_outputs)
            # store in results{} to be gathered by accelerate
            results["outputs"].extend(processed_outputs)
            results["num_tokens"] += num_tokens
            time.sleep(0.1)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                pbar.update( accelerator.num_processes * BATCH_SIZE )

        results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

    results_gathered=gather_object(results)

    if accelerator.is_main_process:
        #print(len(results_gathered))
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered ])
        #results_all = [item.split("#")[0] for r in results_gathered for item in r["outputs"]]
        results_all = [item for r in results_gathered for item in r["outputs"]]

        print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
        #pdb.set_trace()
    
        output_csv = f"{args.output_dir}/{get_filename_w_parents(args.input_csv)}.{args.model_name}.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df = pd.DataFrame({
            'pred': results_all,
            'ref': references
            })
        df.to_csv(output_csv, index=False)
        print(f"{output_csv}, tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")
        file_path = 'EXPERIMENTS_SUMMARY.txt'
        with open(file_path, 'a') as file:
            file.write(f"{output_csv}, tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}\n")
            if (args.output_int):
                measure_prop(output_csv, timediff)
            else:
                measure(output_csv, timediff)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--model_name', type=str, default="mol-t5-l") 
    parser.add_argument('--output_csv', type=str, default=None) 
    parser.add_argument('--output_int', type=bool, default=True) 
    parser.add_argument('--input_csv', type=str, default='./input/molecule-design/mmcl-4-molecule-design.csv') #, required=True)
    parser.add_argument('--limit', type=int, default=100) 
    args = parser.parse_args() 
    
    if (args.output_csv):
        with open(args.output_csv, 'a') as file:
            if (args.output_int):
                measure_prop(args.output_csv, 0)
            else:
                measure(args.output_csv, 0)
    else:
        main(args)