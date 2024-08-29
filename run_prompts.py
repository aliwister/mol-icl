from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from transformers import RobertaTokenizer, T5ForConditionalGeneration

from transformers import LlamaForCausalLM, CodeLlamaTokenizer
from transformers import GPTJForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from statistics import mean
import torch, time, json
import pandas as pd
import pdb
from tqdm import tqdm


def run_prompts(lang_model, prompts_all, output_csv): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    accelerator = Accelerator()
    print(accelerator.device)
    # 10*10 Prompts. Source: https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books
    # load a base model and tokenizer
    if (lang_model == "EleutherAI/gpt-j-6b"):

        model_path = lang_model
        model = GPTJForCausalLM.from_pretrained(
            model_path,    
            #device_map={"": accelerator.process_index},
            torch_dtype=torch.bfloat16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)   
    elif(lang_model == "meta-llama/CodeLlama-7b-hf"):
        tokenizer = CodeLlamaTokenizer.from_pretrained(lang_model)
        model = LlamaForCausalLM.from_pretrained(lang_model).to(device)
    elif(lang_model == "mistralai/Mistral-7B-v0.1"):
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    else:
        raise ValueError(f"Unsupported language model: {lang_model}")


    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start=time.time()

    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        # store output of generations in dict
        results=dict(outputs=[], num_tokens=0)

        # have each GPU do inference, prompt by prompt
        for prompt in tqdm(prompts, desc="Processing SQL prompts"):
            
            prompt_tokenized=tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2040).to(device)
            outputs = model.generate(
                **prompt_tokenized, 
                #max_length=100,         # Limit the length of the output
                temperature=1,       # Control the randomness of the output
                top_p=1,             # Use nucleus sampling
                top_k=50,              # Use top-k sampling
                num_return_sequences=1, 
                max_new_tokens=100
                )[0]

            input_length = len(prompt_tokenized['input_ids'][0])

            # remove prompt from output 
            #output_tokenized=output_tokenized[len(prompt_tokenized["input_ids"][0]):]

            # store outputs and number of tokens in result{}
            # pdb.set_trace()
            generated_text = tokenizer.decode(outputs[input_length:], skip_special_tokens=True) 
            generated_text = generated_text.split('#')[0]

            results["outputs"].append(generated_text)
            results["num_tokens"] += len(generated_text)

        results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

    # collect results from all the GPUs
    results_gathered=gather_object(results)


    if accelerator.is_main_process:
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered ])
        #pdb.set_trace()
        #print(results_gathered)
        #flat_results = [item for sublist in results_gathered[0].values for item in sublist]
        df = pd.DataFrame(results_gathered[0]['outputs'])
        df.to_csv(output_csv, index=False)
        print(f"{output_csv}, tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")
        file_path = 'EXPERIMENTS_SUMMARY.txt'
        with open(file_path, 'a') as file:
            file.write(f"{output_csv}, tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}\n")

        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prompt_csv', type=str, default='/home/ali.lawati/gnn-incontext/spider.csv')
    parser.add_argument('--output_csv', type=str, default='/home/ali.lawati/gnn-incontext/cosql_processed2.csv.gptj.incontext')
    #parser.add_argument('--lang_model', type=str, default="Salesforce/codet5-large") 
    parser.add_argument('--lang_model', type=str, default="EleutherAI/gpt-j-6B") 
    #parser.add_argument('--prompt_num',  type=str, default="prompt1")
    args = parser.parse_args()
    print(args.lang_model)

    df=pd.read_csv(args.prompt_csv)

    prompts_all = df['prompt1'].to_numpy().flatten()
    prompts_all = prompts_all #[0:10]

    #run(args.lang_model, prompts_all, args.output_csv)
