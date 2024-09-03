from argparse import ArgumentParser
from contextlib import nullcontext
import datetime
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import pipeline
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

models = {
    "llama-3-8B": ("meta-llama/Meta-Llama-3-8B-Instruct", 8192),
    "zephy7B": ("HuggingFaceH4/zephyr-7b-beta", 32768),
    "gemma-7B": ("google/gemma-7b", 8192),
    "Qwen-2-7B": ("Qwen/Qwen2-7B-Instruct", 32768),
    #"mistral-7B": ("mistralai/Mistral-7B-Instruct-v0.3", 32768),
    "mistral-7B": ("mistralai/Mistral-7B-v0.1", 2048),
    # "PHI-medium-14B": ("microsoft/Phi-3-medium-128k-instruct", 131072),
    "openchat-8B": ("openchat/openchat-3.6-8b-20240522", 8192),
    "WizardLM-2-7B": ("lucyknada/microsoft_WizardLM-2-7B", 32768),
    "gpt-j-6b": ("EleutherAI/gpt-j-6b", 2048)
}

# Function to generate text
# Function to generate text
def generate_text(model, tokenizer, max_tokens, prompt, device):
        prompt_tokenized=tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
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
        generated_text = tokenizer.decode(outputs[input_length:], skip_special_tokens=True) 
        generated_text = generated_text.split('#')[0]

        return generated_text

def generate_text_pipeline(model, tokenizer, max_tokens, messages, device):
    # Setup the text generation pipeline with the specified model and tokenizer
    text_gen_pipeline = pipeline("text-generation", 
                                model=model, 
                                tokenizer=tokenizer,
                                torch_dtype=torch.bfloat16,
                                trust_remote_code=True
                                )
    input_prompt = messages
    # Measure time for text generation
    start_time = datetime.datetime.now()
    # Use autocast only if using CUDA
    #use_autocast = torch.cuda.is_available()
    #with torch.autocast() if use_autocast else nullcontext():
    try:
        generated_sequences = text_gen_pipeline(input_prompt,
                                                max_length=max_tokens,
                                                do_sample=True,
                                                top_k=5, 
                                                num_return_sequences=1,
                                                eos_token_id=tokenizer.eos_token_id,
                                                return_full_text=False,
                                                truncation=True,
                                                temperature=1, 
                                                top_p=1
                                                )
        generated_text = generated_sequences[0]['generated_text']
        # Print the structure of the generated sequences for debugging
        # print(f"Generated Sequences: {generated_text}")
    except KeyError as e:
        print(f"KeyError: {e}")
        generated_text = "Error in generation: key not found"
    except Exception as e:
        print(f"Unexpected error: {e}")
        generated_text = "Error in generation: unexpected error"

    end_time = datetime.datetime.now()
    time_taken = (end_time - start_time).total_seconds()

    # Return the generated text and time taken
    return {
        "generated_text": generated_text,
        "time_taken": time_taken
    }

def load_model(model_name, model_config, device):
    model = AutoModelForCausalLM.from_pretrained(model_config[0], device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_config[0])
    tokenizer.pad_token = tokenizer.eos_token

    #model.to(device)
    model.half()
    return model, tokenizer, model_config[1]

def run_prompts(model_name, prompts_all, output_csv): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    accelerator = Accelerator()
    print(accelerator.device)
    # 10*10 Prompts. Source: https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books
    # load a base model and tokenizer
    """    if (lang_model == "EleutherAI/gpt-j-6b"):
        
            model_path = lang_model
            tokenizer = AutoTokenizer.from_pretrained(model_path)   
            tokenizer.pad_token = tokenizer.eos_token
            model = GPTJForCausalLM.from_pretrained(
                model_path,    
                #device_map={"": accelerator.process_index},
                torch_dtype=torch.bfloat16,
            ).to(device)
            
        elif(lang_model == "meta-llama/CodeLlama-7b-hf"):
            tokenizer = CodeLlamaTokenizer.from_pretrained(lang_model)
            model = LlamaForCausalLM.from_pretrained(lang_model).to(device)
        elif(lang_model == "mistralai/Mistral-7B-v0.1"):
            model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto").to(device)
            
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(f"Unsupported language model: {lang_model}")
    """

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start=time.time()

    # divide the prompt list onto the available GPUs 
    model, tokenizer, max_tokens = load_model(model_name, models[model_name], device)
    model = accelerator.prepare(model)
    with accelerator.split_between_processes(prompts_all) as prompts:
        # store output of generations in dict
        results=dict(outputs=[], num_tokens=0)
        
        #model = model.to(device)
        # have each GPU do inference, prompt by prompt
        for prompt in tqdm(prompts, desc="Processing SQL prompts"):
            
            
            result = generate_text(model, tokenizer, max_tokens, prompt, device)
            #generated_text = generated_text.split('#')[0]

            results["outputs"].append(result)
            results["num_tokens"] += len(result)

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
