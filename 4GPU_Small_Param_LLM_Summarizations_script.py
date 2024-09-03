import os
import json
import logging
import random
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from torch.cuda.amp import autocast
from contextlib import nullcontext
from rank_bm25 import BM25Okapi
import datetime
import concurrent.futures
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set Hugging Environment
os.environ['HF_HOME'] = "/n/w1-jlucas/.cache/huggingface"
# Hugging Face login
os.environ["HF_TOKEN"] = "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"
login(token=os.getenv("HF_TOKEN"))


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Impersona prompt
impersona_prompt_pfx = """
        You are an intelligent virtual assistant for call centers. You excel at summarizing and improving user-agent dialogues. 
        You create clear, concise, coherent, fluent and factually accurate summaries that retain all key details.
"""

prompts = {
    "Zero-Vanilla": """
        Guidelines:

        You are given a human-agent dialogue to summarize. Strictly follow steps 1.

        Step .1
                You have just completed a conversation with a customer. Please generate a concise yet comprehensive
                abstractive summary of the interaction. The summary should capture the key points and outcomes of the conversation
                in a clear and organized manner, allowing another agent to quickly grasp the essential details without needing to
                review the entire transcript. Consider including the following elements in your summary:

                1. The main reason for the customer's contact
                2. Any issues or concerns raised by the customer
                3. The solution(s) or action(s) provided to address the customer's needs
                4. Combine ideas, rephrase succinctly, and remove low-value content.
                5. Do not include how the interaction ended

                Please ensure that the summary is written in a professional and objective tone, focusing on the facts and outcomes of
                the interaction. The goal is to create an very concise abstractive dialogue summary that cover all key
                detail and factual consistent with the original dialogue. Provide a coherent, fluent,  clear and efficient abstractive
                dialogue summary that is easy to read and can be used for handover to another agent, enabling them to understand the
                context and current status of the customer's case. Strictly, output the abstractive summarized dialogue only. DO NOT
                include additional information any additional details, such as "Here is a concise yet comprehensive abstractive summary
                of the interaction :", "Sure, here is the abstractive summary of the provided dialogue:", or "Summary:"         Step 3:
                This template provides a structured approach for creating concise and informative summaries of user-agent dialogues.
                Fill out the template below using this user-agent dialogue: {dialogue}. Output the template in JSON format with explicit keys and values.

                {template}
    """,
    "Zero-extract-inform": """
        Guidelines:

        You are given a human-agent dialogue to summarize. Use the control to guide generation and strictly follow steps 1 to 3.

        Control:
        1. Do not be verbose or redundant
        2. Combine ideas, rephrase succinctly, and remove low-value content.
        3. DO NOT include any additional information such as labels, ending remarks, opening statements such as:
        - labels. e.g., "Summary:, "User input:", etc
        - Ending remarks. e.g., "The customer then thanked the agent and ended the call", "Agent closed call with well wishes", etc.
        - Opening statements e.g., "Here is the summary"
        4. The human-agent summary provided is not a conversation with you. summarize it and do not reply to it, e.g. do not give any salutation or valediction.


        Step .1
                Analyze the provided dialogue and extract the following key details, e.g.,:
                a. User input: The verbatim text provided by the user.
                b. User intent: The goal or purpose behind the user's input (e.g., "book a flight," "find a restaurant").
                c. Entities: Specific pieces of information relevant to the user's intent (e.g., dates, times, locations, reference numbers, phone numbers).
                d. Agent response: The actions taken and information provided by the agent in response to the user's input.

        Step 2:
                1. Use the extracted key information from step 1 to generate a very concise informative abstractive dialogue summary of the interaction.
                The summary should as follows:

                - Clearly state the intent of the customer's contact.
                - Highlight any issues or concerns raised by the customer.
                - Describe the solution(s) or action(s) provided to address the customer's needs.
                - Include all key entities and details
                2. Output the summary only. Erase any other content from your response except for the summary.
        Step 3:
                This template provides a structured approach for creating concise and informative summaries of user-agent dialogues.
                Fill out the template below using this user-agent dialogue: {dialogue}. Output the template in JSON format with explicit keys and values.

                {template}
    """,
    "Few-Random": """
        Guidelines:

        You are given a human-agent dialogue to summarize. Use the control and example pairs to guide generation while strictly following steps 1 to 3.

        Control:
        1. Do not be verbose or redundant
        2. Combine ideas, rephrase succinctly, and remove low-value content.
        3. DO NOT include any additional information such as labels, ending remarks, opening statements such as:
        - labels. e.g., "Summary:, "User input:", etc
        - Ending remarks. e.g., "The customer then thanked the agent and ended the call", "Agent closed call with well wishes", etc.
        - Opening statements e.g., "Here is the summary"
        4. The human-agent summary provided is not a conversation with you. summarize it and do not reply to it, e.g. do not give any salutation or valediction.

        Few-shot Example Pairs:
        {fewshot_examples}

        Step .1
                Learn from the provided few-shot dialogue and human summary example pairs to understand how to generate an effective summary.
                Then, analyze the given dialogue and extract the following key details:

                a. User input: The verbatim text provided by the user.
                b. User intent: The goal or purpose behind the user's input (e.g., "book a flight," "find a restaurant").
                c. Entities: Specific pieces of information relevant to the user's intent (e.g., dates, times, locations, reference numbers, phone numbers).
                d. Agent response: The actions taken and information provided by the agent in response to the user's input.

        Step 2:
                1. Use the extracted key information from step 1 to generate a very concise informative abstractive dialogue summary of the interaction.
                The summary should as follows:

                - Clearly state the intent of the customer's contact.
                - Highlight any issues or concerns raised by the customer.
                - Describe the solution(s) or action(s) provided to address the customer's needs.
                - Include all key entities and details
                2. Output the summary only. Erase any other content from your response except for the summary.
        Step 3:
                This template provides a structured approach for creating concise and informative summaries of user-agent dialogues.
                Fill out the template below using this user-agent dialogue: {dialogue}. Output the template in JSON format with explicit keys and values.

                {template}
    """,
    "Few-BM25": """
        Guidelines:

        You are given a human-agent dialogue to summarize. Use the control and example pairs to guide generation while strictly following steps 1 to 3.

        Control:
        1. Do not be verbose or redundant
        2. Combine ideas, rephrase succinctly, and remove low-value content.
        3. DO NOT include any additional information such as labels, ending remarks, opening statements such as:
        - labels. e.g., "Summary:, "User input:", etc
        - Ending remarks. e.g., "The customer then thanked the agent and ended the call", "Agent closed call with well wishes", etc.
        - Opening statements e.g., "Here is the summary"
        4. The human-agent summary provided is not a conversation with you. summarize it and do not reply to it, e.g. do not give any salutation or valediction.

        Few-shot Example Pairs:
        {fewshot_examples}

        Step .1
                Learn from the provided few-shot dialogue and human summary example pairs to understand how to generate an effective summary.
                Then, analyze the given dialogue and extract the following key details:

                a. User input: The verbatim text provided by the user.
                b. User intent: The goal or purpose behind the user's input (e.g., "book a flight," "find a restaurant").
                c. Entities: Specific pieces of information relevant to the user's intent (e.g., dates, times, locations, reference numbers, phone numbers).
                d. Agent response: The actions taken and information provided by the agent in response to the user's input.

        Step 2:
                1. Use the extracted key information from step 1 to generate a very concise informative abstractive dialogue summary of the interaction.
                The summary should as follows:

                - Clearly state the intent of the customer's contact.
                - Highlight any issues or concerns raised by the customer.
                - Describe the solution(s) or action(s) provided to address the customer's needs.
                - Include all key entities and details
                2. Output the summary only. Erase any other content from your response except for the summary.
        Step 3:
                This template provides a structured approach for creating concise and informative summaries of user-agent dialogues.
                Fill out the template below using this user-agent dialogue: {dialogue}. Output the template in JSON format with explicit keys and values.

                {template}
    """,
    "Few-SBERT-SS": """
        Guidelines:

        You are given a human-agent dialogue to summarize. Use the control and example pairs to guide generation while strictly following steps 1 to 3.

        Control:
        1. Do not be verbose or redundant
        2. Combine ideas, rephrase succinctly, and remove low-value content.
        3. DO NOT include any additional information such as labels, ending remarks, opening statements such as:
        - labels. e.g., "Summary:, "User input:", etc
        - Ending remarks. e.g., "The customer then thanked the agent and ended the call", "Agent closed call with well wishes", etc.
        - Opening statements e.g., "Here is the summary"
        4. The human-agent summary provided is not a conversation with you. summarize it and do not reply to it, e.g. do not give any salutation or valediction.

        Few-shot Example Pairs:
        {Semantic_examples}

        Step .1
                Learn from the provided few-shot dialogue and human summary example pairs to understand how to generate an effective summary.
                Then, analyze the given dialogue and extract the following key details:

                a. User input: The verbatim text provided by the user.
                b. User intent: The goal or purpose behind the user's input (e.g., "book a flight," "find a restaurant").
                c. Entities: Specific pieces of information relevant to the user's intent (e.g., dates, times, locations, reference numbers, phone numbers).
                d. Agent response: The actions taken and information provided by the agent in response to the user's input.

        Step 2:
                1. Use the extracted key information from step 1 to generate a very concise informative abstractive dialogue summary of the interaction.
                The summary should as follows:

                - Clearly state the intent of the customer's contact.
                - Highlight any issues or concerns raised by the customer.
                - Describe the solution(s) or action(s) provided to address the customer's needs.
                - Include all key entities and details
                2. Output the summary only. Erase any other content from your response except for the summary.
        Step 3:
                This template provides a structured approach for creating concise and informative summaries of user-agent dialogues.
                Fill out the template below using this user-agent dialogue: {dialogue}. Output the template in JSON format with explicit keys and values.

                {template}
    """,
    "Chain-of-Density": """
        Guidelines:

        You are given a human-agent dialogue to summarize. You will generate increasingly concise, entity-dense abstractive dialogue summaries of the user-agent dialogue.

        Repeat the following 2 steps 5 times.

        Step 1. Identify 1-3 informative Entities (";" delimited) from the dialogue which are missing from the previously generated summary.

        Step 2. Write a new, denser summary of identical length that covers every entity and detail from the previous summary plus the Missing Entities while maintaining the essence of a dialogue.

        A Missing Entity is:
        - Relevant: to the main dialogue.
        - Specific: descriptive yet concise (5 words or fewer).
        - Novel: not in the previous summary.
        - Faithful: present in the content piece.
        - Anywhere: located anywhere in the dialogue.

        Guidelines:

        The first summary should be long (4 sentences, - 45 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers to reach - 45 words.
        Make every word count: re-write the previous summary to improve flow and make space for additional entities.
        Make space with fusion, compression, and removal of uninformative phrases like "The ends the conversation".
        The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the original dialogue between user and agent.
        Missing entities can appear anywhere in the new summary.
        Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
        Remember, use the exact same number of words for each summary.
        Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary" from 1-4 and the 5th should "Missing_Entities" and "Final Summary"
    """,
    "Chain-of-Interaction": """
        You are given a user-agent dialogue, follow these 2 steps to fill out the chain of interaction template and create a detailed, concise summary.

        Step 1: Chain of interactions has 7 chains containing clear instructions and examples. Learn from the instructions and examples to understand how to create a chain of interaction summary.

        Chain [1] Interaction Details Extraction:
        - Identify the user intent.
        - List all issues or concerns raised by the user.
        - Note the agent's responses, actions taken, or information provided.
        - Extract key entities relevant to the interaction.

        Example:
        {
            "User Intent": "To find a cheap Portuguese restaurant in Cambridge.",
            "Issues/Concerns": [
                "Requested high-rated venues and European restaurants within city centre",
                "Needed address for chosen venue",
                "Preferred moderate price range for restaurant"
            ],
            "Agent Response": [
                "Provided details for 'The Funky Fun House'",
                "Shared address: 8 Mercers Row, Mercers Row Industrial Estate",
                "Recommended Galleria restaurant",
                "Helped with reservation booking",
                "Booked taxi for transportation",
                "Provided taxi driver's contact details"
            ],
            "Entities": ["Nandos", "South part of town (CB22HA)", "Thursday", "14:45", 8987889876]
        }

        Chain [2] Interaction Response Summary
        - Create a detailed yet concise summary capturing all key details from User Intent, Issues/Concerns, and Agent Response in Chain [1].

        Example:
        {
            "Summary of Interaction": "The user is looking for a cheap Portuguese restaurant in Cambridge, requesting high-rated venues and moderate prices. The agent detailed 'The Funky Fun House' (8 Mercers Row), recommended Galleria, booked a reservation, and arranged a taxi, providing the driver's contact."
        }

        Chain [3] Iterative Refinement
        - Start with the initial summary.
        - Add new informative entities from Chain [1] without increasing the summary length.

        Example:
        {
            "Reviewed Summary": "The user is looking for a cheap Portuguese restaurant in Cambridge (South part of town), requesting high-rated venues and moderate prices. The agent detailed 'The Funky Fun House' (8 Mercers Row), recommended Galleria, booked a reservation on Thursday, 14:45 (CB22HA), and arranged a taxi, providing the driver's contact (8987889876)."
        }

        Chain [4] Review and Adjust
        - Review the summary for conciseness, discourse coherence, coverage, rephrasing, readability, relevance, fluency, and informativeness.
        - Update the summary based on these criteria to ensure high quality and accuracy without increasing its size.

        Example:
        {
            "Reviewed Summary": "The user seeks a cheap, high-rated Portuguese restaurant in South Cambridge. The agent recommended 'The Funky Fun House' (8 Mercers Row) and Galleria, booked a reservation for Thursday at 14:45 (CB22HA), and arranged a taxi with driver contact (8987889876)."
        }

        Chain [5] Assess for Redundancy/Repetition
        - Remove any redundancy or repetitions.

        Example:
        {
            "Reviewed Summary": "The user seeks a cheap, high-rated Portuguese restaurant in South Cambridge. The agent recommended 'The Funky Fun House' (8 Mercers Row) and Galleria, booked a reservation for Thursday at 14:45 (CB22HA), and arranged a taxi with driver contact (8987889876)."
        }

        Chain [6] Fidelity/Hallucination Check
        - Review the summary for logical, factual, contextual, and intent fidelity with the Intent, Issues/Concerns, Agent Response, and entities in Chain [1].
        - Correct any of these types of hallucination inconsistencies.

        Example:
        {
            "Fidelity-Checked Summary": "The user seeks a cheap, high-rated Portuguese restaurant in South Cambridge. The agent detailed 'The Funky Fun House' (8 Mercers Row), recommended Galleria, booked a reservation for Thursday at 14:45 (CB22HA), and arranged a taxi (Nandos) with driver contact (8987889876)."
        }

        Chain [7] Enhance Brevity
        - Enhance brevity without losing any information.
        - Ensure the summary is coherent and self-contained.

        Example:
        {
            "Final Summary": "The user seeks a cheap, high-rated Portuguese restaurant in South Cambridge. The agent recommended 'The Funky Fun House' (8 Mercers Row) and Galleria, booked a Thursday 14:45 reservation (CB22HA), and arranged a taxi (Nandos) with the driver's contact (8987889876)."
        }

        Chain [8] Evaluation and Explainability
        - Definition of key Criteria:
            * Conciseness - Brevity, eliminating unnecessary details, and overall length reduction compared to the original dialogue.
            * Coverage - Includes all vital information from the original dialogue that represents the breadth of critical information, directly and indirectly related to the essential details of the important parts of the original dialogue.
            * Relevance - Focus on the most pertinent information from the original dialogue that excludes less critical or peripheral details by only directly addressing the main topics or issues that align with user intent and requests.
            * Rephrasing - The ability to demonstrate understanding through paraphrased and restructured content that assesses the use of novel phrasing, avoidance of direct copying, information restructuring, and interpretation of key ideas from the original dialogue.
            * Discourse Coherence — Evaluate the coherence of adjacent sentences (cause-effect relationships, salient entity consistency, thematic continuity) and the overall summary (clear structure, consistent information flow, easy-to-follow connections), ensuring information is presented logically and cohesively at the local (sentences) and global (overall) levels.
            * Fidelity — Maintain the original dialogue's meaning, context, facts, and intent while ensuring all information in the candidate logically follows (entailed)  in the source.
            * Readability -  Ease of comprehension and quick absorption based on clarity of language,  logical flow, sentence structure, and idea organization. Ensure avoidance of unnecessary jargon or complexity for efficient understanding and use.
            * Fluency - Free from all grammatical, spelling, and punctuation errors that impede smooth flow and comprehension.
            * Redundancy - avoiding unnecessary repetitions of multiple summaries, information, facts, entities, and ideas, ensuring the information is presented only once in a concise manner.
        - Thoroughly understand these key criteria and use them  to evaluate the final summary of chain [7] for each provided key criteria. For each statement, indicate your level of agreement in chain [8] using a 5-point scale where 1 = Strongly Disagree, 2 = Disagree, 3 = Neither Agree nor Disagree, 4 = Agree, and 5 = Strongly Agree.
        - Provide explicit evidence to justify your answer.

        Example:
            "Conciseness": {
            "Conciseness_Scale(1-5)": "5",
            "Conciseness_evidence": "The summary is significantly shorter than the original dialogue.
            }, ...

        Step 2: This template provides a structured approach for creating concise and informative summaries of user-agent dialogues. It guides through multiple chains, each with specific prompts to ensure detailed and comprehensive summaries without increasing length. Fill out the template below using this user-agent dialogue: {dialogue}. Output the template in JSON format with explicit keys and values.

        Chain of Interaction Prompts Template for Dialogue Summarization:
        {
            "Chain [1] Interaction Details Extraction": {
                "User Intent": "[User's inquiry, goal or purpose]",
                "Issues/Concerns": "[User's requests or input]",
                "Agent Response": "[Agent's response, action taken, or information provided]",
                "Entities": "[Key entities relevant to the interaction]"
            },
            "Chain [2] Interaction Response Summary": {
                "Summary of Interaction": "[Detailed yet concise summary capturing all key details]"
            },
            "Chain [3] Iterative Refinement": {
                "Iterative Summary": "[Add new informative entities from Chain [1] without increasing the summary length]"
            },
            "Chain [4] Review and Adjust": {
                "Reviewed Summary": "[Review the summary for conciseness, discourse coherence, coverage, rephrasing, readability, relevance, fluency, and informativeness. Update for accuracy.]"
            },
            "Chain [5] Assess for Redundancy/Repetition": {
                "Redundancy-Free Summary": "[Remove any redundancy or repetitions.]"
            },
            "Chain [6] Fidelity/Hallucination Check": {
                "Fidelity-Checked Summary": "[Review the summary for logical, factual, contextual, and intent fidelity. Correct any hallucinations.]"
            },
            "Chain [7] Enhance Brevity": {
                "Final Summary": "[Enhance brevity without losing any information, ensuring the summary is coherent and self-contained.]"
            },
            "Chain [8] Explainability": {
                "Conciseness":
                    "Conciseness Scale(1-5)": ["Evaluate to what degree you agree with this statement: Chain [7] final summary is noticeably more succinct than the original dialogue. Use the 5-point Likert scale, where 1 = Strongly Disagree and 5 = Strongly Agree"],
                    "Conciseness Evidence": ["Provide explicit evidence to justify your answer."]
                },
                "Coverage": {
                    "Coverage Scale(1-5)": ["Evaluate to what degree you agree with this statement: Chain [7] final summary covers all the key information in the original dialogue. Key information includes the user's request(s), related agent's response(s) or suggestion(s), the final outcome/action(s) taken, and any informative details like reference numbers, locations, or dates. Use the 5-point Likert scale, where 1 = Strongly Disagree and 5 = Strongly Agree."],
                    "Coverage Evidence": ["Provide explicit evidence to justify your answer."]
                },
                "Relevance": {
                    "Relevance Scale(1-5)": ["Evaluate to what degree you agree with this statement: Chain [7] final summary includes only the information and outcomes related to the user request(s). Use the 5-point Likert scale, where 1 = Strongly Disagree and 5 = Strongly Agree."],
                    "Relevance Evidence": ["Provide explicit evidence to justify your answer."]
                },
                "Rephrasing": {
                    "Rephrasing Scale(1-5)": ["Evaluate to what degree you agree with this statement: Chain [7] final summary explains ideas and presents information in its own words rather than copying directly from the original. Use the 5-point Likert scale, where 1 = Strongly Disagree and 5 = Strongly Agree.."],
                    "Rephrasing Evidence": ["Provide explicit evidence to justify your answer."]
                },
                "Discourse Coherence": {
                    "Discourse Coherence Scale(1-5)": ["Evaluate to what degree you agree with this statement: Chain [7] final summary presents information across sentences in a way that is connected, makes sense, and is easy to follow. Use the 5-point Likert scale, where 1 = Strongly Disagree and 5 = Strongly Agree."],
                    "Discourse Coherence Evidence": ["Provide explicit evidence to justify your answer."]
                },
                "Fidelity": {
                    "Fidelity Scale(1-5)": ["Evaluate to what degree you agree with this statement: Chain [7] final summary does not contain any information that contradicts or misrepresents the facts presented in the dialogue; it preserves the original message and intent without including information that is not in the dialogue. Use the 5-point Likert scale, where 1 = Strongly Disagree and 5 = Strongly Agree."],
                    "Fidelity Evidence": ["Provide explicit evidence to justify your answer."]
                },
                "Readability": {
                    "Readability Scale(1-5)": ["Evaluate to what degree you agree with this statement: Chain [7] final summary is easy to understand and uses clear language. Use the 5-point Likert scale, where 1 = Strongly Disagree and 5 = Strongly Agree."],
                    "Readability Evidence": ["Provide explicit evidence to justify your answer."]
                },
                "Fluency": {
                    "Fluency Scale(1-5)": ["Evaluate to what degree you agree with this statement: Chain [7] final summary does not have any grammatical mistakes. Use the 5-point Likert scale, where 1 = Strongly Disagree and 5 = Strongly Agree."],
                    "Fluency Evidence": ["Provide explicit evidence to justify your answer."]
                },
                "Redundancy": {
                    "Redundancy Scale(1-5)": ["Evaluate to what degree you agree with this statement: Chain [7] final summary avoids repeating information, facts, entities, and ideas. Use the 5-point Likert scale, where 1 = Strongly Disagree and 5 = Strongly Agree"],
                    "Redundancy Evidence": ["Provide explicit evidence to justify your answer."]
                }
            }
        }
    """
}

# Directories
MODEL_SAVE_DIR = "/n/w1-jlucas/Model"
DATA_SAVE_DIR = "/n/w1-jlucas/Data/todsum"
OUTPUT_SAVE_DIR = "/n/w1-jlucas/Output/Small_Param_LLM"
TRAIN_FILE_PATH = os.path.join(DATA_SAVE_DIR, 'train.json')
TEST_FILE_PATH = os.path.join(DATA_SAVE_DIR, 'test.json')

# Function to load few-shot examples using random sampling
def load_fewshot_examples_random(train_file_path, num_samples=2):
    with open(train_file_path, 'r') as file:
        train_data = json.load(file)
    
    fewshot_samples = random.sample(train_data, num_samples)
    formatted_samples = []
    
    for idx, sample in enumerate(fewshot_samples, start=1):
        dialogue = ' '.join(sample['text'])
        human_summary = ' '.join(sample['summary'])
        formatted_samples.append(f"{idx}. Dialogue: {dialogue} Human summary: {human_summary}")
    
    return "\n".join(formatted_samples)

# Function to load few-shot examples using BM25
def load_fewshot_examples_bm25(train_file_path, test_dialogue, num_samples=2):
    with open(train_file_path, 'r') as file:
        train_data = json.load(file)
    
    tokenized_corpus = [entry['text'] for entry in train_data]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = test_dialogue.split()
    top_n = bm25.get_top_n(tokenized_query, train_data, n=num_samples)
    
    formatted_samples = []
    for idx, sample in enumerate(top_n, start=1):
        dialogue = ' '.join(sample['text'])
        human_summary = ' '.join(sample['summary'])
        formatted_samples.append(f"{idx}. Dialogue: {dialogue} Human summary: {human_summary}")
    
    return "\n".join(formatted_samples)

# Function to find the top two most similar few-shot examples based on semantic similarity
def find_top_two_similar_examples(query, candidate_examples):
    # Encode the query sentence
    query_embedding = sbert_model.encode([query])
    
    # Encode the candidate examples
    candidate_embeddings = sbert_model.encode(candidate_examples)
    
    # Compute cosine similarity between query and candidate examples
    similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
    
    # Get the indices of the top two most similar examples
    top_two_indices = similarities.argsort()[-2:][::-1]
    
    # Retrieve the most similar examples
    top_two_examples = [candidate_examples[idx] for idx in top_two_indices]
    
    return top_two_examples

def load_fewshot_examples_semantic(train_file_path, test_dialogue, num_samples=2):
    with open(train_file_path, 'r') as file:
        train_data = json.load(file)
    
    candidate_examples = [{'text': ' '.join(entry['text']), 'summary': ' '.join(entry['summary'])} for entry in train_data]
    candidate_texts = [example['text'] for example in candidate_examples]
    top_two_examples = find_top_two_similar_examples(test_dialogue, candidate_texts)
    
    formatted_samples = []
    for idx, example in enumerate(top_two_examples, start=1):
        example_data = next(item for item in candidate_examples if item['text'] == example)
        dialogue = example_data['text']
        human_summary = example_data['summary']
        formatted_samples.append(f"{idx}. Dialogue: {dialogue} Human summary: {human_summary}")
    
    return "\n".join(formatted_samples)

# Function to generate the complete prompt
def generate_complete_prompt(instruct_prompt_sfx, fewshot_examples=None):
    if fewshot_examples:
        return impersona_prompt_pfx + instruct_prompt_sfx.format(fewshot_examples)
    return impersona_prompt_pfx + instruct_prompt_sfx

# Function to load data from the test file
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load data
todsum_data = load_json_file(TEST_FILE_PATH)

# Create a DataFrame with the required columns
data = []
for sample in todsum_data:
    dialog_id = sample['dialog_id']
    text = ' '.join(sample['text'])
    summary = ' '.join(sample['summary'])
    formatted_text = f'"""{text}"""'
    formatted_summary = f'"""{summary}"""'
    data.append({
        'dialog_id': dialog_id,
        'text': formatted_text,
        'summary': formatted_summary
    })

# Create a DataFrame with the required columns
df = pd.DataFrame(data, columns=['dialog_id', 'text', 'summary'])

# Calculate the number of samples per subset
num_samples = len(df)
samples_per_subset = num_samples // 4

# Split the DataFrame into 4 subsets
split_samples = [df.iloc[i:i + samples_per_subset] for i in range(0, num_samples, samples_per_subset)]

# Ensure the last subset includes any remaining samples
if len(df) % 4 != 0:
    split_samples[-1] = df.iloc[3 * samples_per_subset:]

# Print the number of samples in each subset for verification
for idx, subset in enumerate(split_samples):
    print(f"Subset {idx + 1}: {len(subset)} samples")


# Define the models and their respective max token lengths
models = {
    "llama-3-8B": ("meta-llama/Meta-Llama-3-8B-Instruct", 8192),
    "zephy7B": ("HuggingFaceH4/zephyr-7b-beta", 32768),
    "gemma-7B": ("google/gemma-7b-it", 8192),
    "Qwen-2-7B": ("Qwen/Qwen2-7B-Instruct", 32768),
    "mistral-7B": ("mistralai/Mistral-7B-Instruct-v0.3", 32768),
    # "PHI-medium-14B": ("microsoft/Phi-3-medium-128k-instruct", 131072),
    "openchat-8B": ("openchat/openchat-3.6-8b-20240522", 8192),
    "WizardLM-2-7B": ("lucyknada/microsoft_WizardLM-2-7B", 32768),
}

# Function to download and load model
def download_and_load_model(model_name, model_config, device):
    model_dir = f'{MODEL_SAVE_DIR}/{model_name}'
    
    if not os.path.exists(model_dir) or 'config.json' not in os.listdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        # print(f"Downloading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_config[0], token=HF_TOKEN, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_config[0], token=HF_TOKEN, trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        # print(f"Loading {model_name} from local directory...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

    model.to(device)
    model.half()  # Convert the model to 16-bit precision to save on memory and compute
    return model, tokenizer, model_config[1]

# Function to generate text
def generate_text(model, tokenizer, max_tokens, messages, device):
    # Setup the text generation pipeline with the specified model and tokenizer
    text_gen_pipeline = pipeline("text-generation", 
                                model=model, 
                                tokenizer=tokenizer,
                                torch_dtype=torch.bfloat16,
                                trust_remote_code=True,
                                device=device
                                )
    input_prompt = messages
    # Measure time for text generation
    start_time = datetime.datetime.now()
    # Use autocast only if using CUDA
    use_autocast = torch.cuda.is_available()
    with autocast() if use_autocast else nullcontext():
        try:
            generated_sequences = text_gen_pipeline(input_prompt,
                                                    max_length=max_tokens,
                                                    do_sample=True,
                                                    top_k=5, 
                                                    num_return_sequences=1,
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    return_full_text=False,
                                                    truncation=True,
                                                    temperature=0.8, 
                                                    top_p=0.95
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

def process_samples_on_gpu(samples, gpu_id, overall_progress):
    results = []
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    for index, row in tqdm(samples.iterrows(), total=samples.shape[0], desc=f"Processing samples on GPU {gpu_id}", leave=False):
        dialog_id = row['dialog_id']
        text_sample = "human-agent dialogue: " + row['text']
        summaries = []
        template = json.dumps({
            "User-agent Dialogues Template for Dialogue Summarization": {
                "Final Summary": "[Create the summary here only.]"
            }
        })
        
        for prompt_name, instruct_prompt_sfx in prompts.items():
            for model_name in models.keys():
                try:
                    # print(f"Model name: {model_name} Prompt: {prompt_name}")
                    response = json.dumps({'Summary': 'Response'})
                    if prompt_name == "Few-Random":
                        fewshot_examples = load_fewshot_examples_random(TRAIN_FILE_PATH)
                        messages = [
                            {"role": "user", 
                            "content": impersona_prompt_pfx + instruct_prompt_sfx.format(fewshot_examples=fewshot_examples, response=response, dialogue=text_sample, template=template)}
                        ]
                    elif prompt_name == "Few-BM25":
                        fewshot_examples = load_fewshot_examples_bm25(TRAIN_FILE_PATH, row['text'])
                        messages = [
                            {"role": "user", 
                            "content": impersona_prompt_pfx + instruct_prompt_sfx.format(fewshot_examples=fewshot_examples, response=response, dialogue=text_sample, template=template)}
                        ]
                    elif prompt_name == "Few-SBERT-SS":
                        fewshot_examples = load_fewshot_examples_semantic(TRAIN_FILE_PATH, row['text'])
                        messages = [
                            {"role": "user", 
                            "content": impersona_prompt_pfx + instruct_prompt_sfx.format(Semantic_examples=fewshot_examples, response=response, dialogue=text_sample, template=template)},
                        ]
                    elif prompt_name == "Chain-of-Interaction":
                        messages = [
                            {"role": "user", 
                            "content": impersona_prompt_pfx + instruct_prompt_sfx + text_sample}
                        ]
                    else:
                        messages = [
                            {"role": "user", 
                            "content": impersona_prompt_pfx + instruct_prompt_sfx.format(dialogue=text_sample, template=template)}
                        ]

                    model, tokenizer, max_tokens = download_and_load_model(model_name, models[model_name], device)

                    result = generate_text(model, tokenizer, max_tokens, messages, device)
                    summaries.append({
                        "model": model_name,
                        "prompt_type": prompt_name,
                        "time_taken": result["time_taken"],
                        "generated_text": result["generated_text"],
                        "original_dialogue": row["text"],
                        "human_summary": row["summary"]
                    })
                except torch.cuda.OutOfMemoryError:
                    print(f"Out of memory error for {model_name}. Trying on CPU...")
                    model, tokenizer, _ = download_and_load_model(model_name, models[model_name], 'cpu')  # Reload model and tokenizer on CPU
                    torch.cuda.empty_cache()
                    result = generate_text(model, tokenizer, max_tokens, messages, 'cpu')
                    summaries.append({
                        "model": model_name,
                        "prompt_type": prompt_name,
                        "time_taken": result["time_taken"],
                        "generated_text": result["generated_text"],
                        "original_dialogue": row["text"],
                        "human_summary": row["summary"]
                    })
                finally:
                    # Clear the GPU memory
                    if 'model' in locals():
                        del model
                    if 'tokenizer' in locals():
                        del tokenizer
                    torch.cuda.empty_cache()
        results.append({
            "dialog_id": dialog_id,
            "summaries": summaries
        })
        overall_progress.update(1)
        
    # Ensure directory exists
    os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)
    
    # Save results
    with open(f'{OUTPUT_SAVE_DIR}/smaller_model_outputs_gpu{gpu_id}.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Results saved to smaller_model_outputs_gpu{gpu_id}.json")

# Main processing function
def main():
    with tqdm(total=200, desc="Overall Progress") as overall_progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_samples_on_gpu, split_samples[i], i, overall_progress) for i in range(4)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Set environment variables for debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    main()
