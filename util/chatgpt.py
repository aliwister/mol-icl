from openai import OpenAI
import numpy as np
import pandas as pd

def run_chatgpt(args, prompts_all, file):
    client = OpenAI(
        api_key="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",
    )   
    def get_response_zero(prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an intelligent SQL Code assistant who effectively translates the intent and logic of the SQL queries into natural language that is easy to understand."},
                {"role": "user", "content": """Convert the given SQL query into a clear and concise natural language query limited to 1 sentence.  Ensure that the request accurately represents the actions specified in the SQL query and is easy to understand for someone without technical knowledge of SQL.
                Input: """ + prompt},
            ],
            temperature=1,
            max_tokens=150,
            top_p=1
        )
        return response.choices[0].message.content

    def get_response(prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=1,
            max_tokens=150,
            top_p=1
        )
        return response.choices[0].message.content

    responses = []
    for prompt in prompts_all:
        if (args.method == "zero-gpt"):
            response = get_response_zero(prompt)
        else:
            response = get_response(prompt)
        print(f"Prompt: {prompt}\nResponse: {response}\n")
        responses.append(response)
    df = pd.DataFrame(responses)
    df.to_csv(file, index=False)