import json
import pandas as pd 

def load_cosql_dataset(dataset_train):
    # Open and read the JSON file
    with open(dataset_train, 'r') as file:
        data = json.load(file)

    final_values = [item["final"] for item in data]

    # Print the contents of the JSON file
    df = pd.DataFrame(final_values, columns=['utterance', 'query'])
    return df

def load_spider_dataset(dataset_train):
    # Open and read the JSON file
    with open(dataset_train, 'r') as file:
        data = json.load(file)

    #final_values = [item["final"] for item in data]

    # Print the contents of the JSON file
    df = pd.DataFrame(data, columns=['question', 'query'])
    return df
