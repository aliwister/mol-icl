import torch, os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformers import AutoModel, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def padarray(A, size, value=0):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values = value)

def preprocess_each_sentence(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(
        sentence, truncation=True, max_length=max_seq_len,
        padding='max_length', return_tensors='np')
    
    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()

    sentence_tokens_ids = padarray(input_ids, max_seq_len)
    sentence_masks = padarray(attention_mask, max_seq_len)
    return [sentence_tokens_ids, sentence_masks]


# Function to batch the text inputs and get outputs from the model
def get_batched_text_outputs(device, descriptions, text_tokenizer, text_model, max_seq_len, file_type="none", batch_size=16, is_load_saved=False, is_save=False):
    # Prepare the token IDs and attention masks using the provided tokenizer
    if (is_load_saved):
        file_path = f"./input/scibert/{file_type}.npy"
        if (os.path.exists(file_path)):
            all_outputs = np.load(file_path)
            return torch.cat(all_outputs)

    description_tokens_ids, description_masks = prepare_text_tokens(
        device, descriptions, text_tokenizer, max_seq_len
    )

    # Create a dataset and DataLoader to handle batching
    dataset = TensorDataset(description_tokens_ids, description_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Store outputs in a list to concatenate later
    all_outputs = []

    # Process each batch through the text model
    for i, batch in enumerate(dataloader):
        print (f"batch: {i}")
        batch_input_ids, batch_attention_masks = batch
        with torch.no_grad():  # Disable gradient calculation for inference
            output = text_model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
        
        all_outputs.append(output['pooler_output'])
    if (is_save):
        np.save(file_path, all_outputs)
    return torch.cat(all_outputs)



# This is for BERT
def prepare_text_tokens(device, description, tokenizer, max_seq_len):
    B = len(description)

    tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = [o[0] for o in tokens_outputs]
    masks = np.array([o[1] for o in tokens_outputs])
    tokens_ids = np.array(tokens_ids)
    tokens_ids = torch.from_numpy(tokens_ids).long().to(device)
    masks = torch.from_numpy(masks).bool().to(device)
    return tokens_ids, masks


def get_tokenizer(bert = 'scibert'):
    if (bert == 'bert'):
        text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        text_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
        return text_tokenizer, text_model

    pretrained_SciBERT_folder = os.path.join('../data', 'pretrained_SciBERT')
    text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder)
    text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder).to(device)
    return text_tokenizer, text_model

