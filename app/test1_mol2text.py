from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("QizhiPei/biot5-base-mol2text", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained('QizhiPei/biot5-base-mol2text')

task_definition = 'Definition: You are given a molecule SELFIES. Your job is to generate the molecule description in English that fits the molecule SELFIES.\n\n'
selfies_input = '[C][C][Branch1][C][O][C][C][=Branch1][C][=O][C][=Branch1][C][=O][O-1]'
task_input = f'Now complete the following example -\nInput: <bom>{selfies_input}<eom>\nOutput: '

model_input = task_definition + task_input
input_ids = tokenizer(model_input, return_tensors="pt").input_ids

generation_config = model.generation_config
generation_config.max_length = 512
generation_config.num_beams = 1

outputs = model.generate(input_ids, generation_config=generation_config)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))