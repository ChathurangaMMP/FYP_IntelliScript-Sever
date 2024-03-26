import os
import json
from transformers import AutoTokenizer

model_name = 'NousResearch/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


def get_num_tokens(text_chunk):
    tokenized_text = tokenizer(text_chunk, return_tensors='pt')
    num_tokens = tokenized_text['input_ids'].shape[1]
    return num_tokens


# Paths to the two folder structures
source_files = '../../Extracted-text-data'
json_pdf_output_path = 'pdf-tokens.json'
json_excel_output_path = 'excel-tokens.json'
count = 0
errors = 0
pdf_tokens_data = {}
excel_tokens_data = {}

# Iterate over the text files in folder structure 2
for root, dirs, files in os.walk(source_files):
    for file in files:
        try:
            input_file_path = os.path.join(root, file)

            with open(input_file_path, 'r', encoding='utf-8') as infile:
                data = infile.read()

            if '<new sheet>' in data:
                data_sheets = data.split('<new sheet>')
                temp = {}
                for i in range(len(data_sheets)):
                    temp[i] = get_num_tokens(data_sheets[i])
                excel_tokens_data[input_file_path] = temp

            else:
                pdf_tokens_data[input_file_path] = get_num_tokens(data)

            count += 1
            if count % 100 == 0:
                print(count)

        except Exception as e:
            print(e)
            errors += 1

with open(json_pdf_output_path, 'w') as json_pdffile:
    json.dump(pdf_tokens_data, json_pdffile)

with open(json_excel_output_path, 'w') as json_excelfile:
    json.dump(excel_tokens_data, json_excelfile)

print("Dictionary written to JSON file successfully.")
