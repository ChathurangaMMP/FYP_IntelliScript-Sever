from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from transformers import pipeline
import json
import os

node_count = 1
folder_path = "../Extracted-text-CBSL-data/FINANCIAL SYSTEM"  # TO_CHANGE
model_name = "../Llama-2-7b-chat-hf"  # TO_CHANGE
chunk_s = 512  # TO_CHANGE

ts_file_path = f"ts-{chunk_s}.json"
success_txt_path = f"success-files-ts-{chunk_s}.txt"
success_total_nodes_txt_path = f"success-total-node-details-ts-{chunk_s}.txt"
error_file_path = f"errors-ts-{chunk_s}.json"

llama2 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    ),
    # revision='834565c23f9b28b96ccbeabe614dd906b6db551a'
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def define_pipeline():
    question_generation_kwargs = {
        "max_new_tokens": 2500,
        "temperature": 0.7,
        "do_sample": True,
        # "no_repeat_ngram_size":4,
        # "repetition_penalty" : 1.5
    }

    llama2_QA_pipeline = pipeline(
        "text-generation",
        model=llama2,
        tokenizer=tokenizer,
        **question_generation_kwargs
    )

    return llama2_QA_pipeline


def get_num_tokens(prompt):
    tokenized_text = tokenizer(prompt, return_tensors="pt")
    num_tokens = tokenized_text["input_ids"].shape[1]
    return num_tokens


def generate_label_data_prompt(context):
    text = '''INSTRUCTION: You are a good labeled dataset generator. Please generate labeled data points(query-response pairs) only from the given context.
        - Cover all the information there in the context. Generate as much as pairs you can.
        - Do not repeat the same query-response pair.
        - Use only the given context, do not add your own generated data that are not present in the context.
        - Give the OUTPUT in JSON format with each dictionary contains query and response. Do not generate any text after the JSON in OUTPUT.
        - Do not generate incomplete dictionaries for the output. Always give a complete JSON

        '''

    text += f"CONTEXT: {context}\n\n"
    # text += f"QUESTION: {data_point['question']}\n\n" topic-content
    text += "OUTPUT: "
    return {'text': text}


def generate_topic_summary_prompt(context):
    text = '''INSTRUCTION: You are a good summarizer. Summarize the given context by the topics and sub topics it has.
          - Cover all the information there in the context. Generate as much as pairs you can.
          - Only consider the given context.
          - Give the OUTPUT in JSON format with each dictionary contains topic and summary. Do not generate any text after the JSON in OUTPUT.
          - Do not generate incomplete dictionaries for the output. Always give a complete JSON
          
          '''

    text += f"CONTEXT: {context}\n\n"
    # text += f"QUESTION: {data_point['question']}\n\n"
    text += "OUTPUT: "
    return {'text': text}


def remove_tailed_text(response):
    i = 0
    k = ''
    for l in response[::-1]:
        if l != ']' and l != '}':
            i += 1
        else:
            if l == ']':
                k = ']'
            else:
                k = '}]'
            break

    if i == 0:
        return response
    else:
        return response[:i*(-1)]+k


llama2_QA_pipeline = define_pipeline()

for root, directories, files in os.walk(folder_path):
    for file in files:
        try:
            file_path = os.path.join(root, file)

            file_reader = SimpleDirectoryReader(
                input_files=[file_path], encoding='utf-8')
            documents = file_reader.load_data()

            node_parser = SentenceSplitter(chunk_size=chunk_s)
            nodes = node_parser.get_nodes_from_documents(documents)

            # by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
            temp_node_count = 0
            for idx, node in enumerate(nodes):
                node.id_ = f"node_{node_count}"
                node_count += 1
                temp_node_count += 1

            temp_success_nodes = 0
            temp_error_nodes = 0
            for node in nodes:
                try:
                    prompt_ts = generate_topic_summary_prompt(str(node.text))[
                        'text']
                    output_response_ts = llama2_QA_pipeline(
                        prompt_ts)[0]['generated_text']
                    output_text_ts = output_response_ts.split('\nOUTPUT:')[1]

                    json_ts = json.loads(remove_tailed_text(output_text_ts))
                    write_dict_ts = {
                        'source': file_path[:-4], 'context': node.text, 'ts': json_ts}

                    # Open the file in append mode
                    with open(ts_file_path, "a") as json_file_ts:
                        # Serialize the JSON data
                        json_string_ts = json.dumps(write_dict_ts)
                        json_file_ts.write(json_string_ts + "\n")

                    with open(success_txt_path, 'a') as success_file:
                        success_file.write(f'{node.id_}-{file_path}')
                        print(f'{node.id_}-{file_path}')

                    temp_success_nodes += 1

                except Exception as e:
                    print(
                        f'Node error - {node.id_} - {os.path.join(root, file)} - {e}')
                    temp_error_nodes += 1
                    with open(error_file_path, 'a') as error_file:
                        json_string_error = json.dumps(
                            {'source': file_path[:-4], 'id': node.id_, 'context': node.text})
                        error_file.write(json_string_error + "\n")

            with open(success_total_nodes_txt_path, 'a') as success_nodes_file:
                node_message = f'{file_path}-t-{temp_node_count}-s-{temp_success_nodes}-e-{temp_error_nodes}'
                success_nodes_file.write(node_message)
                print(node_message)

        except Exception as e:
            print(f'Parse file error - {os.path.join(root, file)} - {e}')
