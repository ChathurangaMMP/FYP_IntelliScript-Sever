import os
from transformers import pipeline
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_llama2_quantized(model_name):
    model = AutoModelForCausalLM.from_pretrained(
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def create_document_chunks(file_path, chunk_size):
    file_reader = SimpleDirectoryReader(
        input_files=[file_path], encoding='utf-8')
    documents = file_reader.load_data()

    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
    nodes = node_parser.get_nodes_from_documents(documents)

    # by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
    node_count = 1
    for idx, node in enumerate(nodes):
        node.id_ = f"node_{node_count}"
        node_count += 1

    return nodes


def generate_summary_prompt(query):
    text = """INSTRUCTION: Extractive summarize the given context topic-wise.

    Follow the additional instructions below.
      1. Consider only the given context.
      2. Do not add data from your knowledge.
      3. Do not repeat given context as it is in the output.
      4. Do not add texts after the main output. (Note:)
      5. Do not add 'Topic' label in the output.
      \n\n"""
    text += f'CONTEXT: \n{query}\n\n'
    text += 'OUTPUT: '
    return {'text': text}


def get_summarized(query):
    prompt = generate_summary_prompt(query)['text']
    output_response = llama2_pipeline(prompt)[0]['generated_text']
    output_text = output_response.split('\nOUTPUT:')[1]
    return output_text


source_dir = "../../Extracted-text-CBSL-data-new/"
output_dir = "Extracted-text-summarized/"
count = 0
error_count = 0
short_count = 0

llama2, llama2_tokenizer = load_llama2_quantized(
    'NousResearch/Llama-2-7b-chat-hf')

llama2_kwargs = {
    "max_new_tokens": 2000,
    "temperature": 0.01,
    # "num_return_sequences": 1,
    "top_k": 10,
    "top_p": 0.9,
    "batch_size": 1,
    "do_sample": True,
    # "no_repeat_ngram_size":4,
    # "repetition_penalty" : 1.5
}

llama2_pipeline = pipeline(
    "text-generation",
    model=llama2,
    tokenizer=llama2_tokenizer,
    **llama2_kwargs
)


for root, directories, files in os.walk(source_dir):
    for directory in directories:
        # Create corresponding directories in the output directory
        output_subdir = os.path.join(
            output_dir, os.path.relpath(root, source_dir), directory)
        os.makedirs(output_subdir, exist_ok=True)

    for file in files:
        try:
            input_file_path = os.path.join(root, file)

            with open(input_file_path, 'r', encoding='utf-8') as infile:
                data = infile.read()

            if len(data) > 1500:
                # Determine the output file path
                output_file_path = os.path.join(
                    output_dir, os.path.relpath(root, source_dir), file[:file.rfind('.')+1] + 'txt')

                chunks = create_document_chunks(input_file_path, 512)
                full_text = ''
                for chunk in chunks:
                    summary = get_summarized(chunk.text)
                    full_text += f'{summary}\n'

                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(full_text)

                count += 1

                print(f'File {count} - {input_file_path}')

            else:
                short_count += 1

        except:
            error_count += 1

for root, dirs, files in os.walk(output_dir, topdown=False):
    for dir in dirs:
        folder_path = os.path.join(root, dir)
        if not os.listdir(folder_path):  # Check if folder is empty
            os.rmdir(folder_path)
            print(f"Deleted empty folder: {folder_path}")

print(f"Success: {count} - Errors: {error_count}, {short_count}")
