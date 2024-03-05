from peft import LoraConfig
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import numpy as np
import os
import torch

model_name = "../Llama-2-7b-chat-hf"

llama2 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    load_in_4bit=True,
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    # ),
    # revision='834565c23f9b28b96ccbeabe614dd906b6db551a'
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


documents_fs = SimpleDirectoryReader(
    input_dir="../Extracted-text-CBSL-data/FINANCIAL SYSTEM", required_exts=['.txt'], recursive=True).load_data()
documents_law = SimpleDirectoryReader(
    input_dir="../Extracted-text-CBSL-data/LAWS", required_exts=['.txt'], recursive=True).load_data()


node_parser = SentenceSplitter(chunk_size=512)
nodes_fs = node_parser.get_nodes_from_documents(documents_fs)
nodes_law = node_parser.get_nodes_from_documents(documents_law)
nodes = nodes_fs+nodes_law

node_texts = [tokenizer(t.text) for t in nodes]

df = pd.DataFrame(node_texts)

dataset = Dataset.from_pandas(df.rename(columns={0: "labels"}), split="train")

train_test_data = dataset.train_test_split(test_size=0.2, seed=42)
train_data = train_test_data['train']
val_data = train_test_data['test']


tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

llama2.add_adapter(peft_config)

# Login to the Hugging Face Hub
login(token="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")

training_args = TrainingArguments(
    output_dir="llama-2-7b-clm-model",
    evaluation_strategy="epoch",
    learning_rate=1e-6,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=llama2,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
)

trainer.train()

llama2.save_pretrained('llama-2-7b-clm-model')
