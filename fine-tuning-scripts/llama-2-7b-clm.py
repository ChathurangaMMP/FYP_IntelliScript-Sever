from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
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


documents_fs = SimpleDirectoryReader(
    input_dir="../Extracted-text-CBSL-data/FINANCIAL SYSTEM", required_exts=['.txt'], recursive=True).load_data()
# documents_law = SimpleDirectoryReader(
#     input_dir="../Extracted-text-CBSL-data/LAWS", required_exts=['.txt'], recursive=True).load_data()


node_parser = SentenceSplitter(chunk_size=512)
nodes_fs = node_parser.get_nodes_from_documents(documents_fs)
# nodes_law = node_parser.get_nodes_from_documents(documents_law)
# nodes = nodes_fs+nodes_law

nodes = nodes_fs

node_texts = [tokenizer(t.text) for t in nodes]

df = pd.DataFrame(node_texts)

dataset = Dataset.from_pandas(df.rename(columns={0: "labels"}), split="train")

train_test_data = dataset.train_test_split(test_size=0.2, seed=42)
train_data = train_test_data['train']
val_data = train_test_data['test']


tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


peft_config = LoraConfig(
    r=16,  # dimension of the updated matrices
    lora_alpha=64,  # parameter for scaling
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    lora_dropout=0.1,  # dropout probability for layers
    bias="none",
    task_type="CAUSAL_LM",
)

llama2.gradient_checkpointing_enable()
llama2 = prepare_model_for_kbit_training(llama2)
llama2 = get_peft_model(llama2, peft_config)

# Login to the Hugging Face Hub
login(token="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")

output_dir = 'llama-2-7b-clm-model'
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    do_eval=True,
    auto_find_batch_size=True,  # this is risky
    # per_device_train_batch_size=16,
    log_level="debug",
    # optim="paged_adamw_32bit",
    optim="paged_adamw_32bit",
    save_steps=1000,
    logging_steps=100,
    learning_rate=1e-6,
    weight_decay=0.01,
    # basically just train for 5 epochs, you should train for longer
    max_steps=int(len(train_data) * 1),
    warmup_steps=150,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=llama2,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
)

trainer.train()

trainer.save_state()
trainer.model.save_pretrained('llama-2-7b-clm-model')


tuned_llama2_model = AutoPeftModelForCausalLM.from_pretrained(
    training_args.output_dir,
    torch_dtype=torch.float16,
    # torch_dtype='auto',
    trust_remote_code=True,
    device_map='auto',
    offload_folder="offload/"
)

merged_model = tuned_llama2_model.merge_and_unload()

merged_model.save_pretrained("llama-2-merged_model", safe_serialization=True)
tokenizer.save_pretrained("llama-2-merged_model")

hf_model_repo = 'mmpc/llama-2-7b-clm'
# push merged model to the hub
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)
