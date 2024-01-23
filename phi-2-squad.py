from huggingface_hub import login
from peft import AutoPeftModelForCausalLM
from transformers import TrainingArguments
from peft import LoraConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import os

"""### Load model"""

model_name = "microsoft-phi-2"

phi2 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    # revision='834565c23f9b28b96ccbeabe614dd906b6db551a'
)

for param in phi2.parameters():
    param.requires_grad = True

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

phi2.config.pad_token_id = tokenizer.eos_token_id


dataset = "squad_v2"

data = load_dataset(dataset, split="train")
data = data.shuffle(seed=42)

vdata = load_dataset(dataset, split="validation")
val_data = vdata.shuffle(seed=42)

train_test_data = data.train_test_split(test_size=0.1, seed=42)
train_data = train_test_data['train']
test_data = train_test_data['test']

"""### Define a function to test the similarity between responses"""


def paragraph_similarity(p1, p2):
    # Load the pre-trained model
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-MiniLM-L6-v2")

    # Encode the paragraphs into vectors
    paragraph1_vector = model.encode(p1)
    paragraph2_vector = model.encode(p2)

    # Calculate cosine similarity between the vectors
    cosine_similarity = np.dot(paragraph1_vector, paragraph2_vector) / (
        np.linalg.norm(paragraph1_vector) * np.linalg.norm(paragraph2_vector))

    # Print the similarity score
    return f"{cosine_similarity:.4f}"

    # Interpretation: Higher cosine similarity (closer to 1) indicates more semantically similar paragraphs.


def generate_prompt_for_finetuning(data_point):
    # Samples with additional context info.
    text = 'Instruction: Answer the following questions based on the given context.\n\n'
    text += f'Context: {data_point["context"]}\n\n'
    text += f'Question: {data_point["question"]}\n\n'
    text += f'Answer: {data_point["answers"]}\n\n'
    return {'text': text}


training_data = train_data.map(generate_prompt_for_finetuning)
validation_data = val_data.map(generate_prompt_for_finetuning)

# we set our lora config to be the same as qlora
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    #  The modules to apply the LoRA update matrices.
    target_modules=["Wqkv", "fc1", "fc2"],
    task_type="CAUSAL_LM"
)

"""### Training Args"""


output_dir = 'output-model/'

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    do_eval=True,
    auto_find_batch_size=True,
    # per_device_train_batch_size=16,
    log_level="debug",
    optim="paged_adamw_8bit",
    save_steps=20000,
    logging_steps=10000,
    learning_rate=3e-5,
    weight_decay=0.01,
    # basically just train for 5 epochs, you should train for longer
    max_steps=int(len(training_data) * 1),
    warmup_steps=350,
    # bf16=True,
    # tf32=True,
    gradient_checkpointing=True,
    max_grad_norm=0.3,  # from the paper
    lr_scheduler_type="reduce_lr_on_plateau",
)

"""### Train"""

trainer = SFTTrainer(
    model=phi2,
    args=training_args,
    peft_config=lora_config,
    tokenizer=tokenizer,
    dataset_text_field='text',
    train_dataset=training_data,
    eval_dataset=validation_data,
    max_seq_length=4096,
    dataset_num_proc=os.cpu_count(),
)

trainer.train()

trainer.save_model()


instruction_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
    training_args.output_dir,
    # torch_dtype=torch.float16,
    torch_dtype='auto',
    trust_remote_code=True,
    device_map='auto',
    offload_folder="offload/"
)

merged_model = instruction_tuned_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")


# Login to the Hugging Face Hub
login(token="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")

hf_model_repo = 'mmpc/phi-2-squad2'
# push merged model to the hub
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)

print("New model is uploaded.")
