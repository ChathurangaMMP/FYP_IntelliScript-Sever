from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM
from transformers import TrainingArguments
from peft import LoraConfig
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
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False
    # revision='834565c23f9b28b96ccbeabe614dd906b6db551a'
)

# for param in phi2.parameters():
#     param.requires_grad = True

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

phi2.config.pad_token_id = tokenizer.eos_token_id


dataset = "squad_v2"

data = load_dataset(dataset, split="train")
data = data.shuffle(seed=42)

vdata = load_dataset(dataset, split="validation")
val_data = vdata.shuffle(seed=42)

train_test_data_1 = data.train_test_split(test_size=0.2, seed=42)
train_data = train_test_data_1['train']
other_data = train_test_data_1['test']

train_test_data_2 = other_data.train_test_split(test_size=0.5, seed=42)
val_data_addon = train_test_data_2['train']
test_data = train_test_data_2['test']

val_data = concatenate_datasets([val_data, val_data_addon])


def filter_dataset(dataset):
    df = pd.DataFrame(dataset)

    used_data = []
    to_delete = []
    for i in df.index:
        # Check condition for deletion
        if df.loc[i, 'context'] in used_data:
            to_delete.append(i)
        else:
            used_data.append(df.loc[i, 'context'])

    # Delete rows based on collected indices
    df.drop(to_delete, inplace=True)

    return Dataset.from_pandas(df)


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
    text = 'INSTRUCTION: Answer the following question based on the given context, providing a concise and fact-based response.\n\n'
    text += 'CONSTRAINTS: Do not generate additional text beyond the answer.\n\n'
    text += f'CONTEXT: {data_point["context"]}\n\n'
    text += f'QUESTION: {data_point["question"]}\n\n'
    text += f'ANSWER: {data_point["answers"]}\n\n'
    return {'text': text}


train_data_filtered = filter_dataset(train_data)
val_data_filtered = filter_dataset(val_data)

train_data_mapped = train_data_filtered.map(generate_prompt_for_finetuning)
val_data_mapped = val_data_filtered.map(generate_prompt_for_finetuning)


def slice_dataset(dataset, num_rows):
    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame(dataset)

    # Slice the first 100 rows
    subset_df = df.head(num_rows)

    # Convert the subset DataFrame back to a datasets Dataset
    subset_dataset = Dataset.from_pandas(subset_df)

    # Print information about the subset dataset
    return subset_dataset


training_data = slice_dataset(train_data_mapped, 10000)
validation_data = slice_dataset(val_data_mapped, 2000)

# we set our lora config to be the same as qlora
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    #  The modules to apply the LoRA update matrices.
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'dense',
        'fc1',
        'fc2',
    ],
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
    save_steps=1000,
    logging_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    # basically just train for 5 epochs, you should train for longer
    max_steps=int(len(training_data) * 1),
    warmup_steps=150,
    # bf16=True,
    # tf32=True,
    gradient_checkpointing=True,
    # max_grad_norm=0.3,  # from the paper
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
    max_seq_length=2096,
    dataset_num_proc=os.cpu_count(),
)

trainer.train()

trainer.save_model()


instruction_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
    training_args.output_dir,
    torch_dtype=torch.float16,
    # torch_dtype='auto',
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

hf_model_repo = 'mmpc/phi-2-squad'
# push merged model to the hub
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)

print("New model is uploaded.")
