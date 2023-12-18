from transformers import TrainingArguments
import transformers
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import peft
import time
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import os

"""### BitsAndBytesConfig"""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

"""### Load model"""

model_name = "Deci/DeciLM-7B"

decilm = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

decilm.config.pad_token_id = tokenizer.eos_token_id

"""### Load dataset


"""


dataset = "gbharti/finance-alpaca"

data = load_dataset(dataset, split="train")
data = data.shuffle(seed=42)
data_finance = data.train_test_split(test_size=0.1, seed=42)

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


"""### Define a function to generate the inferencing prompt"""


def generate_prompt(data_point):
    # Samples with additional context info.
    if data_point['input']:
        text = 'Below is an instruction that describes a task, paired with an Input parameter that provides' \
               ' further context. Write a response that appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Input:\n{data_point["input"]}\n\n'
        text += f'### OUTPUT:'
        return {'text': text}

    # Without
    else:
        text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### OUTPUT:'
        return {'text': text}


"""### Define a function to generate the response"""

# def generate_response(model, prompt):
#     # Tokenize the input
#     input_ids = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
#     # Run the model to infere an output
#     outputs = model.generate(**input_ids, max_length=500)

#     output_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#     return output_response


generation_kwargs = {
    "max_new_tokens": 200,
    "early_stopping": True,
    "num_beams": 5,
    "temperature": 0.001,
    "do_sample": True,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.5
}

decilm_tuned_pipeline = pipeline(
    "text-generation",
    model=decilm,
    tokenizer=tokenizer,
    **generation_kwargs
)

generate_prompt(data_finance['test'][0])

"""### Generate the responses for the first 10 entries in the test dataset and measure the similarity"""


all_responses = {}
print('Index - Similarity score - Inference time(s)')
for i in range(10):
    start_time = time.time()
    prompt = generate_prompt(data_finance['test'][i])['text']
    output_response = decilm_tuned_pipeline(prompt, return_full_text=True)[
        0]['generated_text']
    output_response = output_response.split("### OUTPUT:")[1]
    all_responses[i] = [output_response, data_finance['test'][i]['output']]

    print(
        f"{i} - {paragraph_similarity(output_response, data_finance['test'][i]['output'])} - {round(time.time()-start_time, 2)}")

for k in all_responses:
    print(k)
    print("Generated:", all_responses[k][0])
    print()
    print("Dataset:", all_responses[k][1])
    print('-----------------------------------------------------------------------\n')

"""### Define a function to pre-process the data for fine-tuning"""


def generate_prompt_for_finetuning(data_point):
    # Samples with additional context info.
    if data_point['input']:
        text = 'Below is an instruction that describes a task, paired with an Input parameter that provides' \
               ' further context. Write a response that appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Input:\n{data_point["input"]}\n\n'
        text += f'### OUTPUT:\n{data_point["output"]}\n\n'
        return {'text': text}

    # Without
    else:
        text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### OUTPUT:\n{data_point["output"]}\n\n'
        return {'text': text}


training_data = data_finance['train'].map(generate_prompt_for_finetuning)
testing_data = data_finance['test'].map(generate_prompt_for_finetuning)


# we set our lora config to be the same as qlora
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    #  The modules to apply the LoRA update matrices.
    target_modules=["gate_proj", "down_proj", "up_proj"],
    task_type="CAUSAL_LM"
)

"""### Prepare model for peft"""

decilm = prepare_model_for_kbit_training(decilm)

decilm = get_peft_model(decilm, lora_config)

"""### Training Args"""


output_dir = 'output-model/'

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    do_eval=True,
    auto_find_batch_size=True,
    # per_device_train_batch_size=16,
    log_level="debug",
    optim="paged_adamw_32bit",
    save_steps=10000,
    logging_steps=10000,
    learning_rate=3e-4,
    weight_decay=0.01,
    # basically just train for 5 epochs, you should train for longer
    max_steps=int(len(training_data) * 1),
    warmup_steps=150,
    # bf16=True,
    # tf32=True,
    gradient_checkpointing=True,
    max_grad_norm=0.3,  # from the paper
    lr_scheduler_type="reduce_lr_on_plateau",
)

"""### Train"""

trainer = SFTTrainer(
    model=decilm,
    args=training_args,
    peft_config=lora_config,
    tokenizer=tokenizer,
    dataset_text_field='text',
    train_dataset=training_data,
    eval_dataset=testing_data,
    max_seq_length=4096,
    dataset_num_proc=os.cpu_count(),
)

trainer.train()

trainer.save_model()
