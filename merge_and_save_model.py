from huggingface_hub import login
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

for param in phi2.parameters():
    param.requires_grad = True

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

instruction_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
    "output-model/checkpoint-5000",
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

hf_model_repo = 'mmpc/microsoft-phi-2-squad2-qg'
# push merged model to the hub
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)

print("New model is uploaded.")
