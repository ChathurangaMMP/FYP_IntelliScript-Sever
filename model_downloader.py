# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyPixel/Llama-2-7B-bf16-sharded"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

save_directory = "llama-2-7b-bf16/"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
