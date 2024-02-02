from huggingface_hub import snapshot_download

snapshot_download(repo_id="NousResearch/Llama-2-13b-chat-hf",
                  local_dir="Llama-2-13b-chat-hf")

# # Load model directly
# from huggingface_hub import hf_hub_download


# # files = [
# #     ".gitattributes",
# #     "config.json",
# #     "generation_config.json",
# #     "pytorch_model-00001-of-00014.bin",
# #     "pytorch_model-00002-of-00014.bin",
# #     "pytorch_model-00003-of-00014.bin",
# #     "pytorch_model-00004-of-00014.bin",
# #     "pytorch_model-00005-of-00014.bin",
# #     "pytorch_model-00006-of-00014.bin",
# #     "pytorch_model-00007-of-00014.bin",
# #     "pytorch_model-00008-of-00014.bin",
# #     "pytorch_model-00009-of-00014.bin",
# #     "pytorch_model-00010-of-00014.bin",
# #     "pytorch_model-00011-of-00014.bin",
# #     "pytorch_model-00012-of-00014.bin",
# #     "pytorch_model-00013-of-00014.bin",
# #     "pytorch_model-00014-of-00014.bin",
# #     "pytorch_model.bin.index.json",
# #     "special_tokens_map.json",
# #     "tokenizer.json",
# #     "tokenizer.model",
# #     "tokenizer_config.json"
# # ]

# repo_id = "microsoft/phi-2"
# output_dir = "microsoft/phi-2"

# hf_hub_download(repo_id=repo_id, local_dir=output_dir)
