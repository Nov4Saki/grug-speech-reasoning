import os
import shutil
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

base_id = "Nanbeige/Nanbeige4.2-3B"
adapter_dir = "/content/drive/MyDrive/nanbeige4.2-3b-grug/final_adapter"
merged_dir = "/content/merged_nanbeige4.2_3b"
hf_repo = "Novasaki/Nanbeige4.2-3B-GrugSpeech-Native"
model_name = "Nanbeige4.2-3B-GrugSpeech"

print("1. Loading base model in bfloat16 for weight merge...")
tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=token, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    token=token,
    dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)

print("2. Merging LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_dir)
merged = model.merge_and_unload()
merged._tied_weights_keys = {}
os.makedirs(merged_dir, exist_ok=True)
merged.save_pretrained(merged_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_dir)
print(f"Merged model saved successfully to {merged_dir}!")

del model
del base_model
del merged
torch.cuda.empty_cache()

# Upload adapter and merged model card to HF
create_repo(repo_id=hf_repo, repo_type="model", token=token, exist_ok=True)
api.upload_folder(
    folder_path=adapter_dir,
    repo_id=hf_repo,
    repo_type="model",
    path_in_repo="lora_adapter",
    commit_message="Upload Nanbeige 4.2-3B Grug LoRA adapter"
)
print(f"Uploaded LoRA adapter to {hf_repo}")

# Convert to GGUF
bf16_gguf = f"/content/{model_name}-bf16.gguf"
q8_gguf = f"/content/{model_name}-Q8_0.gguf"
q4_gguf = f"/content/{model_name}-Q4_K_M.gguf"

print("3. Converting Nanbeige 4.2-3B to GGUF format...")
cmd_convert = [
    "python3", "/content/llama.cpp/convert_hf_to_gguf.py",
    merged_dir,
    "--outtype", "bf16",
    "--outfile", bf16_gguf
]
subprocess.run(cmd_convert, check=True)
print(f"GGUF bf16 conversion complete: {bf16_gguf}")

print("4. Quantizing to Q8_0 and Q4_K_M...")
subprocess.run(["/content/llama.cpp/build/bin/llama-quantize", bf16_gguf, q8_gguf, "Q8_0"], check=True)
subprocess.run(["/content/llama.cpp/build/bin/llama-quantize", bf16_gguf, q4_gguf, "Q4_K_M"], check=True)
print("Quantization complete!")

# Remove intermediate bf16
if os.path.exists(bf16_gguf):
    os.remove(bf16_gguf)
    print(f"Cleaned up intermediate {bf16_gguf}")

print("5. Uploading quantized GGUF binaries to Hugging Face...")
api.upload_file(path_or_fileobj=q4_gguf, path_in_repo=f"{model_name}-Q4_K_M.gguf", repo_id=hf_repo, repo_type="model")
api.upload_file(path_or_fileobj=q8_gguf, path_in_repo=f"{model_name}-Q8_0.gguf", repo_id=hf_repo, repo_type="model")

# Upload tokenizer and configs to HF root
for fname in ["tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json"]:
    src = os.path.join(merged_dir, fname)
    if os.path.exists(src):
        api.upload_file(path_or_fileobj=src, path_in_repo=fname, repo_id=hf_repo, repo_type="model")

print(f"All Nanbeige 4.2-3B artifacts published to https://huggingface.co/{hf_repo}")
