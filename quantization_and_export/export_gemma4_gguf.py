import os
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

base_id = "google/gemma-4-E2B-it"
adapter_dir = "/content/drive/MyDrive/gemma-4-e2b-grug/final_adapter"
merged_dir = "/content/merged_gemma4_e2b"
hf_repo = "Novasaki/Gemma-4-E2B-GrugSpeech-Native"
model_name = "Gemma4-E2B-GrugSpeech"

print(">>> Merging Gemma 4 adapter...")
tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=token)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    token=token,
    dtype=torch.bfloat16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
merged = model.merge_and_unload()
merged.save_pretrained(merged_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_dir)
print("Merged Gemma 4 saved successfully!")

del model
del base_model
del merged
torch.cuda.empty_cache()

# Upload adapter and merged model card to HF
from huggingface_hub import create_repo
create_repo(repo_id=hf_repo, repo_type="model", token=token, exist_ok=True)
api.upload_folder(
    folder_path=adapter_dir,
    repo_id=hf_repo,
    repo_type="model",
    commit_message="Upload Gemma 4 E2B Grug LoRA adapter"
)
print(f"Uploaded LoRA adapter to {hf_repo}")

# Convert to GGUF
bf16_gguf = f"/content/{model_name}-bf16.gguf"
q8_gguf = f"/content/{model_name}-Q8_0.gguf"
q4_gguf = f"/content/{model_name}-Q4_K_M.gguf"

print(">>> Converting Gemma 4 to GGUF...")
try:
    cmd_convert = [
        "python3", "/content/llama.cpp/convert_hf_to_gguf.py",
        merged_dir,
        "--outtype", "bf16",
        "--outfile", bf16_gguf
    ]
    subprocess.run(cmd_convert, check=True)
    print("Gemma 4 BF16 GGUF created!")
    
    # Quantize
    subprocess.run(["/content/llama.cpp/build/bin/llama-quantize", bf16_gguf, q8_gguf, "Q8_0"], check=True)
    subprocess.run(["/content/llama.cpp/build/bin/llama-quantize", bf16_gguf, q4_gguf, "Q4_K_M"], check=True)
    
    if os.path.exists(bf16_gguf):
        os.remove(bf16_gguf)
        
    api.upload_file(path_or_fileobj=q8_gguf, path_in_repo=f"{model_name}-Q8_0.gguf", repo_id=hf_repo, repo_type="model")
    api.upload_file(path_or_fileobj=q4_gguf, path_in_repo=f"{model_name}-Q4_K_M.gguf", repo_id=hf_repo, repo_type="model")
    print(f"GGUF upload complete for {hf_repo}!")
except Exception as e:
    print("Gemma 4 GGUF conversion note:", e)
