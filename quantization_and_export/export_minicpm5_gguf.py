import os
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

base_id = "openbmb/MiniCPM5-2B"
adapter_dir = "/content/drive/MyDrive/minicpm5-2b-grug/final_adapter"
merged_dir = "/content/merged_minicpm5_2b"
hf_repo = "Novasaki/MiniCPM5-2B-GrugSpeech-Native"
model_name = "MiniCPM5-2B-GrugSpeech"

print(">>> Merging MiniCPM5-2B adapter...")
tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=token, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    token=token,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
merged = model.merge_and_unload()
merged.save_pretrained(merged_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_dir)
print("Merged MiniCPM5-2B saved successfully!")

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
    commit_message="Upload MiniCPM5-2B Grug LoRA adapter"
)
print(f"Uploaded LoRA adapter to {hf_repo}")

# Convert to GGUF
bf16_gguf = f"/content/{model_name}-bf16.gguf"
q8_gguf = f"/content/{model_name}-Q8_0.gguf"
q4_gguf = f"/content/{model_name}-Q4_K_M.gguf"

print(">>> Converting MiniCPM5-2B to GGUF...")
cmd_convert = [
    "python3", "/content/llama.cpp/convert_hf_to_gguf.py",
    merged_dir,
    "--outtype", "bf16",
    "--outfile", bf16_gguf
]
subprocess.run(cmd_convert, check=True)

print(">>> Quantizing to Q8_0 and Q4_K_M...")
subprocess.run(["/content/llama.cpp/build/bin/llama-quantize", bf16_gguf, q8_gguf, "Q8_0"], check=True)
subprocess.run(["/content/llama.cpp/build/bin/llama-quantize", bf16_gguf, q4_gguf, "Q4_K_M"], check=True)

print(">>> Uploading GGUF binaries to Hugging Face...")
api.upload_file(path_or_fileobj=q4_gguf, path_in_repo=f"{model_name}-Q4_K_M.gguf", repo_id=hf_repo, repo_type="model")
api.upload_file(path_or_fileobj=q8_gguf, path_in_repo=f"{model_name}-Q8_0.gguf", repo_id=hf_repo, repo_type="model")

# Cleanup bf16
if os.path.exists(bf16_gguf):
    os.remove(bf16_gguf)

print(f"All MiniCPM5-2B artifacts successfully published to https://huggingface.co/{hf_repo}")
