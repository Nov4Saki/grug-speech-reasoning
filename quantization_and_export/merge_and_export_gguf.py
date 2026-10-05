import os
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

def merge_and_save(base_id, adapter_dir, merged_dir):
    print(f"\n>>> Merging {adapter_dir} into {base_id}...")
    os.makedirs(merged_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    print("Loading base model in bfloat16 on GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    print("Merging weights...")
    merged = model.merge_and_unload()
    
    print(f"Saving merged model to {merged_dir}...")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print("Merged save complete!")
    
    del model
    del base_model
    del merged
    torch.cuda.empty_cache()

def convert_and_quantize(merged_dir, model_name, hf_repo):
    bf16_gguf = f"/content/{model_name}-bf16.gguf"
    q8_gguf = f"/content/{model_name}-Q8_0.gguf"
    q4_gguf = f"/content/{model_name}-Q4_K_M.gguf"
    
    # 1. Convert HF to GGUF BF16
    print(f"\n>>> Converting {merged_dir} to GGUF (BF16)...")
    cmd_convert = [
        "python3", "/content/llama.cpp/convert_hf_to_gguf.py",
        merged_dir,
        "--outtype", "bf16",
        "--outfile", bf16_gguf,
        "--no-mtp"
    ]
    subprocess.run(cmd_convert, check=True)
    print(f"BF16 GGUF created: {bf16_gguf} ({os.path.getsize(bf16_gguf)/(1024**3):.2f} GB)")
    
    # 2. Quantize to Q8_0
    print(f"\n>>> Quantizing to Q8_0: {q8_gguf}...")
    cmd_q8 = [
        "/content/llama.cpp/build/bin/llama-quantize",
        bf16_gguf, q8_gguf, "Q8_0"
    ]
    subprocess.run(cmd_q8, check=True)
    print(f"Q8_0 GGUF created: {q8_gguf} ({os.path.getsize(q8_gguf)/(1024**3):.2f} GB)")
    
    # 3. Quantize to Q4_K_M
    print(f"\n>>> Quantizing to Q4_K_M: {q4_gguf}...")
    cmd_q4 = [
        "/content/llama.cpp/build/bin/llama-quantize",
        bf16_gguf, q4_gguf, "Q4_K_M"
    ]
    subprocess.run(cmd_q4, check=True)
    print(f"Q4_K_M GGUF created: {q4_gguf} ({os.path.getsize(q4_gguf)/(1024**3):.2f} GB)")
    
    # Clean up intermediate BF16 GGUF to save disk
    if os.path.exists(bf16_gguf):
        os.remove(bf16_gguf)
        print("Removed intermediate BF16 GGUF.")
        
    # 4. Upload GGUFs to Hugging Face
    print(f"\n>>> Uploading GGUFs to Hugging Face: {hf_repo}...")
    api.upload_file(
        path_or_fileobj=q8_gguf,
        path_in_repo=f"{model_name}-Q8_0.gguf",
        repo_id=hf_repo,
        repo_type="model",
        commit_message=f"Upload {model_name} Q8_0 GGUF"
    )
    api.upload_file(
        path_or_fileobj=q4_gguf,
        path_in_repo=f"{model_name}-Q4_K_M.gguf",
        repo_id=hf_repo,
        repo_type="model",
        commit_message=f"Upload {model_name} Q4_K_M GGUF"
    )
    print(f"Upload complete for {hf_repo}!")

print("=== STARTING FULL GGUF PIPELINE ===")

# --- Step A: Qwen 3.5 2B ---
merge_and_save(
    base_id="Qwen/Qwen3.5-2B",
    adapter_dir="/content/drive/MyDrive/qwen3.5-2b-grug/final_adapter",
    merged_dir="/content/merged_qwen3.5_2b"
)
convert_and_quantize(
    merged_dir="/content/merged_qwen3.5_2b",
    model_name="Qwen3.5-2B-GrugSpeech",
    hf_repo="Novasaki/Qwen3.5-2B-GrugSpeech-Q8"
)

# --- Step B: Qwen 3.5 4B ---
merge_and_save(
    base_id="Qwen/Qwen3.5-4B",
    adapter_dir="/content/drive/MyDrive/qwen3.5-4b-grug-native/final_adapter",
    merged_dir="/content/merged_qwen3.5_4b"
)
convert_and_quantize(
    merged_dir="/content/merged_qwen3.5_4b",
    model_name="Qwen3.5-4B-GrugSpeech",
    hf_repo="Novasaki/Qwen3.5-4B-GrugSpeech-Native"
)

print("\n=======================================================")
print("ALL MODELS MERGED, QUANTIZED (Q8_0 & Q4_K_M), AND UPLOADED!")
print("=======================================================")
