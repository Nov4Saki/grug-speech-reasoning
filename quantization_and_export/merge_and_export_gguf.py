"""
merge_and_export_gguf.py
========================
Merges fine-tuned LoRA adapter into base weights, exports to BF16 GGUF via llama.cpp,
and compiles quantized deployment binaries (Q8_0 and Q4_K_M).
"""

import os
import sys
import argparse
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA and Export GGUF")
    parser.add_argument("--base_id", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model identifier")
    parser.add_argument("--adapter_dir", type=str, default="/workspace/grug_qwen3b_output/final_adapter", help="LoRA adapter directory")
    parser.add_argument("--merged_dir", type=str, default="/workspace/merged_qwen2.5_3b", help="Directory to save merged weights")
    parser.add_argument("--output_name", type=str, default="Qwen2.5-3B-GrugSpeech", help="Prefix for exported GGUF files")
    parser.add_argument("--export_dir", type=str, default="/content", help="Directory to store GGUF binaries")
    parser.add_argument("--hf_repo", type=str, default=None, help="Optional Hugging Face repository to upload")
    return parser.parse_args()


def merge_and_save(base_id: str, adapter_dir: str, merged_dir: str):
    print("=" * 60)
    print(f">>> Merging LoRA Adapter: {adapter_dir} into {base_id}")
    print("=" * 60)
    os.makedirs(merged_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    print("Loading base model in bfloat16 on GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    print("Merging weights into base model...")
    merged = model.merge_and_unload()
    
    print(f"Saving merged weights to {merged_dir}...")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print("[SUCCESS] Merged save complete!")
    
    del model
    del base_model
    del merged
    torch.cuda.empty_cache()


def convert_and_quantize(merged_dir: str, model_name: str, export_dir: str, hf_repo: str = None):
    bf16_gguf = os.path.join(export_dir, f"{model_name}-bf16.gguf")
    q8_gguf = os.path.join(export_dir, f"{model_name}-Q8_0.gguf")
    q4_gguf = os.path.join(export_dir, f"{model_name}-Q4_K_M.gguf")
    
    llama_cpp_dir = "/content/llama.cpp"
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
    
    # 1. Convert HF to GGUF BF16
    print("=" * 60)
    print(f">>> Converting {merged_dir} to BF16 GGUF...")
    print("=" * 60)
    cmd_convert = [
        sys.executable, convert_script,
        merged_dir,
        "--outtype", "bf16",
        "--outfile", bf16_gguf
    ]
    subprocess.run(cmd_convert, check=True)
    bf16_size_gb = os.path.getsize(bf16_gguf) / (1024**3)
    print(f"[SUCCESS] BF16 GGUF created: {bf16_gguf} ({bf16_size_gb:.2f} GB)")
    
    # 2. Quantize to Q8_0
    print("\n" + "=" * 60)
    print(f">>> Quantizing to Q8_0: {q8_gguf}...")
    print("=" * 60)
    cmd_q8 = [quantize_bin, bf16_gguf, q8_gguf, "Q8_0"]
    subprocess.run(cmd_q8, check=True)
    q8_size_gb = os.path.getsize(q8_gguf) / (1024**3)
    print(f"[SUCCESS] Q8_0 GGUF created: {q8_gguf} ({q8_size_gb:.2f} GB)")
    
    # 3. Quantize to Q4_K_M
    print("\n" + "=" * 60)
    print(f">>> Quantizing to Q4_K_M: {q4_gguf}...")
    print("=" * 60)
    cmd_q4 = [quantize_bin, bf16_gguf, q4_gguf, "Q4_K_M"]
    subprocess.run(cmd_q4, check=True)
    q4_size_gb = os.path.getsize(q4_gguf) / (1024**3)
    print(f"[SUCCESS] Q4_K_M GGUF created: {q4_gguf} ({q4_size_gb:.2f} GB)")
    
    # Clean up intermediate BF16 GGUF to conserve disk space
    if os.path.exists(bf16_gguf):
        os.remove(bf16_gguf)
        print(f"Removed intermediate unquantized BF16 GGUF ({bf16_size_gb:.2f} GB) to conserve disk space.")

    # 4. Optional Hugging Face upload
    if hf_repo and os.environ.get("HF_TOKEN"):
        api = HfApi(token=os.environ.get("HF_TOKEN"))
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
        print(f"[SUCCESS] Upload complete for {hf_repo}!")

    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Q8_0 Binary:   {q8_gguf} ({q8_size_gb:.2f} GB)")
    print(f"Q4_K_M Binary: {q4_gguf} ({q4_size_gb:.2f} GB)")
    print("=" * 60)


def main():
    args = parse_args()
    merge_and_save(args.base_id, args.adapter_dir, args.merged_dir)
    convert_and_quantize(args.merged_dir, args.output_name, args.export_dir, args.hf_repo)


if __name__ == "__main__":
    main()
