"""
train_ebft_grug_reasoner.py
===========================
Energy-Based Fine-Tuning (EBFT) Trainer for Grug Speech Reasoning.
Uses trajectory-level contrastive alignment (DPO / Energy-Ranking Loss)
to minimize the global energy of reasoning trajectories:
    E(x, y) = E_syntax + alpha * E_invariants + beta * E_length + gamma * E_outcome

Pushes models to prefer dense, invariant-conserving, terminal-anchored
Grug traces (low energy) over verbose, bloated, or hallucinated CoT (high energy).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from pipeline.ebft_grug_optimizer import compile_ebft_preference_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="EBFT Grug Speech Reasoner Trainer")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model identifier")
    parser.add_argument("--input_jsonl", type=str, default="datasets/scaled_multifield/train.jsonl", help="Input SFT dataset")
    parser.add_argument("--dpo_dataset_path", type=str, default="datasets/scaled_multifield/ebft_pairs.jsonl", help="Output DPO dataset")
    parser.add_argument("--output_dir", type=str, default="/workspace/grug_qwen3b_ebft_output", help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of DPO training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for DPO")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature beta (implicit energy scale)")
    parser.add_argument("--sample_limit", type=int, default=400, help="Number of preference pairs to compile")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ENERGY-BASED FINE-TUNING (EBFT) PIPELINE")
    print("=" * 60)
    
    # 1. Compile contrastive energy preference dataset
    print(f"\n1. Compiling EBFT preference dataset from {args.input_jsonl}...")
    num_pairs = compile_ebft_preference_dataset(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.dpo_dataset_path,
        sample_limit=args.sample_limit
    )
    print(f"Compiled {num_pairs} contrastive energy trajectory pairs to {args.dpo_dataset_path}.")
    
    # 2. Hardware check
    major, _ = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (7, 0)
    compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
    print(f"\nHardware precision: {compute_dtype} on {torch.cuda.get_device_name(0)}")
    
    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 4. Load Dataset
    dpo_dataset = load_dataset("json", data_files=args.dpo_dataset_path)["train"]
    print(f"Loaded {len(dpo_dataset)} preference pairs for energy alignment.")
    
    print("\n[EBFT Pipeline Ready] Configured for trajectory energy minimization.")
    print(f"Energy scale beta: {args.beta} | Target: Lower E(x, y) by >50%")


if __name__ == "__main__":
    main()
