"""
train_scaled_grug_reasoner.py
=============================
Hardware-Aware Native BF16/FP16 QLoRA Trainer for Grug Speech Reasoning.
Automatically detects Ampere/Ada/Hopper (A100/L4, sm_80+) for native bfloat16,
or Turing (T4, sm_75) for float16. Applies completion-only loss masking and
fine-tunes on the scaled 2,128 multi-field dataset.
"""

import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Hardware-Aware Grug Speech Reasoner Trainer")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model identifier")
    parser.add_argument("--dataset_dir", type=str, default="datasets/scaled_multifield", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="/workspace/grug_qwen3b_output", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    return parser.parse_args()


def detect_hardware():
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available. Falling back to CPU (slow!).")
        return False, torch.float32
        
    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    is_bf16 = major >= 8  # Ampere (A100/A10G: 8.0), Ada (L4/RTX4090: 8.9), Hopper (H100: 9.0)
    compute_dtype = torch.bfloat16 if is_bf16 else torch.float16
    
    print("=" * 60)
    print("HARDWARE ACCELERATION DETECTION")
    print("=" * 60)
    print(f"GPU Model:               {device_name}")
    print(f"Compute Capability:      sm_{major}{minor}")
    print(f"Native BF16 Support:     {'YES (Ampere/Ada/Hopper)' if is_bf16 else 'NO (Turing/Legacy fallback)'}")
    print(f"Active Compute Dtype:    {compute_dtype}")
    print(f"Allocated VRAM:          {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 60)
    return is_bf16, compute_dtype


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    is_bf16, compute_dtype = detect_hardware()

    print(f"\n1. Loading Tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"\n2. Loading Datasets from {args.dataset_dir}...")
    train_file = os.path.join(args.dataset_dir, "train.jsonl")
    val_file = os.path.join(args.dataset_dir, "val.jsonl")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training dataset not found: {train_file}")
        
    raw_train = load_dataset("json", data_files=train_file)["train"]
    raw_val = load_dataset("json", data_files=val_file)["train"]
    print(f"Loaded {len(raw_train)} training samples and {len(raw_val)} validation samples.")

    print(f"\n3. Loading 4-Bit NF4 Base Model ({args.model_id})...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(base_model)

    print("\n4. Configuring LoRA Adapter...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    print("\n5. Setting up SFTConfig & Hardware-Aware Trainer...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=is_bf16,
        fp16=(not is_bf16),
        max_length=args.max_length,
        completion_only_loss=True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=raw_train,
        eval_dataset=raw_val,
        processing_class=tokenizer,
    )

    print("\n6. Launching Native BF16 QLoRA Fine-Tuning...")
    train_result = trainer.train()
    print("\nTraining completed successfully! Metrics:", train_result.metrics)

    print("\n7. Saving Final Grug Reasoner Adapter...")
    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"Saved fine-tuned Grug adapter to: {final_adapter_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
