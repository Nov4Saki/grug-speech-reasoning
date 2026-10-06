import os
import sys
import gc
import json
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Gemma4ForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import HfApi

TOKEN = "HF_TOKEN_REDACTED"
REPO_ID = "Novasaki/Gemma4-E4B-UniversalGrugifier"
BASE_MODEL = "google/gemma-4-E4B-it"
OUTPUT_DIR = "/workspace/gemma4_e4b_grugifier_output"

def format_sample(example, tokenizer):
    system_text = example.get("system", "You are the Universal Grugifier Engine. Transform verbose reasoning traces from any source model into dense, invariant-anchored Grug Speech (<think>...Done.</think>) optimized for Energy-Based Fine-Tuning (EBFT). Preserve all domain, persona, and causal invariants.")
    user_text = example["user"]
    assistant_text = example["assistant"]

    prompt_formatted = (
        f"<bos><|turn>system\n{system_text}<turn|>\n"
        f"<|turn>user\n{user_text}<turn|>\n"
        f"<|turn>model\n"
    )
    full_text = prompt_formatted + assistant_text + "<turn|>\n"

    prompt_tokens = tokenizer(prompt_formatted, add_special_tokens=False)["input_ids"]
    full_tokens = tokenizer(full_text, add_special_tokens=False, max_length=2048, truncation=True)["input_ids"]

    labels = list(full_tokens)
    prompt_len = min(len(prompt_tokens), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    attention_mask = [1] * len(full_tokens)
    return {
        "input_ids": full_tokens,
        "labels": labels,
        "attention_mask": attention_mask
    }

def main():
    print("=" * 80)
    print("TRAINING GEMMA 4 E4B UNIVERSAL GRUGIFIER (NVIDIA A100 SXM4)")
    print(f"Base Architecture: {BASE_MODEL}")
    print(f"Target Hugging Face Hub: {REPO_ID}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Tokenizer
    print("\n[1/5] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 4-bit Quantized Base Model
    print("\n[2/5] Loading Gemma 4 E4B in NF4 BF16...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Gemma4ForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=TOKEN,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    print(f"Model loaded! VRAM: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # 3. LoRA Configuration
    print("\n[3/5] Applying LoRA Adapter (r=16, alpha=32)...")
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Prepare Dataset
    print("\n[4/5] Preparing Multi-Family Dataset...")
    train_file = "/content/grug-speech-reasoning/datasets/universal_grugifier/train.jsonl"
    val_file = "/content/grug-speech-reasoning/datasets/universal_grugifier/val.jsonl"

    raw_train = load_dataset("json", data_files=train_file, split="train")
    raw_val = load_dataset("json", data_files=val_file, split="train")

    tokenized_train = raw_train.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=raw_train.column_names,
        desc="Tokenizing Train Dataset"
    )
    tokenized_val = raw_val.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=raw_val.column_names,
        desc="Tokenizing Validation Dataset"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        max_steps=120,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=12,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=40,
        save_strategy="steps",
        save_steps=120,
        save_total_limit=1,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator
    )

    # 5. Execute Training & Evaluation
    print("\n[5/5] Executing Training...")
    t0 = time.time()
    train_result = trainer.train()
    train_duration = time.time() - t0
    print(f"Training Complete in {train_duration:.1f}s!")

    eval_result = trainer.evaluate()
    print("Validation Results:")
    print(json.dumps(eval_result, indent=2))

    # Save Adapter
    final_adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w") as f:
        json.dump(eval_result, f, indent=2)

    # 6. Eject from GPU Memory
    print("\nEjecting Gemma 4 E4B from GPU Memory...")
    del trainer
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"VRAM after ejection: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # 7. Upload to Hugging Face
    print(f"\nUploading to Hugging Face ({REPO_ID})...")
    api = HfApi(token=TOKEN)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    readme = f"""---
base_model: {BASE_MODEL}
library_name: peft
license: gemma
tags:
- reasoning
- grug-speech
- universal-grugifier
- gemma4
- cognitive-compiler
- ebft
pipeline_tag: text-generation
---

# Gemma4-E4B-UniversalGrugifier

The **Universal Grugifier (Gemma 4 E4B Edition)** is a state-of-the-art cognitive compiler fine-tuned on an NVIDIA A100-SXM4-40GB to transform verbose reasoning traces from any source model into dense, invariant-anchored **Grug Speech** (`<think>...Done.</think>`) optimized for **Energy-Based Fine-Tuning (EBFT)**.

## ⚡ Highlights
- **Base Architecture:** `{BASE_MODEL}` (Google Gemma 4, Sliding-Window + Full Attention, Native Thinking Channels).
- **Validation Loss:** `{eval_result.get('eval_loss', 0.0):.6f}` across 1,100 multi-family validation samples.
- **Cross-Family Distillation:** Covers 11 model profiles (Google Gemma, Alibaba Qwen 3.5/3.8, Liquid LFM, Nanbeige, MiniCPM5) across 10 operational domains.
- **Published by:** Novasaki Research.
"""
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add Gemma 4 E4B Universal Grugifier model card"
    )
    api.upload_folder(
        folder_path=final_adapter_dir,
        path_in_repo=".",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload Gemma 4 E4B Universal Grugifier LoRA adapter weights"
    )
    print(f"Successfully published to https://huggingface.co/{REPO_ID}!")

if __name__ == "__main__":
    main()
