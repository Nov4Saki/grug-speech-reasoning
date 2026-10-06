"""
Generic Modular Grugifier Trainer & GPU Ejector
Trains a specified base model on the multi-family Grugifier dataset, saves artifacts,
evaluates metrics, and explicitly ejects the model from GPU memory.
"""

import os
import sys
import gc
import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def format_sample_for_training(example, tokenizer):
    system_text = example.get("system", "You are the Universal Grugifier Engine.")
    user_text = example["user"]
    assistant_text = example["assistant"]

    # Use model's native chat formatting or ChatML
    prompt_formatted = (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    full_text = prompt_formatted + assistant_text + "<|im_end|>"

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

def train_and_eject(base_model_name: str, output_dir: str, max_steps: int = 150, lr: float = 2e-4):
    train_file = "/content/grug-speech-reasoning/datasets/universal_grugifier/train.jsonl"
    val_file = "/content/grug-speech-reasoning/datasets/universal_grugifier/val.jsonl"
    
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print(f"TRAINING GRUGIFIER: [{base_model_name}] on NVIDIA A100")
    print(f"Output Directory: {output_dir}")
    print(f"Steps: {max_steps} | Learning Rate: {lr}")
    print("=" * 80)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. BitsAndBytes 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 3. Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Dataset
    raw_dataset = load_dataset("json", data_files={"train": train_file, "val": val_file})
    processed_train = raw_dataset["train"].map(
        lambda ex: format_sample_for_training(ex, tokenizer),
        remove_columns=raw_dataset["train"].column_names,
        desc="Tokenizing train"
    )
    processed_val = raw_dataset["val"].map(
        lambda ex: format_sample_for_training(ex, tokenizer),
        remove_columns=raw_dataset["val"].column_names,
        desc="Tokenizing val"
    )

    # 6. Training Args
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=15,
        bf16=True,
        logging_steps=15,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=max_steps,
        save_total_limit=1,
        dataloader_num_workers=2,
        optim="paged_adamw_8bit",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    trainer.train()

    # 7. Save Adapter & Tokenizer
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # 8. Evaluate on validation set
    eval_metrics = trainer.evaluate()
    with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Final Validation Metrics for {base_model_name}:", json.dumps(eval_metrics, indent=2))

    # 9. EXPLICIT GPU EJECTION
    print("Ejecting model and trainer from GPU...")
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("GPU Ejection Complete. Current VRAM Allocated:", torch.cuda.memory_allocated(0), "bytes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_and_eject(args.base_model, args.output_dir, args.max_steps, args.lr)
