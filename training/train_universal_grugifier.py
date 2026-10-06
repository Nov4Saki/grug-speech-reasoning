"""
Training Script for the Universal Grugifier Engine on NVIDIA A100 GPU
Fine-tunes Qwen2.5-3B via native BF16 QLoRA to serve as a high-throughput,
multi-family, multi-domain cognitive compiler for EBFT reasoning distillation.
"""

import os
import sys
import json
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
    """
    Formats the sample into ChatML tokens with prompt masking:
    Loss is calculated ONLY on the assistant's Grug thinking and answer.
    """
    system_text = example.get("system", "You are the Universal Grugifier Engine.")
    user_text = example["user"]
    assistant_text = example["assistant"]

    prompt_formatted = (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    full_text = prompt_formatted + assistant_text + "<|im_end|>"

    prompt_tokens = tokenizer(prompt_formatted, add_special_tokens=False)["input_ids"]
    full_tokens = tokenizer(full_text, add_special_tokens=False, max_length=2048, truncation=True)["input_ids"]

    labels = list(full_tokens)
    # Mask prompt tokens so loss is only calculated on Grug output
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
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    train_file = "/content/grug-speech-reasoning/datasets/universal_grugifier/train.jsonl"
    val_file = "/content/grug-speech-reasoning/datasets/universal_grugifier/val.jsonl"
    output_dir = "/workspace/universal_grugifier_output"
    
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("UNIVERSAL GRUGIFIER TRAINING PIPELINE (NVIDIA A100)")
    print(f"Base Model: {model_name}")
    print(f"Train Dataset: {train_file}")
    print(f"Validation Dataset: {val_file}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Configure 4-bit Quantization (NF4) with BF16 compute
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 3. Load Base Model
    print("Loading base model in BF16 NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # 4. Configure LoRA
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

    # 5. Load & Process Dataset
    print("Loading JSONL datasets...")
    raw_dataset = load_dataset(
        "json",
        data_files={"train": train_file, "val": val_file}
    )

    print("Tokenizing datasets with prompt masking...")
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

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=150,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=15,
        bf16=True,
        logging_steps=15,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=150,
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

    # 7. Execute Training
    print("Starting training on NVIDIA A100...")
    train_result = trainer.train()
    
    # 8. Save Final Model Adapter & Tokenizer
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    print(f"Saving fine-tuned adapter to {final_adapter_dir}...")
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # 9. Evaluate Final Metrics
    eval_metrics = trainer.evaluate()
    print("Final Validation Metrics:", json.dumps(eval_metrics, indent=2))
    
    with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print("Universal Grugifier Training Complete!")

if __name__ == "__main__":
    main()
