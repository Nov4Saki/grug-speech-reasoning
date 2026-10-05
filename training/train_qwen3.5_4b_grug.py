import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

model_id = "Qwen/Qwen3.5-4B"
dataset_dir = "/content/drive/MyDrive/grug_multifield_dataset"
output_dir = "/content/drive/MyDrive/qwen3.5-4b-grug-native"

print("1. Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("2. Loading Multi-Field Datasets...")
train_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "train.jsonl"))["train"]
eval_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "val.jsonl"))["train"]
print(f"Loaded {len(train_dataset)} train samples, {len(eval_dataset)} eval samples.")

print("3. Loading 8-Bit Quantized Qwen3.5-4B...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
base_model = prepare_model_for_kbit_training(base_model)

print("4. Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()

print("5. Setting up SFTConfig & SFTTrainer...")
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=15,
    save_total_limit=2,
    bf16=True,
    max_length=512,
    completion_only_loss=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

print("6. Starting Fine-Tuning of Qwen 3.5 4B...")
train_result = trainer.train()
print("Training complete! Metrics:", train_result.metrics)

print("7. Saving final adapter and tokenizer...")
final_save_dir = os.path.join(output_dir, "final_adapter")
trainer.save_model(final_save_dir)
tokenizer.save_pretrained(final_save_dir)
print("Saved Qwen 3.5 4B Grug Model to:", final_save_dir)
