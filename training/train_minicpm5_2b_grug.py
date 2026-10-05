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

token = os.environ.get("HF_TOKEN")
model_id = "openbmb/MiniCPM5-2B"
dataset_dir = "/content/drive/MyDrive/grug_multifield_dataset"
output_dir = "/content/drive/MyDrive/minicpm5-2b-grug"

print("1. Loading MiniCPM5 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("2. Loading Multi-Field Datasets...")
train_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "train.jsonl"))["train"]
eval_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "val.jsonl"))["train"]
print(f"Loaded {len(train_dataset)} train samples, {len(eval_dataset)} eval samples.")

print("3. Loading 8-Bit Quantized MiniCPM5-2B...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=token,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
base_model = prepare_model_for_kbit_training(base_model)

print("4. Configuring LoRA...")
lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=lora_targets,
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
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    max_seq_length=1024,
    dataset_text_field="text",
    report_to="none"
)

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    tokenizer=tokenizer
)

print("6. Training MiniCPM5-2B-GrugSpeech-Native...")
trainer.train()

print("7. Saving Final LoRA Adapter...")
final_adapter_dir = os.path.join(output_dir, "final_adapter")
trainer.model.save_pretrained(final_adapter_dir)
tokenizer.save_pretrained(final_adapter_dir)
print(f"MiniCPM5 Grug Adapter saved successfully to {final_adapter_dir}")
