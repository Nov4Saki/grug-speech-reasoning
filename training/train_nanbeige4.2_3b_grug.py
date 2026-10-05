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
model_id = "Nanbeige/Nanbeige4.2-3B"
dataset_dir = "/content/drive/MyDrive/grug_multifield_dataset"
output_dir = "/content/drive/MyDrive/nanbeige4.2-3b-grug"

print("1. Loading Nanbeige Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("2. Loading & Adapting Multi-Field Datasets...")
train_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "train.jsonl"))["train"]
eval_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "val.jsonl"))["train"]

def adapt_for_nanbeige(example):
    new_messages = []
    for m in example["messages"]:
        if m["role"] == "system":
            content = m["content"].replace("You are Qwen 3.5", "You are Nanbeige 4.2")
            new_messages.append({"role": "system", "content": content})
        else:
            new_messages.append(m)
    return {"messages": new_messages}

train_dataset = train_dataset.map(adapt_for_nanbeige)
eval_dataset = eval_dataset.map(adapt_for_nanbeige)
print(f"Loaded and mapped {len(train_dataset)} train samples, {len(eval_dataset)} eval samples.")

print("3. Loading 8-Bit Quantized Nanbeige 4.2-3B...")
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
    eval_strategy="epoch",
    save_strategy="epoch",
    bf16=True,
    max_length=1024,
    completion_only_loss=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer
)

print("6. Training Nanbeige4.2-3B-GrugSpeech-Native...")
train_result = trainer.train()
print("Training completed! Metrics:", train_result.metrics)

print("7. Evaluating on Validation Set...")
eval_metrics = trainer.evaluate()
print("Evaluation metrics:", eval_metrics)

print("8. Saving Final LoRA Adapter...")
final_adapter_dir = os.path.join(output_dir, "final_adapter")
os.makedirs(final_adapter_dir, exist_ok=True)
trainer.save_model(final_adapter_dir)
tokenizer.save_pretrained(final_adapter_dir)
print(f"Nanbeige 4.2-3B Grug Adapter successfully saved to {final_adapter_dir}")
