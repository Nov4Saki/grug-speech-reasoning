import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

adapter_dir = "/content/drive/MyDrive/qwen3.5-2b-grug/final_adapter"
base_model_id = "Qwen/Qwen3.5-2B"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

SYSTEM_PROMPT = (
    "You are a Grug Speech Optimizer. Convert the verbose speech or reasoning into grug speech—an ultra-terse, "
    "token-compressed internal reasoning style (GPT-5.6 grug brain). "
    "Rules: Strip all conversational fluff, keep 100% of facts, numbers, logic steps, constraints, and conclusions."
)

def grugify(text, max_new=200):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Optimize this verbose reasoning into grug speech:\n\n{text}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.1)
    gen = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# 1. Test on Hermes Tool Use <think> trace
ds_tool = load_dataset('interstellarninja/hermes_reasoning_tool_use', split='train[:2]')
sample_tool = ds_tool[1]
gpt_turn = sample_tool['conversations'][2]['value'] # has <think>...</think>
think_match = re.search(r'<think>(.*?)</think>', gpt_turn, re.DOTALL)
verbose_think = think_match.group(1).strip() if think_match else gpt_turn[:300]

print("=== HERMES TOOL USE SAMPLE ===")
print("Original Verbose Think (first 300 chars):")
print(verbose_think[:300] + "...")
in_tok = len(tokenizer.encode(verbose_think))

grug_out = grugify(verbose_think[:400])
out_tok = len(tokenizer.encode(grug_out))

print(f"\nOriginal Tokens: {in_tok} -> Grug Tokens: {out_tok} (Compression: {in_tok/out_tok:.2f}x)")
print("Grugified Thought:")
print(grug_out)
