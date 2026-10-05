import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

base_model_id = "Qwen/Qwen3.5-2B"
adapter_dir = "/content/drive/MyDrive/qwen3.5-2b-grug/final_adapter"

print("1. Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

print("2. Loading Base Model in 8-bit...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

print("3. Loading Peft LoRA Model...")
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

SYSTEM_PROMPT_OPTIMIZE = (
    "You are a Grug Speech Optimizer. Convert the verbose speech or reasoning into grug speech—an ultra-terse, "
    "token-compressed internal reasoning style (GPT-5.6 grug brain). "
    "Rules:\n"
    "1. Strip all conversational fluff, filler words, polite padding, and redundant grammar.\n"
    "2. Use telegraphic shorthand, short imperative verbs, colons, and arrows (->).\n"
    "3. Keep 100% of facts, numbers, equations, logical steps, constraints, and final conclusions intact.\n"
    "4. Maximum semantic density with minimum tokens."
)

SYSTEM_PROMPT_SOLVE = (
    "You are a reasoning model that reasons in Grug Speech. When given a problem or question, think and answer in "
    "ultra-terse, token-compressed telegraphic shorthand (GPT-5.6 grug style). "
    "Retain every critical step and number, but eliminate all filler words and verbosity. State the final answer clearly."
)

def run_test_sample(title, sys_prompt, user_msg):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    gen_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    in_tok_count = len(tokenizer.encode(user_msg))
    out_tok_count = len(gen_tokens)
    comp_ratio = in_tok_count / out_tok_count if out_tok_count > 0 else 1.0
    
    print(f"==================================================")
    print(f"TEST: {title}")
    print(f"==================================================")
    print(f"USER INPUT ({in_tok_count} tokens):\n{user_msg}\n")
    print(f"GENERATED GRUG SPEECH ({out_tok_count} tokens):\n{generated_text}\n")
    print(f"Compression: {comp_ratio:.2f}x ({(1 - out_tok_count/in_tok_count)*100:.1f}% token reduction)")
    print(f"==================================================\n")
    return {
        "title": title,
        "input": user_msg,
        "output": generated_text,
        "in_tokens": in_tok_count,
        "out_tokens": out_tok_count,
        "compression": comp_ratio
    }

results = []

# TEST 1: Completely unseen dataset - OpenBookQA
obqa = load_dataset("allenai/openbookqa", split="test[10:11]")[0]
q_obqa = obqa["question_stem"]
c_obqa = ", ".join([f"({l}) {t}" for l, t in zip(obqa["choices"]["label"], obqa["choices"]["text"])])
ans_obqa = obqa["answerKey"]
obqa_verbose = (
    f"We are considering the scientific question: '{q_obqa}'. The options provided are {c_obqa}. "
    f"When considering the biological and physiological implications of these choices, we know that organisms require "
    f"energy and specific environmental conditions to survive. The correct biological option is ({ans_obqa}) because "
    f"it aligns directly with evolutionary adaptation and organism survival requirements, whereas the other options describe "
    f"conditions that are detrimental or irrelevant. Therefore, choice ({ans_obqa}) is the correct answer."
)
r1 = run_test_sample(
    "Unseen Dataset 1 (OpenBookQA) - Speech Optimization",
    SYSTEM_PROMPT_OPTIMIZE,
    f"Optimize the following verbose scientific explanation into grug speech:\n\n{obqa_verbose}"
)
results.append(r1)

# TEST 2: Completely unseen dataset - MMLU (Elementary Mathematics)
mmlu_item = load_dataset("cais/mmlu", "elementary_mathematics", split="test[0:1]")[0]
# e.g. 24 = 2p
q_mmlu = mmlu_item["question"]
c_mmlu = ", ".join([f"({i}) {ch}" for i, ch in enumerate(mmlu_item["choices"])])
mmlu_verbose = (
    f"In order to solve the algebraic equation '{q_mmlu}', we must isolate the variable p. "
    f"Currently, p is multiplied by the coefficient 2. To undo multiplication by 2, we must apply the inverse "
    f"operation, which is division by 2, to both sides of the equation. "
    f"Dividing the left side by 2 gives 24 / 2 = 12. "
    f"Dividing the right side by 2 gives 2p / 2 = p. "
    f"Thus, we find that p = 12, which corresponds to option (2). Hence, the correct value is 12."
)
r2 = run_test_sample(
    "Unseen Dataset 2 (MMLU Elementary Math) - Speech Optimization",
    SYSTEM_PROMPT_OPTIMIZE,
    f"Optimize the following verbose mathematical reasoning into grug speech:\n\n{mmlu_verbose}"
)
results.append(r2)

# TEST 3: Unseen GSM8K Test Split (Problem Solving in Grug Speech)
gsm_test = load_dataset("openai/gsm8k", "main", split="test[0:1]")[0]
r3 = run_test_sample(
    "Unseen Dataset 3 (GSM8K Test Split) - Direct Grug Problem Solving",
    SYSTEM_PROMPT_SOLVE,
    f"Solve this math problem in grug speech:\n\n{gsm_test['question']}"
)
results.append(r3)

# TEST 4: Complex Engineering / Systems Speech Optimization
sys_verbose = (
    "In order to optimize our distributed web crawler, we should replace our synchronous blocking HTTP client with an "
    "asynchronous event loop using uvloop and aiohttp. Currently, each crawling thread blocks waiting on socket I/O, "
    "which limits our throughput to roughly 50 requests per second and exhausts our system thread pool. By switching "
    "to an async connection pool with a maximum concurrency limit of 1000 open sockets and HTTP keep-alive enabled, "
    "we can achieve over 2,500 requests per second while reducing CPU utilization from 85% down to 25%."
)
r4 = run_test_sample(
    "Engineering / Architecture Speech Optimization",
    SYSTEM_PROMPT_OPTIMIZE,
    f"Optimize this verbose engineering speech into grug speech:\n\n{sys_verbose}"
)
results.append(r4)

with open("/content/drive/MyDrive/qwen3.5-2b-grug/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete and saved to /content/drive/MyDrive/qwen3.5-2b-grug/evaluation_results.json")
