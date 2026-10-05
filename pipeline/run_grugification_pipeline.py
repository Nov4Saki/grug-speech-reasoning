import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm

print("==================================================")
print("STARTING BATCH GRUGIFICATION DISTILLATION PIPELINE")
print("==================================================")

adapter_dir = "/content/drive/MyDrive/qwen3.5-2b-grug/final_adapter"
base_model_id = "Qwen/Qwen3.5-2B"
output_dir = "/content/drive/MyDrive/grug_distilled_reasoning"
os.makedirs(output_dir, exist_ok=True)

print("1. Loading Tokenizer and Grugifier Engine...")
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
grug_engine = PeftModel.from_pretrained(base_model, adapter_dir)
grug_engine.eval()

SYSTEM_PROMPT = (
    "You are a Grug Speech Optimizer. Convert the verbose speech or reasoning into grug speech—an ultra-terse, "
    "token-compressed internal reasoning style (GPT-5.6 grug brain). "
    "Rules: Strip all conversational fluff, keep 100% of facts, numbers, logic steps, constraints, and conclusions."
)

def grugify_trace(verbose_text, max_new=250):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Optimize this verbose reasoning into grug speech:\n\n{verbose_text}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = grug_engine.generate(
            **inputs, 
            max_new_tokens=max_new, 
            do_sample=False, 
            repetition_penalty=1.1
        )
    gen = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def verify_invariants(orig_text, grug_text, critical_terms):
    in_tok = len(tokenizer.encode(orig_text))
    out_tok = len(tokenizer.encode(grug_text))
    comp_ratio = in_tok / max(1, out_tok)
    
    # Check if critical terms (function names, key numbers, targets) are kept
    term_matches = [term.lower() in grug_text.lower() for term in critical_terms if term]
    accuracy = sum(term_matches) / max(1, len(term_matches))
    passed = (comp_ratio >= 1.2) and (accuracy >= 0.5)
    return passed, comp_ratio, in_tok, out_tok

distilled_samples = []

# --- DOMAIN A: Tool Use & Agent Loops (from Hermes Reasoning Tool Use) ---
print("\n2. Processing Tool-Use Agent Reasoning Traces...")
ds_tool = load_dataset('interstellarninja/hermes_reasoning_tool_use', split='train[:35]')

for item in tqdm(ds_tool, desc="Grugifying Tool Traces"):
    convs = item['conversations']
    user_msg = ""
    gpt_turn = ""
    for c in convs:
        role = c.get('from') or c.get('role')
        if role in ['human', 'user']:
            user_msg = c.get('value') or c.get('content')
        elif role in ['gpt', 'assistant']:
            gpt_turn = c.get('value') or c.get('content')
            break
            
    if not user_msg or not gpt_turn:
        continue
        
    think_match = re.search(r'<think>(.*?)</think>', gpt_turn, re.DOTALL)
    if not think_match:
        continue
    verbose_think = think_match.group(1).strip()
    after_think = gpt_turn[think_match.end():].strip()
    
    # Find critical terms (e.g. tool names from tool call)
    tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', after_think, re.DOTALL)
    critical_terms = []
    if tool_calls:
        try:
            call_obj = json.loads(tool_calls[0])
            critical_terms.append(call_obj.get("name", ""))
        except:
            pass
            
    grug_think = grugify_trace(verbose_think[:600])
    passed, comp_ratio, in_tok, out_tok = verify_invariants(verbose_think[:600], grug_think, critical_terms)
    
    if passed:
        distilled_sample = {
            "source_benchmark": "hermes_reasoning_tool_use",
            "task_type": "agentic_tool_use",
            "prompt": user_msg,
            "original_reasoning": verbose_think,
            "grug_reasoning": grug_think,
            "after_think_action": after_think,
            "metrics": {
                "original_tokens": in_tok,
                "grug_tokens": out_tok,
                "compression_ratio": round(comp_ratio, 2)
            },
            "training_sample": {
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": f"<think>\n{grug_think}\n</think>\n{after_think}"}
                ]
            }
        }
        distilled_samples.append(distilled_sample)

# --- DOMAIN B: Multi-Step Math & Algorithmic Deductions (from Bespoke-Stratos) ---
print("\n3. Processing Math & Algorithmic Deep Reasoning Traces...")
ds_stratos = load_dataset('bespokelabs/Bespoke-Stratos-17k', split='train[:35]')

for item in tqdm(ds_stratos, desc="Grugifying Math Traces"):
    convs = item['conversations']
    user_msg = ""
    asst_msg = ""
    for c in convs:
        role = c.get('from') or c.get('role')
        if role in ['human', 'user']:
            user_msg = c.get('value') or c.get('content')
        elif role in ['gpt', 'assistant']:
            asst_msg = c.get('value') or c.get('content')
            break
            
    if not user_msg or not asst_msg:
        continue
        
    think_match = re.search(r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>', asst_msg, re.DOTALL)
    if not think_match:
        # Fallback if no end tag
        verbose_think = asst_msg[:800]
        final_solution = asst_msg[800:]
    else:
        verbose_think = think_match.group(1).strip()
        final_solution = asst_msg[think_match.end():].strip()
        
    # Extract boxed answer or numbers
    boxed = re.findall(r'\\boxed\{([^}]+)\}', final_solution)
    critical_terms = boxed if boxed else []
    
    grug_think = grugify_trace(verbose_think[:600])
    passed, comp_ratio, in_tok, out_tok = verify_invariants(verbose_think[:600], grug_think, critical_terms)
    
    if passed:
        distilled_sample = {
            "source_benchmark": "bespoke_stratos_reasoning",
            "task_type": "deep_reasoning_math",
            "prompt": user_msg,
            "original_reasoning": verbose_think,
            "grug_reasoning": grug_think,
            "after_think_action": final_solution,
            "metrics": {
                "original_tokens": in_tok,
                "grug_tokens": out_tok,
                "compression_ratio": round(comp_ratio, 2)
            },
            "training_sample": {
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": f"<think>\n{grug_think}\n</think>\n{final_solution}"}
                ]
            }
        }
        distilled_samples.append(distilled_sample)

print(f"\nSuccessfully verified and distilled {len(distilled_samples)} traces!")

# Save to drive
out_path = os.path.join(output_dir, "grug_distilled_cot_dataset.jsonl")
with open(out_path, "w") as f:
    for s in distilled_samples:
        f.write(json.dumps(s) + "\n")

# Compute overall metrics
avg_comp = sum(s['metrics']['compression_ratio'] for s in distilled_samples) / max(1, len(distilled_samples))
avg_in = sum(s['metrics']['original_tokens'] for s in distilled_samples) / max(1, len(distilled_samples))
avg_out = sum(s['metrics']['grug_tokens'] for s in distilled_samples) / max(1, len(distilled_samples))

summary = {
    "dataset_name": "Grug-Distilled-Reasoning-Traces",
    "total_verified_samples": len(distilled_samples),
    "average_original_tokens": round(avg_in, 1),
    "average_grug_tokens": round(avg_out, 1),
    "average_compression_ratio": f"{avg_comp:.2f}x",
    "token_savings_percent": f"{(1 - avg_out/avg_in)*100:.1f}%",
    "domains": ["agentic_tool_use", "deep_reasoning_math"]
}

summary_path = os.path.join(output_dir, "distillation_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\nSummary Metrics:")
print(json.dumps(summary, indent=2))

# 4. Upload Distilled Corpus to Hugging Face
print("\n4. Uploading Distilled Reasoning Corpus to Hugging Face...")
token = os.environ.get("HF_TOKEN")
distill_repo = "Novasaki/grug-distilled-reasoning-traces"
api = HfApi(token=token)

create_repo(repo_id=distill_repo, repo_type="dataset", token=token, exist_ok=True)
api.upload_folder(
    folder_path=output_dir,
    repo_id=distill_repo,
    repo_type="dataset",
    commit_message="Release Grug-distilled reasoning and tool-use traces"
)
print(f"Dataset live at: https://huggingface.co/datasets/{distill_repo}")
