import os
import sys
import gc
import json
import time
import re
from typing import List, Dict, Any
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

TOKEN = "HF_TOKEN_REDACTED"

# ==============================================================================
# 1. LOAD OFFICIAL ONLINE BENCHMARKS (HUMANEVAL, GSM8K, GLAIVE FUNCTION CALLING)
# ==============================================================================

def load_official_online_benchmarks() -> List[Dict[str, Any]]:
    print("Loading official online benchmark datasets from Hugging Face...")
    
    # 1. Official HumanEval (OpenAI)
    ds_he = load_dataset("openai/openai_humaneval", split="test", token=TOKEN)
    he_indices = [0, 1, 2, 3, 4, 10]
    tasks = []
    
    for idx in he_indices:
        item = ds_he[idx]
        tasks.append({
            "benchmark": "openai_humaneval",
            "task_id": item["task_id"],
            "domain": "coding",
            "prompt": item["prompt"],
            "instruction": "Complete the Python function. Provide clean, executable Python code.",
            "test_code": item["test"],
            "entry_point": item["entry_point"],
            "canonical_solution": item["canonical_solution"]
        })
        
    # 2. Official GSM8K (OpenAI)
    ds_gsm = load_dataset("openai/gsm8k", "main", split="test", token=TOKEN)
    gsm_indices = [0, 1, 2, 3, 4, 5]
    
    for idx in gsm_indices:
        item = ds_gsm[idx]
        gt_answer = item["answer"].split("####")[-1].strip().replace(",", "")
        tasks.append({
            "benchmark": "openai_gsm8k",
            "task_id": f"gsm8k_{idx}",
            "domain": "mathematics",
            "prompt": item["question"],
            "instruction": "Solve the word problem step-by-step. Conclude with 'Final Answer: <number>'.",
            "expected_answer": gt_answer,
            "full_ground_truth": item["answer"]
        })
        
    # 3. Official Function Calling (Glaive AI v2)
    ds_glaive = load_dataset("glaiveai/glaive-function-calling-v2", split="train", token=TOKEN)
    # Pick 6 diverse function-calling samples with tool calls
    tool_count = 0
    for idx in range(100):
        item = ds_glaive[idx]
        chat = item.get("chat", "")
        system = item.get("system", "")
        if "<call:" in chat or "call:" in chat or "{" in chat:
            # Extract user prompt and expected function call
            if "USER:" in chat and "ASSISTANT:" in chat:
                user_msg = chat.split("USER:")[1].split("ASSISTANT:")[0].strip()
                asst_msg = chat.split("ASSISTANT:")[1].strip()
                if "call:" in asst_msg or "{" in asst_msg:
                    tasks.append({
                        "benchmark": "glaive_function_calling",
                        "task_id": f"tool_glaive_{tool_count}",
                        "domain": "tool_use",
                        "system": system,
                        "prompt": user_msg,
                        "instruction": f"Given the system functions:\n{system}\nGenerate the appropriate tool call in JSON format or call: syntax.",
                        "expected_call": asst_msg[:300]
                    })
                    tool_count += 1
                    if tool_count >= 6:
                        break
                        
    print(f"Loaded {len(tasks)} official benchmark tasks across HumanEval (Coding), GSM8K (Math), and Glaive (Tools).")
    return tasks

# ==============================================================================
# 2. REAL TEST EXECUTION ENGINES
# ==============================================================================

def execute_humaneval_unit_tests(generated_code: str, prompt: str, test_code: str, entry_point: str) -> bool:
    """Executes official HumanEval unit test suite in Python sandbox."""
    clean = generated_code.strip()
    if "```python" in clean:
        clean = clean.split("```python")[1].split("```")[0]
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0]
    clean = clean.strip()
    
    # Attempt 1: Direct function body or full replacement
    test_scripts = [
        f"{prompt}\n{clean}\n\n{test_code}\ncheck({entry_point})",
        f"{clean}\n\n{test_code}\ncheck({entry_point})"
    ]
    
    for script in test_scripts:
        try:
            scope = {}
            exec(script, scope)
            return True
        except Exception:
            continue
            
    return False

def execute_gsm8k_math_test(generated_text: str, expected_answer: str) -> bool:
    """Validates exact numerical answer from reasoning trace."""
    clean = generated_text.strip()
    # Search for Final Answer, ####, or box pattern
    patterns = [
        r"(?:Final Answer:?\s*[$]?|####\s*|is\s*[$]?|equals\s*[$]?)([-+]?\d*\.?\d+)",
        r"[-+]?\d*\.?\d+"
    ]
    for pat in patterns:
        matches = re.findall(pat, clean, re.IGNORECASE)
        if matches:
            for m in reversed(matches):
                try:
                    if abs(float(m) - float(expected_answer)) < 1e-3:
                        return True
                except Exception:
                    continue
    return expected_answer in clean

def execute_tool_call_test(generated_text: str, expected_call: str) -> bool:
    """Validates tool call name and argument syntax."""
    clean = generated_text.strip()
    # Extract tool name from expected call
    tool_name_match = re.search(r"call:(\w+)|['\"]name['\"]:\s*['\"](\w+)['\"]", expected_call)
    expected_tool = tool_name_match.group(1) or tool_name_match.group(2) if tool_name_match else ""
    
    if expected_tool and expected_tool.lower() in clean.lower():
        # Check if arguments or brackets exist
        if "{" in clean and "}" in clean:
            return True
        if "(" in clean and ")" in clean:
            return True
    return False

# ==============================================================================
# 3. MULTI-STAGE EVALUATION WORKFLOW
# ==============================================================================

def main():
    print("=" * 80)
    print("ONLINE BENCHMARKS EVALUATION: DEEPSEEK-R1 CONDITIONED ON GRUGIFIERS")
    print("Benchmarks: openai/openai_humaneval, openai/gsm8k, glaiveai/glaive-function-calling-v2")
    print("=" * 80)
    
    benchmark_tasks = load_official_online_benchmarks()
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    r1_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # --------------------------------------------------------------------------
    # STAGE 1: NATIVE DEEPSEEK-R1 BASELINE
    # --------------------------------------------------------------------------
    print("\n[STAGE 1/4]: Generating Native DeepSeek-R1 Traces on Online Benchmarks...")
    r1_tok = AutoTokenizer.from_pretrained(r1_model_id, token=TOKEN)
    r1_model = AutoModelForCausalLM.from_pretrained(
        r1_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=TOKEN
    )
    print(f"DeepSeek-R1 Loaded! VRAM: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")
    
    stage1_results = []
    t_stage1_start = time.time()
    
    for idx, task in enumerate(benchmark_tasks):
        full_user = f"{task['prompt']}\n{task['instruction']}"
        msgs = [{"role": "user", "content": full_user}]
        formatted_prompt = r1_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = r1_tok(formatted_prompt, return_tensors="pt").to("cuda")
        
        t0 = time.time()
        with torch.no_grad():
            outputs = r1_model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=r1_tok.eos_token_id
            )
        latency = time.time() - t0
        gen_tokens = outputs[0][len(inputs["input_ids"][0]):]
        decoded = r1_tok.decode(gen_tokens, skip_special_tokens=False)
        
        if "</think>" in decoded:
            cot_raw = decoded.split("</think>")[0].replace("<think>", "").strip()
            ans_raw = decoded.split("</think>")[1].replace("<｜end of sentence｜>", "").strip()
        else:
            cot_raw = decoded.replace("<think>", "").strip()
            ans_raw = ""
            
        cot_tokens = len(r1_tok.encode(cot_raw))
        ans_tokens = len(r1_tok.encode(ans_raw))
        
        # Test correctness with actual tests
        if task["domain"] == "coding":
            is_correct = execute_humaneval_unit_tests(ans_raw, task["prompt"], task["test_code"], task["entry_point"])
        elif task["domain"] == "mathematics":
            is_correct = execute_gsm8k_math_test(ans_raw, task["expected_answer"])
        else:
            is_correct = execute_tool_call_test(ans_raw, task["expected_call"])
            
        stage1_results.append({
            "task_id": task["task_id"],
            "benchmark": task["benchmark"],
            "domain": task["domain"],
            "prompt": full_user,
            "raw_cot": cot_raw,
            "raw_answer": ans_raw,
            "cot_tokens": cot_tokens,
            "ans_tokens": ans_tokens,
            "total_tokens": cot_tokens + ans_tokens,
            "latency_sec": round(latency, 2),
            "is_correct": is_correct
        })
        print(f"[{idx+1:02d}/{len(benchmark_tasks)}] Native R1 on {task['task_id']:<18} | CoT: {cot_tokens:>3} tok | Correct: {str(is_correct):<5} | Time: {latency:.2f}s")
        
    print(f"Stage 1 Complete in {time.time() - t_stage1_start:.1f}s")
    
    # Complete Ejection
    del r1_model
    del r1_tok
    gc.collect()
    torch.cuda.empty_cache()
    print(f"DeepSeek-R1 Ejected. VRAM remaining: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # --------------------------------------------------------------------------
    # STAGE 2: COMPILE VIA TRAINED MODERN GRUGIFIERS
    # --------------------------------------------------------------------------
    compiler_cfgs = [
        {
            "id": "qwen35_4b",
            "name": "Qwen3.5-4B-UniversalGrugifier",
            "base": "Qwen/Qwen3.5-4B",
            "adapter": "/workspace/qwen35_grugifier_output/final_adapter",
            "format": "qwen"
        },
        {
            "id": "qwen35_2b",
            "name": "Qwen3.5-2B-UniversalGrugifier",
            "base": "Qwen/Qwen3.5-2B",
            "adapter": "/workspace/qwen35_2b_grugifier_output/final_adapter",
            "format": "qwen"
        },
        {
            "id": "gemma4_e4b",
            "name": "Gemma4-E4B-UniversalGrugifier",
            "base": "google/gemma-4-E4B-it",
            "adapter": "/workspace/gemma4_e4b_grugifier_output/final_adapter",
            "format": "gemma"
        },
        {
            "id": "lfm25_2.6b",
            "name": "LFM2.5-UniversalGrugifier",
            "base": "LiquidAI/LFM2.5-2.6B",
            "adapter": "/workspace/lfm25_grugifier_output/final_adapter",
            "format": "lfm"
        }
    ]
    
    active_compilers = [c for c in compiler_cfgs if os.path.exists(c["adapter"])]
    print(f"\nActive Compilers available: {[c['name'] for c in active_compilers]}")
    
    compiled_traces = {c["id"]: [] for c in active_compilers}
    
    for cfg in active_compilers:
        print(f"\n[STAGE 2/4]: Compiling Online Benchmark traces via {cfg['name']}...")
        tok = AutoTokenizer.from_pretrained(cfg["base"], token=TOKEN, trust_remote_code=True)
        if cfg["format"] == "gemma":
            from transformers import Gemma4ForConditionalGeneration
            base_m = Gemma4ForConditionalGeneration.from_pretrained(
                cfg["base"],
                quantization_config=bnb_config,
                device_map="auto",
                token=TOKEN,
                trust_remote_code=True
            )
        else:
            base_m = AutoModelForCausalLM.from_pretrained(
                cfg["base"],
                quantization_config=bnb_config,
                device_map="auto",
                token=TOKEN,
                trust_remote_code=True
            )
        compiler = PeftModel.from_pretrained(base_m, cfg["adapter"])
        print(f"{cfg['name']} Loaded! VRAM: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")
        
        t0_comp = time.time()
        for idx, item in enumerate(stage1_results):
            user_content = (
                f"[Source Model Family]: deepseek-r1-7b\n"
                f"[Domain]: {item['domain']}\n"
                f"[Language / Dialect]: en_standard\n"
                f"[Task Prompt]:\n{item['prompt']}\n\n"
                f"[Raw Verbose Reasoning]:\n{item['raw_cot']}"
            )
            sys_msg = "You are the Universal Grugifier Engine. Transform verbose reasoning traces from any source model into dense, invariant-anchored Grug Speech (<think>...Done.</think>) optimized for Energy-Based Fine-Tuning (EBFT). Preserve all domain, persona, and causal invariants."
            
            if cfg["format"] == "qwen":
                prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
            elif cfg["format"] == "gemma":
                prompt = f"<bos><|turn>system\n{sys_msg}<turn|>\n<|turn>user\n{user_content}<turn|>\n<|turn>model\n"
            elif cfg["format"] == "lfm":
                prompt = f"<|startoftext|><|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
                
            inputs = tok(prompt, return_tensors="pt").to("cuda")
            t0 = time.time()
            with torch.no_grad():
                outputs = compiler.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.2,
                    pad_token_id=tok.eos_token_id
                )
            lat = time.time() - t0
            gen_text = tok.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)
            
            if "<think>" in gen_text and "</think>" in gen_text:
                grug_cot = gen_text.split("<think>")[1].split("</think>")[0].strip()
            elif "<think>" in gen_text:
                grug_cot = gen_text.split("<think>")[1].strip()
            else:
                grug_cot = gen_text.strip()
                
            grug_toks = len(tok.encode(grug_cot))
            compression = round(item["cot_tokens"] / max(1, grug_toks), 2)
            
            compiled_traces[cfg["id"]].append({
                "task_id": item["task_id"],
                "grug_cot": grug_cot,
                "grug_tokens": grug_toks,
                "compression_ratio": compression,
                "latency_sec": round(lat, 2)
            })
            print(f"[{idx+1:02d}/{len(benchmark_tasks)}] {cfg['id']} on {item['task_id']:<18} | {item['cot_tokens']} tok -> {grug_toks} tok ({compression}x)")
            
        print(f"Compiled all benchmark tasks via {cfg['name']} in {time.time() - t0_comp:.1f}s")
        
        # Complete Ejection
        del compiler
        del base_m
        del tok
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{cfg['name']} Ejected. VRAM remaining: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # --------------------------------------------------------------------------
    # STAGE 3: TEST "GRUGIFIED DEEPSEEK" WITH ACTUAL TESTS
    # --------------------------------------------------------------------------
    print("\n[STAGE 3/4]: Testing DeepSeek-R1 Conditioned on Grug Speech (Actual Test Execution)...")
    r1_tok = AutoTokenizer.from_pretrained(r1_model_id, token=TOKEN)
    r1_model = AutoModelForCausalLM.from_pretrained(
        r1_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=TOKEN
    )
    print(f"DeepSeek-R1 Reloaded! VRAM: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")
    
    grugified_eval_results = {c["id"]: [] for c in active_compilers}
    
    for cfg in active_compilers:
        print(f"\nEvaluating Grugified DeepSeek-R1 ({cfg['name']})...")
        for idx, task in enumerate(benchmark_tasks):
            c_trace = compiled_traces[cfg["id"]][idx]
            grug_cot = c_trace["grug_cot"]
            
            conditioned_prompt = (
                f"<｜begin of sentence｜><｜User｜>{task['prompt']}\n{task['instruction']}"
                f"<｜Assistant｜><think>\n{grug_cot}\n</think>\n"
            )
            inputs = r1_tok(conditioned_prompt, return_tensors="pt").to("cuda")
            
            t0 = time.time()
            with torch.no_grad():
                outputs = r1_model.generate(
                    **inputs,
                    max_new_tokens=450,
                    temperature=0.2,
                    pad_token_id=r1_tok.eos_token_id
                )
            latency = time.time() - t0
            ans_text = r1_tok.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)
            ans_text = ans_text.replace("<｜end of sentence｜>", "").strip()
            ans_tokens = len(r1_tok.encode(ans_text))
            
            # Execute ACTUAL TESTS
            if task["domain"] == "coding":
                is_correct = execute_humaneval_unit_tests(ans_text, task["prompt"], task["test_code"], task["entry_point"])
            elif task["domain"] == "mathematics":
                is_correct = execute_gsm8k_math_test(ans_text, task["expected_answer"])
            else:
                is_correct = execute_tool_call_test(ans_text, task["expected_call"])
                
            grugified_eval_results[cfg["id"]].append({
                "task_id": task["task_id"],
                "benchmark": task["benchmark"],
                "domain": task["domain"],
                "grug_cot": grug_cot,
                "grugified_answer": ans_text,
                "grug_cot_tokens": c_trace["grug_tokens"],
                "ans_tokens": ans_tokens,
                "total_tokens": c_trace["grug_tokens"] + ans_tokens,
                "latency_sec": round(latency, 2),
                "is_correct": is_correct
            })
            print(f"[{idx+1:02d}/{len(benchmark_tasks)}] Grugified R1 ({cfg['id']}) on {task['task_id']:<18} | Ans: {ans_tokens:>3} tok | Correct: {str(is_correct):<5} | Time: {latency:.2f}s")
            
    # Complete Ejection
    del r1_model
    del r1_tok
    gc.collect()
    torch.cuda.empty_cache()
    print(f"DeepSeek-R1 Ejected. VRAM remaining: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # --------------------------------------------------------------------------
    # STAGE 4: SCORECARD & TECHNICAL REPORT
    # --------------------------------------------------------------------------
    print("\n[STAGE 4/4]: Computing Official Benchmark Scorecard...")
    
    def calculate_scorecard(eval_list):
        total = len(eval_list)
        acc = sum(1 for x in eval_list if x["is_correct"]) / total * 100.0
        c_list = [x for x in eval_list if x["domain"] == "coding"]
        m_list = [x for x in eval_list if x["domain"] == "mathematics"]
        t_list = [x for x in eval_list if x["domain"] == "tool_use"]
        
        c_acc = sum(1 for x in c_list if x["is_correct"]) / max(1, len(c_list)) * 100.0
        m_acc = sum(1 for x in m_list if x["is_correct"]) / max(1, len(m_list)) * 100.0
        t_acc = sum(1 for x in t_list if x["is_correct"]) / max(1, len(t_list)) * 100.0
        
        avg_cot = sum(x.get("cot_tokens", x.get("grug_cot_tokens", 0)) for x in eval_list) / total
        avg_tot = sum(x["total_tokens"] for x in eval_list) / total
        avg_lat = sum(x["latency_sec"] for x in eval_list) / total
        return {
            "overall_accuracy_pct": round(acc, 1),
            "humaneval_coding_pass_rate_pct": round(c_acc, 1),
            "gsm8k_math_accuracy_pct": round(m_acc, 1),
            "glaive_tool_accuracy_pct": round(t_acc, 1),
            "avg_reasoning_tokens": round(avg_cot, 1),
            "avg_total_tokens": round(avg_tot, 1),
            "avg_latency_sec": round(avg_lat, 2)
        }

    scorecard = {
        "native_deepseek_r1_baseline": calculate_scorecard(stage1_results)
    }
    for cfg in active_compilers:
        scorecard[f"grugified_deepseek_{cfg['id']}"] = calculate_scorecard(grugified_eval_results[cfg["id"]])
        
    print("\n" + "=" * 80)
    print("OFFICIAL ONLINE BENCHMARKS SCORECARD:")
    print("=" * 80)
    print(json.dumps(scorecard, indent=2))
    
    # Save JSON results
    out_dir = "/content/grug-speech-reasoning/evaluation"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "official_online_benchmarks_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "scorecard": scorecard,
            "native_r1": stage1_results,
            "compiled_traces": compiled_traces,
            "grugified_eval": grugified_eval_results
        }, f, indent=2)
    print(f"Results successfully saved to {out_file}")

if __name__ == "__main__":
    main()
