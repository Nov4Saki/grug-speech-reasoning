import os
import sys
import gc
import json
import time
import re
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

TOKEN = "HF_TOKEN_REDACTED"

# ==============================================================================
# BENCHMARK TASK DEFINITIONS (15 REAL-WORLD & PUBLIC TASKS)
# ==============================================================================

HUMANEVAL_TASKS = [
    {
        "id": "humaneval_0",
        "domain": "coding",
        "name": "has_close_elements",
        "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold. \"\"\"\n",
        "instruction": "Complete the Python function. Return the implementation inside ```python ```.",
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\ncheck(has_close_elements)\n"
    },
    {
        "id": "humaneval_1",
        "domain": "coding",
        "name": "separate_paren_groups",
        "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input is a string containing multiple groups of nested parentheses. Separate those groups into separate strings and return the list of those. \"\"\"\n",
        "instruction": "Complete the Python function. Return the implementation inside ```python ```.",
        "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n    assert candidate('(()(())((())))') == ['(()(())((())))']\ncheck(separate_paren_groups)\n"
    },
    {
        "id": "humaneval_2",
        "domain": "coding",
        "name": "truncate_number",
        "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into and integer part and decimals. Return the decimal part of the number. \"\"\"\n",
        "instruction": "Complete the Python function. Return the implementation inside ```python ```.",
        "test": "def check(candidate):\n    assert abs(candidate(3.5) - 0.5) < 1e-4\n    assert abs(candidate(1.33) - 0.33) < 1e-4\n    assert abs(candidate(123.456) - 0.456) < 1e-4\ncheck(truncate_number)\n"
    },
    {
        "id": "humaneval_3",
        "domain": "coding",
        "name": "below_zero",
        "prompt": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account. Detect if at any point the balance of account falls below zero. \"\"\"\n",
        "instruction": "Complete the Python function. Return the implementation inside ```python ```.",
        "test": "def check(candidate):\n    assert candidate([]) == False\n    assert candidate([1, 2, -3, 1, 2, -3]) == False\n    assert candidate([1, 2, -4, 5, 6]) == True\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False\ncheck(below_zero)\n"
    },
    {
        "id": "humaneval_4",
        "domain": "coding",
        "name": "mean_absolute_deviation",
        "prompt": "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation around the mean of this dataset. \"\"\"\n",
        "instruction": "Complete the Python function. Return the implementation inside ```python ```.",
        "test": "def check(candidate):\n    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-4\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-4\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-4\ncheck(mean_absolute_deviation)\n"
    }
]

TOOL_TASKS = [
    {
        "id": "tool_0",
        "domain": "tool_use",
        "name": "execute_sql_query",
        "prompt": "Execute the database query tool `execute_sql_query` to query active users created after 2026-01-01 with a limit of 10 in read_only mode.",
        "instruction": "Output a valid JSON object with 'name' and 'arguments' for the tool call. Do not add markdown outside the JSON block.",
        "expected_tool": "execute_sql_query",
        "required_params": ["query", "read_only"]
    },
    {
        "id": "tool_1",
        "domain": "tool_use",
        "name": "create_stripe_payment_intent",
        "prompt": "Call `create_stripe_payment_intent` to charge 4500 cents (USD) for customer 'cus_9921' with payment method 'card' and confirm immediately.",
        "instruction": "Output a valid JSON object with 'name' and 'arguments' for the tool call. Do not add markdown outside the JSON block.",
        "expected_tool": "create_stripe_payment_intent",
        "required_params": ["amount", "currency", "customer_id", "confirm"]
    },
    {
        "id": "tool_2",
        "domain": "tool_use",
        "name": "deploy_kubernetes_deployment",
        "prompt": "Trigger `deploy_kubernetes_deployment` to update deployment 'payment-service' in namespace 'production' to image 'registry.internal/payment:v2.4.0' with 3 replicas.",
        "instruction": "Output a valid JSON object with 'name' and 'arguments' for the tool call. Do not add markdown outside the JSON block.",
        "expected_tool": "deploy_kubernetes_deployment",
        "required_params": ["name", "namespace", "image", "replicas"]
    },
    {
        "id": "tool_3",
        "domain": "tool_use",
        "name": "fetch_weather_forecast",
        "prompt": "Use `fetch_weather_forecast` to get a 7-day metric forecast for latitude 37.7749 and longitude -122.4194.",
        "instruction": "Output a valid JSON object with 'name' and 'arguments' for the tool call. Do not add markdown outside the JSON block.",
        "expected_tool": "fetch_weather_forecast",
        "required_params": ["lat", "lon", "units", "days"]
    },
    {
        "id": "tool_4",
        "domain": "tool_use",
        "name": "dispatch_pagerduty_alert",
        "prompt": "Send a critical incident notification via `dispatch_pagerduty_alert` for service 'auth-cluster' with severity 'CRITICAL' and summary '502 Bad Gateway rate at 42%'.",
        "instruction": "Output a valid JSON object with 'name' and 'arguments' for the tool call. Do not add markdown outside the JSON block.",
        "expected_tool": "dispatch_pagerduty_alert",
        "required_params": ["service_id", "severity", "summary"]
    }
]

MATH_TASKS = [
    {
        "id": "math_0",
        "domain": "mathematics",
        "name": "arithmetic_train_speed",
        "prompt": "A train travels at 60 mph for 2 hours and then 75 mph for 3 hours. What is its average speed for the entire trip?",
        "instruction": "Calculate the exact average speed. Conclude with 'Final Answer: <number> mph'.",
        "expected_answer": "69",
        "invariants": ["60", "2", "75", "3", "345", "5", "69"]
    },
    {
        "id": "math_1",
        "domain": "mathematics",
        "name": "system_of_equations",
        "prompt": "A baker sells cookies for $2 each and muffins for $3 each. If 50 items were sold for a total of $120, how many muffins were sold?",
        "instruction": "Solve for the number of muffins. Conclude with 'Final Answer: <number>'.",
        "expected_answer": "20",
        "invariants": ["2c + 3m = 120", "c + m = 50", "20"]
    },
    {
        "id": "math_2",
        "domain": "mathematics",
        "name": "geometric_progression",
        "prompt": "What is the 6th term of a geometric sequence where the first term is 3 and the common ratio is 2?",
        "instruction": "Calculate the 6th term. Conclude with 'Final Answer: <number>'.",
        "expected_answer": "96",
        "invariants": ["3", "2", "3 * 2^5", "96"]
    },
    {
        "id": "math_3",
        "domain": "mathematics",
        "name": "discount_coupon",
        "prompt": "A store offers 25% off an $80 jacket. An additional 10% coupon applies to the discounted price. What is the final price in dollars?",
        "instruction": "Calculate the final price. Conclude with 'Final Answer: $<number>'.",
        "expected_answer": "54",
        "invariants": ["80", "25%", "60", "10%", "54"]
    },
    {
        "id": "math_4",
        "domain": "mathematics",
        "name": "linear_equation",
        "prompt": "Solve for x in the equation: 4x - 7 = 2x + 15.",
        "instruction": "Solve for x. Conclude with 'Final Answer: x = <number>'.",
        "expected_answer": "11",
        "invariants": ["4x - 7", "2x + 15", "2x = 22", "11"]
    }
]

ALL_BENCHMARK_TASKS = [
    HUMANEVAL_TASKS[0], # has_close_elements
    HUMANEVAL_TASKS[2], # truncate_number
    TOOL_TASKS[1],      # create_stripe_payment_intent
    TOOL_TASKS[2],      # deploy_kubernetes_deployment
    MATH_TASKS[0],      # arithmetic_train_speed
    MATH_TASKS[3]       # discount_coupon
]

# ==============================================================================
# VALIDATION UTILITIES
# ==============================================================================

def execute_code_test(code_text: str, test_code: str, prompt_header: str = "") -> bool:
    clean = code_text.strip()
    if "```python" in clean:
        clean = clean.split("```python")[1].split("```")[0]
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0]
    clean = clean.strip()

    try:
        scope = {}
        exec(clean, scope)
        exec(test_code, scope)
        return True
    except Exception:
        pass

    if prompt_header:
        try:
            scope = {}
            exec(prompt_header + "\n" + clean, scope)
            exec(test_code, scope)
            return True
        except Exception:
            pass

    return False

def execute_tool_test(tool_text: str, expected_tool: str, required_params: List[str]) -> bool:
    try:
        clean = tool_text.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0]
        clean = clean.strip()
        data = json.loads(clean)

        tool_name = data.get("name") or data.get("tool") or data.get("function")
        if tool_name != expected_tool:
            return False

        args = data.get("arguments") or data.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass

        for param in required_params:
            if param not in args and param not in data:
                return False
        return True
    except Exception:
        return False

def execute_math_test(math_text: str, expected_answer: str) -> bool:
    clean = math_text.strip()
    matches = re.findall(r"(?:Final Answer:?\s*[$]?|x\s*=\s*|is\s*[$]?)([-+]?\d*\.?\d+)", clean, re.IGNORECASE)
    if matches:
        final_val = matches[-1]
        try:
            return abs(float(final_val) - float(expected_answer)) < 1e-3
        except Exception:
            pass
    return expected_answer in clean

def compute_energy_metric(tokens: int, domain: str) -> float:
    domain_weights = {"coding": 0.08, "tool_use": 0.06, "mathematics": 0.09}
    weight = domain_weights.get(domain, 0.07)
    return round(tokens * weight, 2)

# ==============================================================================
# MAIN MULTI-STAGE MODERN BENCHMARK ENGINE
# ==============================================================================

def main():
    print("=" * 80)
    print("MODERN ARCHITECTURE DEEPSEEK-R1 GRUGIFICATION BENCHMARK SUITE")
    print("Target Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (NF4 BF16)")
    print("Compilers: Qwen3.5-4B, Qwen3.5-2B, Gemma4-E4B, LFM2.5-2.6B")
    print(f"Total Benchmark Tasks: {len(ALL_BENCHMARK_TASKS)} (Coding, Tools, Math)")
    print("=" * 80)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    r1_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # --------------------------------------------------------------------------
    # STAGE 1: NATIVE DEEPSEEK-R1 GENERATION
    # --------------------------------------------------------------------------
    print("\n[STAGE 1]: Running Native DeepSeek-R1-Distill-Qwen-7B on Benchmark Tasks...")
    r1_tok = AutoTokenizer.from_pretrained(r1_model_id, token=TOKEN)
    r1_model = AutoModelForCausalLM.from_pretrained(
        r1_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=TOKEN
    )
    print(f"DeepSeek-R1 Loaded! VRAM allocated: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    stage1_traces = []
    t_stage1_start = time.time()

    for idx, task in enumerate(ALL_BENCHMARK_TASKS):
        full_user_content = f"{task['prompt']}\n{task['instruction']}"
        msgs = [{"role": "user", "content": full_user_content}]
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

        if task["domain"] == "coding":
            is_correct = execute_code_test(ans_raw, task["test"], task["prompt"])
        elif task["domain"] == "tool_use":
            is_correct = execute_tool_test(ans_raw, task["expected_tool"], task["required_params"])
        else:
            is_correct = execute_math_test(ans_raw, task["expected_answer"])

        stage1_traces.append({
            "task_id": task["id"],
            "domain": task["domain"],
            "name": task["name"],
            "prompt": full_user_content,
            "raw_cot": cot_raw,
            "raw_answer": ans_raw,
            "cot_tokens": cot_tokens,
            "ans_tokens": ans_tokens,
            "total_tokens": cot_tokens + ans_tokens,
            "latency_sec": round(latency, 2),
            "is_correct": is_correct,
            "energy": compute_energy_metric(cot_tokens + ans_tokens, task["domain"])
        })
        print(f"[{idx+1:02d}/06] {task['domain']:<11} {task['name']:<25} | CoT: {cot_tokens:>3} tok | Ans: {ans_tokens:>3} tok | Correct: {str(is_correct):<5} | Time: {latency:.2f}s")

    print(f"Stage 1 Complete in {time.time() - t_stage1_start:.1f}s")

    # EJECT DeepSeek-R1 from GPU
    del r1_model
    del r1_tok
    gc.collect()
    torch.cuda.empty_cache()
    print(f"DeepSeek-R1 Ejected. VRAM remaining: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # --------------------------------------------------------------------------
    # STAGE 2: COMPILE TRACES VIA MODERN GRUGIFIERS
    # --------------------------------------------------------------------------
    modern_compiler_configs = [
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

    # Filter to configs whose adapter exists
    active_configs = [c for c in modern_compiler_configs if os.path.exists(c["adapter"])]
    print(f"\nActive Modern Compilers detected: {[c['name'] for c in active_configs]}")

    grug_compiled_results = {cfg["id"]: [] for cfg in active_configs}

    for cfg in active_configs:
        print(f"\n[STAGE 2]: Loading Modern Grugifier: {cfg['name']}...")
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
        model = PeftModel.from_pretrained(base_m, cfg["adapter"])
        print(f"{cfg['name']} Loaded! VRAM: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

        t_grug_start = time.time()
        for idx, item in enumerate(stage1_traces):
            user_content = (
                f"[Source Model Family]: deepseek-r1-7b\n"
                f"[Domain]: {item['domain']}\n"
                f"[Language / Dialect]: en_standard\n"
                f"[Task Prompt]:\n{item['prompt']}\n\n"
                f"[Raw Verbose Reasoning]:\n{item['raw_cot']}"
            )
            system_msg = "You are the Universal Grugifier Engine. Transform verbose reasoning traces from any source model into dense, invariant-anchored Grug Speech (<think>...Done.</think>) optimized for Energy-Based Fine-Tuning (EBFT). Preserve all domain, persona, and causal invariants."

            if cfg["format"] == "qwen":
                prompt = (
                    f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                    f"<|im_start|>user\n{user_content}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            elif cfg["format"] == "gemma":
                prompt = (
                    f"<bos><|turn>system\n{system_msg}<turn|>\n"
                    f"<|turn>user\n{user_content}<turn|>\n"
                    f"<|turn>model\n"
                )
            elif cfg["format"] == "lfm":
                prompt = (
                    f"<|startoftext|><|im_start|>system\n{system_msg}<|im_end|>\n"
                    f"<|im_start|>user\n{user_content}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

            inputs = tok(prompt, return_tensors="pt").to("cuda")

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.2,
                    pad_token_id=tok.eos_token_id
                )
            latency = time.time() - t0
            gen_text = tok.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)

            if "<think>" in gen_text and "</think>" in gen_text:
                grug_cot = gen_text.split("<think>")[1].split("</think>")[0].strip()
                tag_valid = True
            elif "<think>" in gen_text:
                grug_cot = gen_text.split("<think>")[1].strip()
                tag_valid = False
            else:
                grug_cot = gen_text.strip()
                tag_valid = False

            grug_tokens = len(tok.encode(grug_cot))
            raw_tokens = item["cot_tokens"]
            compression = round(raw_tokens / max(1, grug_tokens), 2)
            energy_red = round((1.0 - (grug_tokens / max(1, raw_tokens))) * 100.0, 1)

            grug_compiled_results[cfg["id"]].append({
                "task_id": item["task_id"],
                "grug_cot": grug_cot,
                "grug_tokens": grug_tokens,
                "compression_ratio": compression,
                "energy_reduction_pct": energy_red,
                "tag_valid": tag_valid,
                "latency_sec": round(latency, 2)
            })
            print(f"[{idx+1:02d}/06] Compiled via {cfg['id']}: {raw_tokens} tok -> {grug_tokens} tok ({compression}x) | Tag: {tag_valid}")

        print(f"Compiled all tasks via {cfg['name']} in {time.time() - t_grug_start:.1f}s")

        # EJECT Compiler from GPU
        del model
        del base_m
        del tok
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{cfg['name']} Ejected. VRAM remaining: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # --------------------------------------------------------------------------
    # STAGE 3: TEST "GRUGIFIED DEEPSEEK" CONDITIONED REASONING
    # --------------------------------------------------------------------------
    print("\n[STAGE 3]: Reloading DeepSeek-R1 to test conditioned execution on modern Grug traces...")
    r1_tok = AutoTokenizer.from_pretrained(r1_model_id, token=TOKEN)
    r1_model = AutoModelForCausalLM.from_pretrained(
        r1_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=TOKEN
    )
    print(f"DeepSeek-R1 Reloaded! VRAM: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    grugified_deepseek_eval = {cfg["id"]: [] for cfg in active_configs}

    for cfg in active_configs:
        print(f"\nTesting DeepSeek-R1 conditioned on {cfg['name']} reasoning traces...")
        for idx, task in enumerate(ALL_BENCHMARK_TASKS):
            grug_data = grug_compiled_results[cfg["id"]][idx]
            grug_cot = grug_data["grug_cot"]

            conditioned_prompt = (
                f"<｜begin of sentence｜><｜User｜>{task['prompt']}\n{task['instruction']}"
                f"<｜Assistant｜><think>\n{grug_cot}\n</think>\n"
            )
            inputs = r1_tok(conditioned_prompt, return_tensors="pt").to("cuda")

            t0 = time.time()
            with torch.no_grad():
                outputs = r1_model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.2,
                    pad_token_id=r1_tok.eos_token_id
                )
            latency = time.time() - t0
            ans_text = r1_tok.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)
            ans_text = ans_text.replace("<｜end of sentence｜>", "").strip()

            ans_tokens = len(r1_tok.encode(ans_text))
            total_tokens = grug_data["grug_tokens"] + ans_tokens

            if task["domain"] == "coding":
                is_correct = execute_code_test(ans_text, task["test"], task["prompt"])
            elif task["domain"] == "tool_use":
                is_correct = execute_tool_test(ans_text, task["expected_tool"], task["required_params"])
            else:
                is_correct = execute_math_test(ans_text, task["expected_answer"])

            grugified_deepseek_eval[cfg["id"]].append({
                "task_id": task["id"],
                "domain": task["domain"],
                "name": task["name"],
                "grug_cot": grug_cot,
                "grugified_answer": ans_text,
                "grug_cot_tokens": grug_data["grug_tokens"],
                "ans_tokens": ans_tokens,
                "total_tokens": total_tokens,
                "latency_sec": round(latency, 2),
                "is_correct": is_correct,
                "energy": compute_energy_metric(total_tokens, task["domain"])
            })
            print(f"[{idx+1:02d}/06] Grugified-R1 ({cfg['id']}) on {task['name']:<22} | Ans: {ans_tokens:>3} tok | Correct: {str(is_correct):<5} | Time: {latency:.2f}s")

    # EJECT DeepSeek-R1 from GPU
    del r1_model
    del r1_tok
    gc.collect()
    torch.cuda.empty_cache()
    print(f"DeepSeek-R1 Ejected. VRAM remaining: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # --------------------------------------------------------------------------
    # STAGE 4: AGGREGATE RESULTS & QUALITATIVE ANALYSIS
    # --------------------------------------------------------------------------
    print("\n[STAGE 4]: Computing Scorecards and Qualitative Differences...")

    def calculate_stats(eval_list):
        total = len(eval_list)
        acc = sum(1 for x in eval_list if x["is_correct"]) / total * 100.0
        n_c = max(1, sum(1 for x in eval_list if x["domain"] == "coding"))
        n_t = max(1, sum(1 for x in eval_list if x["domain"] == "tool_use"))
        n_m = max(1, sum(1 for x in eval_list if x["domain"] == "mathematics"))
        coding_acc = sum(1 for x in eval_list if x["domain"] == "coding" and x["is_correct"]) / n_c * 100.0
        tool_acc = sum(1 for x in eval_list if x["domain"] == "tool_use" and x["is_correct"]) / n_t * 100.0
        math_acc = sum(1 for x in eval_list if x["domain"] == "mathematics" and x["is_correct"]) / n_m * 100.0
        avg_cot = sum(x.get("cot_tokens", x.get("grug_cot_tokens", 0)) for x in eval_list) / total
        avg_tot = sum(x["total_tokens"] for x in eval_list) / total
        avg_time = sum(x["latency_sec"] for x in eval_list) / total
        avg_energy = sum(x["energy"] for x in eval_list) / total
        return {
            "overall_accuracy": round(acc, 1),
            "coding_accuracy": round(coding_acc, 1),
            "tool_accuracy": round(tool_acc, 1),
            "math_accuracy": round(math_acc, 1),
            "avg_cot_tokens": round(avg_cot, 1),
            "avg_total_tokens": round(avg_tot, 1),
            "avg_latency_sec": round(avg_time, 2),
            "avg_energy": round(avg_energy, 2)
        }

    summary_card = {
        "baseline_raw_deepseek_r1": calculate_stats(stage1_traces)
    }
    for cfg in active_configs:
        summary_card[f"grugified_deepseek_{cfg['id']}"] = calculate_stats(grugified_deepseek_eval[cfg["id"]])

    # Qualitative comparison subset
    qualitative_subset = []
    for idx, task in enumerate(ALL_BENCHMARK_TASKS):
        base_item = stage1_traces[idx]
        entry = {
            "task_id": task["id"],
            "name": task["name"],
            "domain": task["domain"],
            "prompt": task["prompt"],
            "baseline_cot": base_item["raw_cot"][:300] + ("..." if len(base_item["raw_cot"]) > 300 else ""),
            "baseline_cot_tokens": base_item["cot_tokens"],
            "baseline_answer": base_item["raw_answer"][:200],
            "baseline_correct": base_item["is_correct"],
            "compilers": {}
        }
        for cfg in active_configs:
            c_data = grug_compiled_results[cfg["id"]][idx]
            r1_cond = grugified_deepseek_eval[cfg["id"]][idx]
            entry["compilers"][cfg["id"]] = {
                "compiler_name": cfg["name"],
                "grug_cot": c_data["grug_cot"],
                "grug_tokens": c_data["grug_tokens"],
                "compression_ratio": c_data["compression_ratio"],
                "r1_answer": r1_cond["grugified_answer"][:200],
                "r1_correct": r1_cond["is_correct"],
                "r1_latency": r1_cond["latency_sec"]
            }
        qualitative_subset.append(entry)

    # Save to disk
    eval_dir = "/content/grug-speech-reasoning/evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    out_eval_path = os.path.join(eval_dir, "deepseek_modern_grugifiers_benchmark_results.json")
    out_qual_path = os.path.join(eval_dir, "deepseek_modern_grugifier_qualitative.json")

    with open(out_eval_path, "w") as f:
        json.dump({
            "summary_scorecard": summary_card,
            "raw_stage1_traces": stage1_traces,
            "grug_compiled": grug_compiled_results,
            "conditioned_eval": grugified_deepseek_eval
        }, f, indent=2)

    with open(out_qual_path, "w") as f:
        json.dump(qualitative_subset, f, indent=2)

    print("\n" + "=" * 80)
    print("MODERN ARCHITECTURE EVALUATION SCORECARD:")
    print("=" * 80)
    print(json.dumps(summary_card, indent=2))
    print(f"\nArtifacts saved to {out_eval_path} and {out_qual_path}")

if __name__ == "__main__":
    main()
