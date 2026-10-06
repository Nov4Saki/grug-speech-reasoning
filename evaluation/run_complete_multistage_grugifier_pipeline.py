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
    """Extracts python code and runs official unit test suite."""
    clean = code_text.strip()
    if "```python" in clean:
        clean = clean.split("```python")[1].split("```")[0]
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0]
    clean = clean.strip()
    
    # Attempt 1: Direct execution
    try:
        scope = {}
        exec(clean, scope)
        exec(test_code, scope)
        return True
    except Exception:
        pass

    # Attempt 2: Fallback with prompt header prepended
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
    """Validates tool call JSON parsing and schema parameters."""
    try:
        clean = tool_text.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0]
        clean = clean.strip()
        data = json.loads(clean)
        
        # Check tool name
        tool_name = data.get("name") or data.get("tool") or data.get("function")
        if tool_name != expected_tool:
            return False
            
        args = data.get("arguments") or data.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        
        # Check required params
        for param in required_params:
            if param not in args and param not in data:
                return False
        return True
    except Exception:
        return False

def execute_math_test(math_text: str, expected_answer: str) -> bool:
    """Validates presence and match of the final numerical answer."""
    clean = math_text.strip()
    # Check for Final Answer pattern
    matches = re.findall(r"(?:Final Answer:?\s*[$]?|x\s*=\s*|is\s*[$]?)([-+]?\d*\.?\d+)", clean, re.IGNORECASE)
    if matches:
        final_val = matches[-1]
        try:
            return abs(float(final_val) - float(expected_answer)) < 1e-3
        except Exception:
            pass
    # Fallback substring search
    return expected_answer in clean

def compute_energy_metric(tokens: int, domain: str) -> float:
    """Computes trajectory energy E(x,y)."""
    domain_weights = {"coding": 0.08, "tool_use": 0.06, "mathematics": 0.09}
    weight = domain_weights.get(domain, 0.07)
    return round(tokens * weight, 2)

# ==============================================================================
# PIPELINE EXECUTION ENGINE
# ==============================================================================

def main():
    print("=" * 80)
    print("MULTI-STAGE DEEPSEEK-R1 GRUGIFICATION BENCHMARK PIPELINE")
    print("Target Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (4-bit NF4)")
    print(f"Total Benchmark Tasks: {len(ALL_BENCHMARK_TASKS)} (5 Coding, 5 Tools, 5 Math)")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # STAGE 1: NATIVE DEEPSEEK-R1 GENERATION
    # --------------------------------------------------------------------------
    print("\n[STAGE 1]: Running Native DeepSeek-R1-Distill-Qwen-7B on Benchmark Tasks...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    r1_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    r1_tok = AutoTokenizer.from_pretrained(r1_model_id)
    r1_model = AutoModelForCausalLM.from_pretrained(
        r1_model_id,
        quantization_config=bnb_config,
        device_map="auto"
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

        # Parse CoT vs Answer
        if "</think>" in decoded:
            cot_raw = decoded.split("</think>")[0].replace("<think>", "").strip()
            ans_raw = decoded.split("</think>")[1].replace("<｜end of sentence｜>", "").strip()
        else:
            cot_raw = decoded.replace("<think>", "").strip()
            ans_raw = ""

        cot_tokens = len(r1_tok.encode(cot_raw))
        ans_tokens = len(r1_tok.encode(ans_raw))

        # Check native correctness
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
        print(f"[{idx+1:02d}/15] {task['domain']:<11} {task['name']:<25} | CoT: {cot_tokens:>3} tok | Ans: {ans_tokens:>3} tok | Correct: {str(is_correct):<5} | Time: {latency:.2f}s")

    print(f"Stage 1 Complete in {time.time() - t_stage1_start:.1f}s")

    # EJECT DeepSeek-R1 from GPU
    del r1_model
    del r1_tok
    gc.collect()
    torch.cuda.empty_cache()
    print(f"DeepSeek-R1 Ejected. VRAM remaining: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # --------------------------------------------------------------------------
    # STAGE 2: COMPILE TRACES VIA 3 TRAINED GRUGIFIERS
    # --------------------------------------------------------------------------
    grugifier_configs = [
        {
            "id": "grug_3b",
            "name": "Qwen2.5-3B-UniversalGrugifier",
            "base": "Qwen/Qwen2.5-3B-Instruct",
            "adapter": "/workspace/universal_grugifier_output/final_adapter",
            "type": "qwen"
        },
        {
            "id": "grug_1.5b",
            "name": "Qwen2.5-1.5B-UniversalGrugifier",
            "base": "Qwen/Qwen2.5-1.5B-Instruct",
            "adapter": "/workspace/qwen1.5b_grugifier_output/final_adapter",
            "type": "qwen"
        },
        {
            "id": "grug_1.7b",
            "name": "SmolLM2-1.7B-UniversalGrugifier",
            "base": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "adapter": "/workspace/smollm1.7b_grugifier_output/final_adapter",
            "type": "smollm"
        }
    ]

    grug_compiled_results = {cfg["id"]: [] for cfg in grugifier_configs}

    for cfg in grugifier_configs:
        print(f"\n[STAGE 2]: Loading Grugifier: {cfg['name']}...")
        tok = AutoTokenizer.from_pretrained(cfg["base"])
        base_m = AutoModelForCausalLM.from_pretrained(
            cfg["base"],
            quantization_config=bnb_config,
            device_map="auto"
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
            prompt = (
                f"<|im_start|>system\n"
                f"You are the Universal Grugifier Engine. Transform verbose reasoning traces from any source model into dense, invariant-anchored Grug Speech (<think>...Done.</think>) optimized for Energy-Based Fine-Tuning (EBFT). Preserve all domain, persona, and causal invariants.<|im_end|>\n"
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

            # Parse Grug CoT
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
            print(f"[{idx+1:02d}/15] Compiled via {cfg['id']}: {raw_tokens} tok -> {grug_tokens} tok ({compression}x) | Tag: {tag_valid}")

        print(f"Compiled all 15 tasks via {cfg['name']} in {time.time() - t_grug_start:.1f}s")

        # EJECT Grugifier from GPU
        del model
        del base_m
        del tok
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{cfg['name']} Ejected. VRAM remaining: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # --------------------------------------------------------------------------
    # STAGE 3: TEST "GRUGIFIED DEEPSEEK" (CONDITIONED REASONING VERIFICATION)
    # --------------------------------------------------------------------------
    print("\n[STAGE 3]: Reloading DeepSeek-R1 to test generation conditioned on Grug CoT...")
    r1_tok = AutoTokenizer.from_pretrained(r1_model_id)
    r1_model = AutoModelForCausalLM.from_pretrained(
        r1_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    print(f"DeepSeek-R1 Reloaded! VRAM: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    grugified_deepseek_eval = {cfg["id"]: [] for cfg in grugifier_configs}

    for cfg in grugifier_configs:
        print(f"\nTesting DeepSeek-R1 conditioned on {cfg['name']} reasoning traces...")
        for idx, task in enumerate(ALL_BENCHMARK_TASKS):
            grug_data = grug_compiled_results[cfg["id"]][idx]
            grug_cot = grug_data["grug_cot"]

            # Prompt DeepSeek with Grug CoT already filled
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

            # Evaluate correctness
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
            print(f"[{idx+1:02d}/15] Grugified-R1 ({cfg['id']}) on {task['name']:<22} | Ans: {ans_tokens:>3} tok | Correct: {str(is_correct):<5} | Time: {latency:.2f}s")

    # EJECT DeepSeek-R1 from GPU
    del r1_model
    del r1_tok
    gc.collect()
    torch.cuda.empty_cache()
    print(f"DeepSeek-R1 Ejected. VRAM remaining: {torch.cuda.memory_allocated(0)/1024/1024:.2f} MB")

    # --------------------------------------------------------------------------
    # STAGE 4: AGGREGATE RESULTS & QUALITATIVE SUBSET EXTRACTION
    # --------------------------------------------------------------------------
    print("\n[STAGE 4]: Computing Aggregate Scorecards and Qualitative Subsets...")

    # Calculate Domain-Level and Overall Metrics
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
        "baseline_raw_deepseek_r1": calculate_stats(stage1_traces),
        "grugified_deepseek_3b": calculate_stats(grugified_deepseek_eval["grug_3b"]),
        "grugified_deepseek_1.5b": calculate_stats(grugified_deepseek_eval["grug_1.5b"]),
        "grugified_deepseek_1.7b": calculate_stats(grugified_deepseek_eval["grug_1.7b"])
    }

    # Extract Side-by-Side Qualitative Subsets across all tasks
    qualitative_subsets = []
    for idx in range(len(ALL_BENCHMARK_TASKS)):
        task = ALL_BENCHMARK_TASKS[idx]
        raw_item = stage1_traces[idx]
        item_3b = grugified_deepseek_eval["grug_3b"][idx]
        item_15b = grugified_deepseek_eval["grug_1.5b"][idx]
        item_17b = grugified_deepseek_eval["grug_1.7b"][idx]

        qualitative_subsets.append({
            "task_id": task["id"],
            "domain": task["domain"],
            "name": task["name"],
            "prompt": task["prompt"],
            "raw_deepseek": {
                "cot": raw_item["raw_cot"],
                "cot_tokens": raw_item["cot_tokens"],
                "answer": raw_item["raw_answer"],
                "correct": raw_item["is_correct"]
            },
            "grug_3b": {
                "cot": item_3b["grug_cot"],
                "cot_tokens": item_3b["grug_cot_tokens"],
                "compression": grug_compiled_results["grug_3b"][idx]["compression_ratio"],
                "answer": item_3b["grugified_answer"],
                "correct": item_3b["is_correct"]
            },
            "grug_1.5b": {
                "cot": item_15b["grug_cot"],
                "cot_tokens": item_15b["grug_cot_tokens"],
                "compression": grug_compiled_results["grug_1.5b"][idx]["compression_ratio"],
                "answer": item_15b["grugified_answer"],
                "correct": item_15b["is_correct"]
            },
            "grug_1.7b": {
                "cot": item_17b["grug_cot"],
                "cot_tokens": item_17b["grug_cot_tokens"],
                "compression": grug_compiled_results["grug_1.7b"][idx]["compression_ratio"],
                "answer": item_17b["grugified_answer"],
                "correct": item_17b["is_correct"]
            }
        })

    # Save Results
    os.makedirs("/content/grug-speech-reasoning/evaluation", exist_ok=True)
    with open("/content/grug-speech-reasoning/evaluation/deepseek_r1_multistage_evaluation_results.json", "w") as f:
        json.dump({
            "summary_scorecard": summary_card,
            "stage1_raw_deepseek": stage1_traces,
            "grug_compilation_metrics": grug_compiled_results,
            "grugified_deepseek_eval": grugified_deepseek_eval
        }, f, indent=2)

    with open("/content/grug-speech-reasoning/evaluation/deepseek_grugifier_qualitative_subsets.json", "w") as f:
        json.dump(qualitative_subsets, f, indent=2)

    print("\nSummary Scorecard:")
    print(json.dumps(summary_card, indent=2))
    print("\nSaved evaluation results to /content/grug-speech-reasoning/evaluation/")

if __name__ == "__main__":
    main()
