import os
import sys
import re
import json
import time
import subprocess

LLAMA_CLI = "/content/llama.cpp/build/bin/llama-cli"

MODELS = [
    {
        "name": "gemma-4-e2b-grugspeech-native",
        "path": "/content/Gemma4-E2B-GrugSpeech-Q4_K_M.gguf",
        "family": "Google Gemma 4",
        "param_b": 4.6,
        "is_grug": True,
        "chat_format": "gemma"
    },
    {
        "name": "gemma-4-e2b-base",
        "path": "/content/models/base/gemma-4-E2B-it-Q4_K_M.gguf",
        "family": "Google Gemma 4",
        "param_b": 4.6,
        "is_grug": False,
        "chat_format": "gemma"
    },
    {
        "name": "qwen3.5-2b-grugspeech-native",
        "path": "/content/Qwen3.5-2B-GrugSpeech-Q4_K_M.gguf",
        "family": "Alibaba Qwen 3.5",
        "param_b": 1.9,
        "is_grug": True,
        "chat_format": "qwen"
    },
    {
        "name": "qwen3.5-2b-base",
        "path": "/content/models/base/Qwen3.5-2B-Q4_K_M.gguf",
        "family": "Alibaba Qwen 3.5",
        "param_b": 1.9,
        "is_grug": False,
        "chat_format": "qwen"
    },
    {
        "name": "qwen3.5-4b-grugspeech-native",
        "path": "/content/Qwen3.5-4B-GrugSpeech-Q4_K_M.gguf",
        "family": "Alibaba Qwen 3.5",
        "param_b": 4.0,
        "is_grug": True,
        "chat_format": "qwen"
    },
    {
        "name": "qwen3.5-4b-base",
        "path": "/content/models/base/Qwen3.5-4B-Q4_K_M.gguf",
        "family": "Alibaba Qwen 3.5",
        "param_b": 4.0,
        "is_grug": False,
        "chat_format": "qwen"
    },
    {
        "name": "bonsai-8b-prismml",
        "path": "/content/models/Bonsai-8B.gguf",
        "family": "PrismML 1-Bit",
        "param_b": 8.2,
        "is_grug": False,
        "chat_format": "qwen"
    },
    {
        "name": "bonsai-4b-prismml",
        "path": "/content/models/Bonsai-4B.gguf",
        "family": "PrismML 1-Bit",
        "param_b": 4.0,
        "is_grug": False,
        "chat_format": "qwen"
    },
    {
        "name": "bonsai-1.7b-prismml",
        "path": "/content/models/Bonsai-1.7B.gguf",
        "family": "PrismML 1-Bit",
        "param_b": 1.7,
        "is_grug": False,
        "chat_format": "qwen"
    },
    {
        "name": "minicpm5-2b-grugspeech-native",
        "path": "/content/MiniCPM5-2B-GrugSpeech-Q4_K_M.gguf",
        "family": "OpenBMB MiniCPM",
        "param_b": 2.0,
        "is_grug": True,
        "chat_format": "minicpm"
    },
    {
        "name": "nanbeige4.2-3b-grugspeech-native",
        "path": "/content/Nanbeige4.2-3B-GrugSpeech-Q4_K_M.gguf",
        "family": "BOSS Zhipin Nanbeige",
        "param_b": 3.0,
        "is_grug": True,
        "chat_format": "qwen"
    }
]

BASE_SCHEMA_CONTEXT = """Dataframe Columns: ['Order_ID', 'Customer', 'Product', 'Category', 'Price', 'Quantity', 'Discount_Percent', 'Total', 'City', 'Payment_Method', 'Status', 'Order_Date']
Sample Rows: [{"Order_ID": 1001, "Customer": "Ahmed", "Product": "Laptop", "Category": "Electronics", "Price": 1200, "Quantity": 2, "Discount_Percent": 0.1, "Total": 2160, "City": "Cairo", "Payment_Method": "Credit Card", "Status": "Completed", "Order_Date": "2024-01-15"}]

Available Tools:
- aggregate_tool: (column, operation, by_column). Operations: sum, mean, min, max, count. by_column for grouping. Note: preserved column name (e.g. 'Total').
- filter_tool: (column, condition, value) or (query). E.g. query: "Quantity > 5" or "Total > 500".
- view_tool: (sort_by, ascending). Sorts data from highest (ascending=False) or lowest.
- chart_tool: (chart_type, x, y, color_by). Types: 'bar', 'pie', 'line', 'scatter'."""

TEST_CASES = [
    {
        "id": "T1_Simple_EN",
        "name": "Basic Aggregation (EN)",
        "language": "English",
        "complexity": "Low",
        "query": "What is the total sales amount?",
        "expected_tools": ["aggregate_tool"],
        "expected_keys": ["Total", "sum"]
    },
    {
        "id": "T2_MultiStep_EN",
        "name": "Multi-step Pipeline (EN)",
        "language": "English",
        "complexity": "High",
        "query": "Filter rows where Total is greater than 500, then group by City and show the sum of Total for each city, sorted from highest to lowest",
        "expected_tools": ["filter_tool", "aggregate_tool", "view_tool"],
        "expected_keys": ["Total", "City"]
    },
    {
        "id": "T3_Simple_AR",
        "name": "Basic Aggregation (AR Standard)",
        "language": "Arabic (MSA)",
        "complexity": "Low",
        "query": "احسب متوسط السعر لكل فئة من المنتجات",
        "expected_tools": ["aggregate_tool"],
        "expected_keys": ["Price", "mean", "Category"]
    },
    {
        "id": "T4_MultiStep_AR",
        "name": "Multi-step Pipeline (AR Slang)",
        "language": "Arabic (Dialect)",
        "complexity": "High",
        "query": "فلتر الصفوف اللي فيها الكمية أكبر من 5، وبعدين احسب مجموع المبيعات لكل فئة، وترتيب النتيجة من الأعلى للأقل",
        "expected_tools": ["filter_tool", "aggregate_tool", "view_tool"],
        "expected_keys": ["Quantity", "Total", "Category"]
    },
    {
        "id": "T5_Chart_Pie_AR",
        "name": "Pie Chart Intent (AR Slang)",
        "language": "Arabic (Dialect)",
        "complexity": "Medium",
        "query": "عايز باي شارت يوضح نسبة مبيعات كل فئة من المنتجات",
        "expected_tools": ["chart_tool"],
        "expected_keys": ["pie", "Category", "Total"]
    },
    {
        "id": "T6_Chart_GroupedBar_EN",
        "name": "Grouped Bar Chart (EN)",
        "language": "English",
        "complexity": "Medium",
        "query": "create a grouped bar chart showing total sales for each category, grouped by city",
        "expected_tools": ["chart_tool"],
        "expected_keys": ["bar", "Category", "Total", "City"]
    },
    {
        "id": "T7_Debug_BugTriage",
        "name": "Code Bug Triage (Token Economy)",
        "language": "Code/Technical",
        "complexity": "High Reasoning",
        "query": "We are getting IndexError: list index out of range on line 12 when parsing tokens[1]. Explain root cause and provide fix.",
        "expected_tools": [],
        "expected_keys": ["len", "IndexError", "tokens"]
    }
]

def format_prompt(model_cfg, test_case):
    is_grug = model_cfg["is_grug"]
    fmt = model_cfg["chat_format"]
    query = test_case["query"]
    is_code = (test_case["id"] == "T7_Debug_BugTriage")
    
    if is_code:
        if is_grug:
            sys_msg = "You are an expert engineer. Reason in Grug Speech inside <think> tags (Goal: ... Cause: ... Fix: ... Done.). Keep solution minimal."
        else:
            sys_msg = "You are an expert software engineer. Provide a thorough explanation of the error and resolution."
        user_msg = query
    else:
        if is_grug:
            sys_msg = f"""You are a data analysis planner. Reason in Grug Speech inside <think> tags.
{BASE_SCHEMA_CONTEXT}
Output format:
<think>
(Telegraphic Grug Speech reasoning)
</think>
{{
    "steps": [
        {{ "tool_name": "exact_tool_name", "tool_args": {{ "arg_name": "arg_value" }} }}
    ]
}}"""
        else:
            sys_msg = f"""You are an expert data analysis planner. Break down the user's request into a sequential execution plan of tools.
{BASE_SCHEMA_CONTEXT}
Output ONLY a valid JSON object matching this schema:
{{
    "steps": [
        {{ "tool_name": "exact_tool_name", "tool_args": {{ "arg_name": "arg_value" }} }}
    ]
}}"""
        user_msg = f"User Query: {query}"
        
    if fmt == "gemma":
        return f"<start_of_turn>user\n{sys_msg}\n\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"
    elif fmt == "minicpm":
        return f"<user>\n{sys_msg}\n\n{user_msg}\n<assistant>\n"
    else:
        return f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

def extract_clean_output(proc_stdout, fmt):
    out = proc_stdout
    if "\navailable commands:" in out:
        out = out.split("\navailable commands:")[-1]
        
    if "... (truncated)" in out:
        out = out.split("... (truncated)")[-1]
    elif fmt == "gemma" and "<start_of_turn>model" in out:
        out = out.split("<start_of_turn>model")[-1]
    elif fmt == "minicpm" and "<assistant>" in out:
        out = out.split("<assistant>")[-1]
    elif "<|im_start|>assistant" in out:
        out = out.split("<|im_start|>assistant")[-1]
    elif "\n> " in out:
        out = out.split("\n> ", 1)[-1]
        lines = out.split("\n", 1)
        if len(lines) > 1:
            out = lines[1]
            
    # Then strip trailing llama-cli timing banner
    if "[ Prompt:" in out:
        out = out.split("[ Prompt:")[0]
        
    return out.strip()

def evaluate_response(test_case, output_text):
    is_code = (test_case["id"] == "T7_Debug_BugTriage")
    
    # Extract think block across different thinking conventions
    think_text = ""
    if "<think>" in output_text and "</think>" in output_text:
        think_text = output_text.split("<think>")[1].split("</think>")[0].strip()
    elif "[Start thinking]" in output_text:
        t_part = output_text.split("[Start thinking]")[1]
        if "[End thinking]" in t_part:
            think_text = t_part.split("[End thinking]")[0].strip()
        elif "<final_response>" in t_part:
            think_text = t_part.split("<final_response>")[0].strip()
        elif "```json" in t_part:
            think_text = t_part.split("```json")[0].strip()
        elif "{" in t_part:
            think_text = t_part.split("{")[0].strip()
        else:
            think_text = t_part[:200].strip()
            
    think_tokens = int(len(think_text.split()) * 1.3) if think_text else 0
    total_tokens = int(len(output_text.split()) * 1.3)
    
    if is_code:
        has_len_check = any(k in output_text.lower() for k in ["len(", "len (", "length", ">= 2", "> 1", ">1", ">=2"])
        has_cause = any(k in output_text.lower() for k in ["empty", "fewer than", "less than 2", "out of range", "index"])
        passed = has_len_check and has_cause
        return {
            "passed": passed,
            "json_valid": True,
            "intent_correct": passed,
            "hallucination": False,
            "reasoning_tokens": think_tokens,
            "total_tokens": total_tokens,
            "think_preview": think_text[:100] if think_text else "(No think tag)"
        }
        
    # Isolate body text after thinking tags
    body_text = output_text
    if "</think>" in body_text:
        body_text = body_text.split("</think>")[-1]
    elif "[End thinking]" in body_text:
        body_text = body_text.split("[End thinking]")[-1]
        
    # Extract JSON
    clean_json_str = ""
    # Try finding inside markdown code blocks
    code_match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", body_text, re.DOTALL)
    if code_match:
        clean_json_str = code_match.group(1).strip()
    else:
        # Fallback to direct json match
        json_match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[.*\])", body_text, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0).strip()
            
    json_valid = False
    intent_correct = False
    hallucination = False
    steps = []
    
    if clean_json_str:
        try:
            parsed = json.loads(clean_json_str)
            if isinstance(parsed, list) and len(parsed) > 0:
                json_valid = True
                steps = parsed
            elif isinstance(parsed, dict):
                if "steps" in parsed and isinstance(parsed["steps"], list) and len(parsed["steps"]) > 0:
                    json_valid = True
                    steps = parsed["steps"]
                elif "tool_calls" in parsed and isinstance(parsed["tool_calls"], list) and len(parsed["tool_calls"]) > 0:
                    json_valid = True
                    for tc in parsed["tool_calls"]:
                        steps.append({
                            "tool_name": tc.get("name") or tc.get("tool_name"),
                            "tool_args": tc.get("args") or tc.get("tool_args", {})
                        })
                elif "tool_name" in parsed:
                    json_valid = True
                    steps = [parsed]
                elif "tool" in parsed or "operation" in parsed:
                    json_valid = True
                    steps = [{"tool_name": parsed.get("tool") or parsed.get("operation"), "tool_args": parsed}]
        except Exception:
            json_valid = False
            
    if json_valid:
        tools_found = [s.get("tool_name") for s in steps if isinstance(s, dict)]
        expected = test_case["expected_tools"]
        
        # Check tool intent
        if not expected:
            intent_correct = True
        elif set(expected).issubset(set(tools_found)) or (len(expected) == 1 and expected[0] in tools_found):
            intent_correct = True
        else:
            intent_correct = False
            
        # Specific check for chart tool type
        if test_case["id"] == "T5_Chart_Pie_AR":
            pie_found = False
            for s in steps:
                args = s.get("tool_args", {})
                if s.get("tool_name") == "chart_tool" and (args.get("chart_type") == "pie" or args.get("type") == "pie"):
                    pie_found = True
                    break
            intent_correct = pie_found
            
        if test_case["id"] == "T6_Chart_GroupedBar_EN":
            bar_found = False
            for s in steps:
                args = s.get("tool_args", {})
                if s.get("tool_name") == "chart_tool" and (args.get("chart_type") == "bar" or args.get("type") == "bar"):
                    bar_found = True
                    break
            intent_correct = bar_found

        # Hallucination check
        dumped = json.dumps(steps, ensure_ascii=False)
        if any(h in dumped for h in ["sum_Total", "sum(Total)", "avg_Total", "الفئة", "نسبة"]):
            hallucination = True
            
    passed = json_valid and intent_correct and not hallucination
    
    return {
        "passed": passed,
        "json_valid": json_valid,
        "intent_correct": intent_correct,
        "hallucination": hallucination,
        "reasoning_tokens": think_tokens,
        "total_tokens": total_tokens,
        "think_preview": think_text[:100] if think_text else "(No think tag)"
    }

def run_single_benchmark(model_cfg, test_case):
    prompt = format_prompt(model_cfg, test_case)
    cmd = [
        LLAMA_CLI,
        "-m", model_cfg["path"],
        "-ngl", "99",
        "-p", prompt,
        "-n", "256",
        "-st",
        "--simple-io"
    ]
    
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        t_wall = time.perf_counter() - t0
        combined = proc.stdout + "\n" + proc.stderr
        
        prompt_m = re.search(r"Prompt:\s*([\d\.]+)\s*t/s", combined)
        gen_m = re.search(r"Generation:\s*([\d\.]+)\s*t/s", combined)
        
        prefill_tps = float(prompt_m.group(1)) if prompt_m else 0.0
        gen_tps = float(gen_m.group(1)) if gen_m else 0.0
        
        assistant_output = extract_clean_output(proc.stdout, model_cfg["chat_format"])
        eval_metrics = evaluate_response(test_case, assistant_output)
        
        return {
            "model": model_cfg["name"],
            "test_id": test_case["id"],
            "wall_sec": round(t_wall, 2),
            "prefill_tps": round(prefill_tps, 1),
            "gen_tps": round(gen_tps, 1),
            "output_preview": assistant_output[:140].replace("\n", " "),
            "raw_output": assistant_output,
            **eval_metrics
        }
    except subprocess.TimeoutExpired:
        return {
            "model": model_cfg["name"],
            "test_id": test_case["id"],
            "wall_sec": 60.0,
            "prefill_tps": 0.0,
            "gen_tps": 0.0,
            "passed": False,
            "json_valid": False,
            "intent_correct": False,
            "hallucination": False,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "think_preview": "TIMEOUT",
            "output_preview": "TIMEOUT"
        }
    except Exception as e:
        return {
            "model": model_cfg["name"],
            "test_id": test_case["id"],
            "wall_sec": 0.0,
            "prefill_tps": 0.0,
            "gen_tps": 0.0,
            "passed": False,
            "json_valid": False,
            "intent_correct": False,
            "hallucination": False,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "think_preview": f"ERROR: {e}",
            "output_preview": f"ERROR: {e}"
        }

def main():
    print("="*90)
    print("  STARTING ACCURATE FULL-SUIT BENCHMARK: GRUG VS NON-GRUG ACROSS 11 MODELS")
    print("="*90)

    all_results = {}

    for m_idx, m_cfg in enumerate(MODELS, 1):
        m_name = m_cfg["name"]
        print(f"\n[{m_idx}/{len(MODELS)}] STRICT ISOLATION: {m_name}")
        print(f"    Path: {m_cfg['path']} | Size: {m_cfg['param_b']}B | Grug: {m_cfg['is_grug']}")

        if not os.path.exists(m_cfg["path"]):
            print(f"    ❌ Model file not found at {m_cfg['path']}. Skipping.")
            continue

        m_test_results = []

        for t_idx, tc in enumerate(TEST_CASES, 1):
            print(f"    --> ({t_idx}/7) {tc['id']}: {tc['name']} ({tc['language']}) ...", end=" ", flush=True)
            res = run_single_benchmark(m_cfg, tc)
            status_sym = "✅ PASS" if res["passed"] else ("⚠️ FAIL(Alias)" if res["hallucination"] else ("⚠️ FAIL(Intent)" if (res["json_valid"] and not res["intent_correct"]) else "❌ FAIL(Schema)"))
            print(f"{status_sym} | {res['wall_sec']}s | Gen: {res['gen_tps']} t/s | Pre: {res['prefill_tps']} t/s | CoT: {res['reasoning_tokens']} tok")
            m_test_results.append(res)

        all_results[m_name] = m_test_results
        print(f"    [UNLOADING MODEL {m_name} & CLEARING MEMORY... DONE]\n")
        time.sleep(1)

    output_path = "/content/full_suit_benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved raw full benchmark results to {output_path}")

    print("\n" + "="*110)
    print(f"  FULL-SUIT MASTER LEADERBOARD & COMPARATIVE METRICS")
    print("="*110)
    print(f"{'Model Name':<32} | {'Type':<8} | {'Pass Rate':<9} | {'Avg Wall':<8} | {'Avg Gen':<10} | {'Avg Pre':<10} | {'Avg CoT':<8} | {'Schema'}")
    print("-" * 110)

    summary_rows = []
    for m_name, tests in all_results.items():
        cfg = next(c for c in MODELS if c["name"] == m_name)
        m_type = "Grug" if cfg["is_grug"] else "Base"
        passed = sum(1 for t in tests if t["passed"])
        total = len(tests)
        avg_wall = sum(t["wall_sec"] for t in tests) / total if total else 0
        avg_gen = sum(t["gen_tps"] for t in tests if t["gen_tps"] > 0) / max(1, sum(1 for t in tests if t["gen_tps"] > 0))
        avg_pre = sum(t["prefill_tps"] for t in tests if t["prefill_tps"] > 0) / max(1, sum(1 for t in tests if t["prefill_tps"] > 0))
        avg_cot = sum(t["reasoning_tokens"] for t in tests) / total if total else 0
        schema_ok = sum(1 for t in tests if t["json_valid"])

        print(f"{m_name:<32} | {m_type:<8} | {passed}/{total} ({passed/total*100:4.1f}%) | {avg_wall:6.2f}s | {avg_gen:8.1f} t/s | {avg_pre:8.1f} t/s | {avg_cot:6.1f} t | {schema_ok}/{total}")
        summary_rows.append({
            "model": m_name,
            "type": m_type,
            "pass_rate_pct": round(passed/total*100, 1),
            "avg_wall_s": round(avg_wall, 2),
            "avg_gen_tps": round(avg_gen, 1),
            "avg_prefill_tps": round(avg_pre, 1),
            "avg_cot_tokens": round(avg_cot, 1),
            "schema_ok_ratio": f"{schema_ok}/{total}"
        })

    with open("/content/full_suit_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()
