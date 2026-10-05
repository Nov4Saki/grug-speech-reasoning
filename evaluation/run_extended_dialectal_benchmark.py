import os
import sys
import re
import json
import time
import subprocess

LLAMA_CLI = "/content/llama.cpp/build/bin/llama-cli"

MODELS = [
    {
        "name": "qwen3.5-4b-grugspeech-native",
        "path": "/content/Qwen3.5-4B-GrugSpeech-Q4_K_M.gguf",
        "is_grug": True,
        "chat_format": "qwen"
    },
    {
        "name": "qwen3.5-4b-base",
        "path": "/content/models/base/Qwen3.5-4B-Q4_K_M.gguf",
        "is_grug": False,
        "chat_format": "qwen"
    },
    {
        "name": "bonsai-8b-prismml",
        "path": "/content/models/Bonsai-8B.gguf",
        "is_grug": False,
        "chat_format": "qwen"
    },
    {
        "name": "gemma-4-e2b-grugspeech-native",
        "path": "/content/Gemma4-E2B-GrugSpeech-Q4_K_M.gguf",
        "is_grug": True,
        "chat_format": "gemma"
    },
    {
        "name": "gemma-4-e2b-base",
        "path": "/content/models/base/gemma-4-E2B-it-Q4_K_M.gguf",
        "is_grug": False,
        "chat_format": "gemma"
    },
    {
        "name": "minicpm5-2b-grugspeech-native",
        "path": "/content/MiniCPM5-2B-GrugSpeech-Q4_K_M.gguf",
        "is_grug": True,
        "chat_format": "minicpm"
    },
    {
        "name": "nanbeige4.2-3b-grugspeech-native",
        "path": "/content/Nanbeige4.2-3B-GrugSpeech-Q4_K_M.gguf",
        "is_grug": True,
        "chat_format": "qwen"
    }
]

BASE_SCHEMA_CONTEXT = """Dataframe Columns: ['Order_ID', 'Customer', 'Product', 'Category', 'Price', 'Quantity', 'Discount_Percent', 'Total', 'City', 'Payment_Method', 'Status', 'Order_Date']
Sample Rows: [{"Order_ID": 1001, "Customer": "Ahmed", "Product": "Laptop", "Category": "Electronics", "Price": 1200, "Quantity": 2, "Discount_Percent": 0.1, "Total": 2160, "City": "Cairo", "Payment_Method": "Credit Card", "Status": "Completed", "Order_Date": "2024-01-15"}]

Available Tools:
- aggregate_tool: (column, operation, by_column). Operations: sum, mean, min, max, count. by_column for grouping. Note: preserved column name (e.g. 'Total').
- filter_tool: (column, condition, value) or (query). E.g. query: "Total > 1000" or column: "Status", condition: "==", value: "Completed".
- view_tool: (sort_by, ascending). Sorts data from highest (ascending=False) or lowest.
- chart_tool: (chart_type, x, y, color_by). Types: 'bar', 'pie', 'line', 'scatter'."""

EXTENDED_TESTS = [
    {
        "id": "E1_Egypt_Slang_Filter_Chart",
        "name": "Egyptian Slang Electronics City Chart",
        "language": "Arabic (Egyptian Slang)",
        "query": "عايز أعرف أكتر مدينة دفعت فلوس في الإلكترونيات وتعملي بار شارت",
        "expected_tools": ["filter_tool", "chart_tool"],
        "expected_chart_type": "bar"
    },
    {
        "id": "E2_Gulf_MultiFilter_Sort",
        "name": "Gulf Dialect High-Value Completed Orders",
        "language": "Arabic (Gulf Dialect)",
        "query": "طلع لي كل الطلبات اللي قيمتها فوق الالف ريال وتمت بنجاح ورتبها من الأعلى للأقل",
        "expected_tools": ["filter_tool", "view_tool"],
        "expected_chart_type": None
    },
    {
        "id": "E3_Payment_Method_Multi_Agg",
        "name": "Payment Method Dual Aggregation",
        "language": "Arabic (Standard)",
        "query": "احسب متوسط السعر وإجمالي المبيعات لكل وسيلة دفع",
        "expected_tools": ["aggregate_tool"],
        "expected_chart_type": None
    },
    {
        "id": "E4_Asyncio_Task_Destroyed",
        "name": "Asyncio Task Destroyed Triage",
        "language": "Code/Technical",
        "query": "We are getting RuntimeError: Task was destroyed but it is pending in asyncio loop. Explain root cause in Grug Speech and provide fix.",
        "expected_tools": [],
        "expected_chart_type": None
    }
]

def format_prompt(model_cfg, test_case):
    is_grug = model_cfg["is_grug"]
    fmt = model_cfg["chat_format"]
    query = test_case["query"]
    is_code = (test_case["id"] == "E4_Asyncio_Task_Destroyed")
    
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
    if "[ Prompt:" in out:
        out = out.split("[ Prompt:")[0]
    return out.strip()

def evaluate_response(test_case, output_text):
    is_code = (test_case["id"] == "E4_Asyncio_Task_Destroyed")
    
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
        has_cause = any(k in output_text.lower() for k in ["reference", "garbage", "destroyed", "gc", "pending", "loop closed", "not awaited", "unawaited"])
        has_fix = any(k in output_text.lower() for k in ["await", "asyncio.gather", "tasks.add", "set()", "gather", "wait", "reference"])
        passed = has_cause and has_fix
        return {
            "passed": passed,
            "json_valid": True,
            "intent_correct": passed,
            "hallucination": False,
            "reasoning_tokens": think_tokens,
            "total_tokens": total_tokens,
            "think_preview": think_text[:100] if think_text else "(No think tag)"
        }
        
    body_text = output_text
    if "</think>" in body_text:
        body_text = body_text.split("</think>")[-1]
    elif "[End thinking]" in body_text:
        body_text = body_text.split("[End thinking]")[-1]
        
    clean_json_str = ""
    code_match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", body_text, re.DOTALL)
    if code_match:
        clean_json_str = code_match.group(1).strip()
    else:
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
        
        if not expected:
            intent_correct = True
        elif set(expected).issubset(set(tools_found)) or (len(expected) == 1 and expected[0] in tools_found):
            intent_correct = True
        else:
            intent_correct = False
            
        if test_case.get("expected_chart_type"):
            chart_type_found = False
            for s in steps:
                args = s.get("tool_args", {})
                if s.get("tool_name") == "chart_tool" and (args.get("chart_type") == test_case["expected_chart_type"] or args.get("type") == test_case["expected_chart_type"]):
                    chart_type_found = True
                    break
            intent_correct = intent_correct and chart_type_found

        dumped = json.dumps(steps, ensure_ascii=False)
        if any(h in dumped for h in ["sum_Total", "sum(Total)", "avg_Total", "المدينة", "فلوس", "طلبات"]):
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

def main():
    print("="*90)
    print("  RUNNING EXTENDED DIALECTAL & DOMAIN-SPECIFIC BENCHMARK")
    print("="*90)
    
    extended_results = {}
    
    for m_idx, m_cfg in enumerate(MODELS, 1):
        m_name = m_cfg["name"]
        print(f"\n[{m_idx}/{len(MODELS)}] STRICT ISOLATION: {m_name}")
        
        if not os.path.exists(m_cfg["path"]):
            print(f"    ❌ Model not found: {m_cfg['path']}")
            continue
            
        m_tests = []
        for t_idx, tc in enumerate(EXTENDED_TESTS, 1):
            print(f"    --> ({t_idx}/4) {tc['id']}: {tc['name']} ...", end=" ", flush=True)
            prompt = format_prompt(m_cfg, tc)
            cmd = [LLAMA_CLI, "-m", m_cfg["path"], "-ngl", "99", "-p", prompt, "-n", "256", "-st", "--simple-io"]
            
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
                t_wall = time.perf_counter() - t0
                combined = proc.stdout + "\n" + proc.stderr
                
                gen_m = re.search(r"Generation:\s*([\d\.]+)\s*t/s", combined)
                pre_m = re.search(r"Prompt:\s*([\d\.]+)\s*t/s", combined)
                gen_tps = float(gen_m.group(1)) if gen_m else 0.0
                prefill_tps = float(pre_m.group(1)) if pre_m else 0.0
                
                assistant_out = extract_clean_output(proc.stdout, m_cfg["chat_format"])
                eval_res = evaluate_response(tc, assistant_out)
                
                status_sym = "✅ PASS" if eval_res["passed"] else ("⚠️ FAIL(Intent)" if eval_res["json_valid"] else "❌ FAIL(Schema)")
                print(f"{status_sym} | {t_wall:.2f}s | Gen: {gen_tps:.1f} t/s | CoT: {eval_res['reasoning_tokens']}t")
                
                m_tests.append({
                    "model": m_name,
                    "test_id": tc["id"],
                    "wall_sec": round(t_wall, 2),
                    "gen_tps": round(gen_tps, 1),
                    "prefill_tps": round(prefill_tps, 1),
                    "output_preview": assistant_out[:140].replace("\n", " "),
                    "raw_output": assistant_out,
                    **eval_res
                })
            except Exception as e:
                print(f"ERROR: {e}")
                
        extended_results[m_name] = m_tests
        print(f"    [UNLOADED {m_name}]")
        time.sleep(1)
        
    out_path = "/content/extended_dialectal_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extended_results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved extended benchmark results to {out_path}")

if __name__ == "__main__":
    main()
