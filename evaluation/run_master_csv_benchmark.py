import os
import sys
import io
import re
import json
import time
import subprocess
import pandas as pd

# Setup paths
PROJECT_DIR = "/content/csv_project/final project"
sys.path.insert(0, PROJECT_DIR)

from tools import aggregate_tool, filter_tool, view_tool, transform_tool, chart_tool
import nodes

LLAMA_CLI = "/content/llama.cpp/build/bin/llama-cli"

MODELS = [
    {
        "name": "bonsai-8b-prismml",
        "path": "/content/models/Bonsai-8B.gguf",
        "category": "Base / 1-Bit",
        "param_b": 8.2,
        "is_grug": False,
        "chat_format": "qwen"
    },
    {
        "name": "qwen3.5-4b-grugspeech-native",
        "path": "/content/Qwen3.5-4B-GrugSpeech-Q4_K_M.gguf",
        "category": "Grug Native",
        "param_b": 4.0,
        "is_grug": True,
        "chat_format": "qwen"
    },
    {
        "name": "qwen3.5-4b-base",
        "path": "/content/models/base/Qwen3.5-4B-Q4_K_M.gguf",
        "category": "Base Peer",
        "param_b": 4.0,
        "is_grug": False,
        "chat_format": "qwen"
    },
    {
        "name": "gemma-4-e2b-grugspeech-native",
        "path": "/content/Gemma4-E2B-GrugSpeech-Q4_K_M.gguf",
        "category": "Grug Native",
        "param_b": 4.6,
        "is_grug": True,
        "chat_format": "gemma"
    },
    {
        "name": "gemma-4-e2b-base",
        "path": "/content/models/base/gemma-4-E2B-it-Q4_K_M.gguf",
        "category": "Base Peer",
        "param_b": 4.6,
        "is_grug": False,
        "chat_format": "gemma"
    },
    {
        "name": "minicpm5-2b-grugspeech-native",
        "path": "/content/MiniCPM5-2B-GrugSpeech-Q4_K_M.gguf",
        "category": "Grug Native",
        "param_b": 2.0,
        "is_grug": True,
        "chat_format": "minicpm"
    },
    {
        "name": "nanbeige4.2-3b-grugspeech-native",
        "path": "/content/Nanbeige4.2-3B-GrugSpeech-Q4_K_M.gguf",
        "category": "Grug Native",
        "param_b": 3.0,
        "is_grug": True,
        "chat_format": "qwen"
    },
    {
        "name": "qwen3.5-2b-grugspeech-native",
        "path": "/content/Qwen3.5-2B-GrugSpeech-Q4_K_M.gguf",
        "category": "Grug Native",
        "param_b": 1.9,
        "is_grug": True,
        "chat_format": "qwen"
    },
    {
        "name": "qwen3.5-2b-base",
        "path": "/content/models/base/Qwen3.5-2B-Q4_K_M.gguf",
        "category": "Base Peer",
        "param_b": 1.9,
        "is_grug": False,
        "chat_format": "qwen"
    },
    {
        "name": "bonsai-4b-prismml",
        "path": "/content/models/Bonsai-4B.gguf",
        "category": "Base / 1-Bit",
        "param_b": 4.0,
        "is_grug": False,
        "chat_format": "qwen"
    },
    {
        "name": "bonsai-1.7b-prismml",
        "path": "/content/models/Bonsai-1.7B.gguf",
        "category": "Base / 1-Bit",
        "param_b": 1.7,
        "is_grug": False,
        "chat_format": "qwen"
    }
]

TEST_CASES = [
    {
        "id": "T1_Simple_EN",
        "category": "Basic Aggregation (EN)",
        "query": "What is the total sales amount?",
        "expected_tool": "aggregate_tool"
    },
    {
        "id": "T2_MultiStep_EN",
        "category": "Multi-step Filter + Grouping (EN)",
        "query": "Filter rows where Total is greater than 500, then group by City and show the sum of Total for each city, sorted from highest to lowest",
        "expected_tool": "filter_tool"
    },
    {
        "id": "T3_Simple_AR",
        "category": "Basic Aggregation (AR)",
        "query": "احسب متوسط السعر لكل فئة من المنتجات",
        "expected_tool": "aggregate_tool"
    },
    {
        "id": "T4_MultiStep_AR",
        "category": "Multi-step Filter + Grouping (AR)",
        "query": "فلتر الصفوف اللي فيها الكمية أكبر من 5، وبعدين احسب مجموع المبيعات لكل فئة، وترتيب النتيجة من الأعلى للأقل",
        "expected_tool": "filter_tool"
    },
    {
        "id": "T5_Chart_Pie_AR",
        "category": "Pie Chart (AR)",
        "query": "عايز باي شارت يوضح نسبة مبيعات كل فئة من المنتجات",
        "expected_tool": "chart_tool"
    },
    {
        "id": "T6_Chart_GroupedBar_EN",
        "category": "Grouped Bar Chart (EN)",
        "query": "create a grouped bar chart showing total sales for each category, grouped by city",
        "expected_tool": "chart_tool"
    }
]

def build_prompt(df, user_query, is_grug, chat_format):
    columns = list(df.columns)
    raw_sample = json.dumps(df.head(2).to_dict(orient="records"), default=str)
    sample_data = nodes._sanitize(raw_sample)
    
    if is_grug:
        sys_msg = f"""You are a data analysis planner. Reason in Grug Speech inside <think> tags.
Dataframe Columns: {columns}
Sample Rows: {sample_data}

Available Tools:
{nodes.tool_descriptions}

Rules:
- Match user intent (Arabic/English) to tools.
- Output ordered steps in "steps" list.
- Use EXACT column names from dataframe (do not invent or rename e.g. 'Total').
- For charts: use chart_tool with chart_type 'pie', 'bar', 'line', 'scatter'.

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
Dataframe Columns: {columns}
Sample Rows: {sample_data}

Available Tools:
{nodes.tool_descriptions}

Rules:
- Match user intent (Arabic/English) to tools.
- Output ordered steps in "steps" list.
- Use EXACT column names from dataframe (do not invent or rename e.g. 'Total').
- For charts: use chart_tool with chart_type 'pie', 'bar', 'line', 'scatter'.

Output ONLY a valid JSON object matching:
{{
    "steps": [
        {{ "tool_name": "exact_tool_name", "tool_args": {{ "arg_name": "arg_value" }} }}
    ]
}}"""

    user_msg = f"User Query: {user_query}"
    
    if chat_format == "gemma":
        return f"<start_of_turn>user\n{sys_msg}\n\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"
    elif chat_format == "minicpm":
        return f"<user>\n{sys_msg}\n\n{user_msg}\n<assistant>\n"
    else:
        return f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

def clean_output(proc_stdout, fmt):
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

def extract_plan(output_text):
    think_text = ""
    if "<think>" in output_text and "</think>" in output_text:
        think_text = output_text.split("<think>")[1].split("</think>")[0].strip()
    elif "[Start thinking]" in output_text:
        t_part = output_text.split("[Start thinking]")[1]
        if "[End thinking]" in t_part:
            think_text = t_part.split("[End thinking]")[0].strip()
            
    body_text = output_text
    if "</think>" in body_text:
        body_text = body_text.split("</think>")[-1]
    elif "[End thinking]" in body_text:
        body_text = body_text.split("[End thinking]")[-1]
        
    code_m = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", body_text, re.DOTALL)
    if code_m:
        raw_json = code_m.group(1).strip()
    else:
        json_m = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[.*\])", body_text, re.DOTALL)
        raw_json = json_m.group(0).strip() if json_m else ""
        
    steps = []
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, list):
                steps = parsed
            elif isinstance(parsed, dict):
                if "steps" in parsed and isinstance(parsed["steps"], list):
                    steps = parsed["steps"]
                elif "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                    for tc in parsed["tool_calls"]:
                        steps.append({"tool_name": tc.get("name") or tc.get("tool_name"), "tool_args": tc.get("args") or tc.get("tool_args", {})})
                elif "tool_name" in parsed:
                    steps = [parsed]
        except Exception:
            steps = []
            
    think_tokens = int(len(think_text.split()) * 1.3) if think_text else 0
    total_tokens = int(len(output_text.split()) * 1.3)
    return think_text, think_tokens, total_tokens, steps

def execute_on_sales_df(df_base, steps):
    if not steps:
        return False, "No steps generated", None
        
    current_df = df_base.copy()
    last_res = None
    
    for idx, s in enumerate(steps):
        tool_name = s.get("tool_name")
        args = s.get("tool_args", {})
        
        # Normalize tool args for pandas execution
        if tool_name == "filter_tool":
            q_str = args.get("query_string") or args.get("query")
            if not q_str:
                col = args.get("column")
                cond = args.get("condition")
                val = args.get("value")
                if col and cond and val:
                    q_str = f"`{col}` {cond} {repr(val) if isinstance(val, str) else val}"
                elif col and val:
                    q_str = f"`{col}` == {repr(val) if isinstance(val, str) else val}"
            if not q_str:
                return False, f"Step {idx+1} filter_tool missing query", None
            try:
                current_df = filter_tool(current_df, q_str)
                last_res = current_df
            except Exception as e:
                return False, f"Step {idx+1} filter error: {e}", None
                
        elif tool_name == "aggregate_tool":
            col = args.get("column") or args.get("metric")
            op = args.get("operation") or args.get("op") or "sum"
            by_col = args.get("by_column") or args.get("by")
            if not col and by_col and "Total" in current_df.columns:
                col = "Total"
            if not col:
                return False, f"Step {idx+1} aggregate_tool missing column", None
            try:
                res = aggregate_tool(current_df, column=col, operation=op, by_column=by_col)
                if isinstance(res, pd.DataFrame):
                    current_df = res
                last_res = res
            except Exception as e:
                return False, f"Step {idx+1} aggregate error: {e}", None
                
        elif tool_name == "view_tool":
            sort_by = args.get("sort_by")
            asc = args.get("ascending", True)
            n_rows = args.get("n")
            # If sorting column was renamed to sum_Total, map back to Total
            if sort_by and sort_by not in current_df.columns:
                for alt in ["Total", "Price", "Quantity"]:
                    if alt in current_df.columns:
                        sort_by = alt
                        break
            try:
                res = view_tool(current_df, sort_by=sort_by, ascending=asc, n=n_rows)
                if isinstance(res, pd.DataFrame):
                    current_df = res
                last_res = res
            except Exception as e:
                return False, f"Step {idx+1} view error: {e}", None
                
        elif tool_name == "chart_tool":
            c_type = args.get("chart_type") or args.get("type") or "bar"
            x = args.get("x")
            y = args.get("y")
            color_by = args.get("color_by")
            out_p = os.path.join(PROJECT_DIR, "charts", f"bench_{int(time.time()*1000)}.html")
            os.makedirs(os.path.join(PROJECT_DIR, "charts"), exist_ok=True)
            try:
                chart_tool(current_df, chart_type=c_type, x=x, y=y, color_by=color_by, output_path=out_p)
                last_res = f"Chart generated: {c_type} (saved to {out_p})"
            except Exception as e:
                return False, f"Step {idx+1} chart error: {e}", None
                
        elif tool_name == "transform_tool":
            act = args.get("action")
            try:
                current_df = transform_tool(current_df, act, **args)
                last_res = current_df
            except Exception as e:
                return False, f"Step {idx+1} transform error: {e}", None
        else:
            return False, f"Unrecognized tool: {tool_name}", None

    summary_out = str(last_res) if last_res is not None else "Completed"
    if isinstance(last_res, pd.DataFrame):
        summary_out = f"DataFrame ({len(last_res)} rows x {len(last_res.columns)} cols)"
    return True, summary_out[:120], last_res

def main():
    print("="*100)
    print("  STARTING MASTER CSV BENCHMARK (sales.xlsx, 100,300 ROWS) ACROSS 11 MODELS")
    print("="*100)
    
    excel_path = os.path.join(PROJECT_DIR, "sales.xlsx")
    print(f"Loading {excel_path} ...")
    t0 = time.perf_counter()
    df = pd.read_excel(excel_path)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns in {time.perf_counter()-t0:.2f}s.\n")
    
    all_results = {}
    summary_table = []
    
    for m_idx, m_cfg in enumerate(MODELS, 1):
        m_name = m_cfg["name"]
        print(f"[{m_idx}/{len(MODELS)}] STRICT ISOLATION: {m_name} ({m_cfg['category']})")
        if not os.path.exists(m_cfg["path"]):
            print(f"  ❌ Model file not found at {m_cfg['path']}. Skipping.")
            continue
            
        m_tests = []
        
        for t_idx, tc in enumerate(TEST_CASES, 1):
            print(f"    --> ({t_idx}/6) {tc['id']}: {tc['category']} ...", end=" ", flush=True)
            prompt = build_prompt(df, tc["query"], m_cfg["is_grug"], m_cfg["chat_format"])
            cmd = [LLAMA_CLI, "-m", m_cfg["path"], "-ngl", "99", "-p", prompt, "-n", "256", "-st", "--simple-io"]
            
            t_start = time.perf_counter()
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            wall_t = time.perf_counter() - t_start
            
            combined = proc.stdout + "\n" + proc.stderr
            gen_m = re.search(r"Generation:\s*([\d\.]+)\s*t/s", combined)
            pre_m = re.search(r"Prompt:\s*([\d\.]+)\s*t/s", combined)
            gen_tps = float(gen_m.group(1)) if gen_m else 0.0
            prefill_tps = float(pre_m.group(1)) if pre_m else 0.0
            
            raw_out = clean_output(proc.stdout, m_cfg["chat_format"])
            think_text, cot_tok, total_tok, steps = extract_plan(raw_out)
            
            exec_ok, exec_msg, _ = execute_on_sales_df(df, steps)
            
            status_sym = "✅ EXEC OK" if exec_ok else "❌ EXEC FAIL"
            print(f"{status_sym} | {wall_t:.2f}s | Gen: {gen_tps:.1f} t/s | CoT: {cot_tok}t | Plan: {len(steps)} steps")
            
            m_tests.append({
                "model": m_name,
                "test_id": tc["id"],
                "category": tc["category"],
                "wall_sec": round(wall_t, 2),
                "gen_tps": round(gen_tps, 1),
                "prefill_tps": round(prefill_tps, 1),
                "cot_tokens": cot_tok,
                "total_tokens": total_tok,
                "steps_count": len(steps),
                "execution_success": exec_ok,
                "execution_message": exec_msg,
                "steps": steps,
                "raw_output": raw_out[:300]
            })
            
        all_results[m_name] = m_tests
        
        exec_pass_count = sum(1 for t in m_tests if t["execution_success"])
        avg_wall = sum(t["wall_sec"] for t in m_tests) / len(m_tests)
        avg_cot = sum(t["cot_tokens"] for t in m_tests) / len(m_tests)
        avg_total_tok = sum(t["total_tokens"] for t in m_tests) / len(m_tests)
        avg_gen = sum(t["gen_tps"] for t in m_tests) / len(m_tests)
        avg_pre = sum(t["prefill_tps"] for t in m_tests) / len(m_tests)
        
        summary_table.append({
            "model": m_name,
            "category": m_cfg["category"],
            "exec_pass_rate": f"{exec_pass_count}/6 ({exec_pass_count/6*100:.1f}%)",
            "exec_pass_pct": round(exec_pass_count/6*100, 1),
            "avg_wall_s": round(avg_wall, 2),
            "avg_gen_tps": round(avg_gen, 1),
            "avg_prefill_tps": round(avg_pre, 1),
            "avg_cot_tokens": round(avg_cot, 1),
            "avg_total_tokens": round(avg_total_tok, 1)
        })
        
        print(f"    [UNLOADED {m_name} & RELEASED VRAM]\n")
        time.sleep(1)
        
    out_results_path = os.path.join(PROJECT_DIR, "master_csv_benchmarks_all_models.json")
    out_summary_path = os.path.join(PROJECT_DIR, "master_csv_benchmarks_summary.json")
    
    with open(out_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)
        
    print("="*110)
    print("  MASTER CSV BENCHMARK SUMMARY (sales.xlsx, 100,300 ROWS)")
    print("="*110)
    print(f"{'Model Name':<32} | {'Category':<12} | {'Exec Pass':<14} | {'Wall (s)':<9} | {'Gen (t/s)':<10} | {'Prefill (t/s)':<14} | {'CoT':<6} | {'Total Tok'}")
    print("-" * 110)
    for row in summary_table:
        print(f"{row['model']:<32} | {row['category']:<12} | {row['exec_pass_rate']:<14} | {row['avg_wall_s']:<9.2f} | {row['avg_gen_tps']:<10.1f} | {row['avg_prefill_tps']:<14.1f} | {row['avg_cot_tokens']:<6.1f} | {row['avg_total_tokens']:.1f}")
        
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()
