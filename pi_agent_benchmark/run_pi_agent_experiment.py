import os
import sys
import io
import time
import re
import json
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Ensure utf-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
BASE_DIR = "/content/pi_agent_app_benchmark"

SYSTEM_PROMPT_NORMAL = """You are an autonomous AI coding agent following the minimalist Pi Agent / Oh My Pi philosophy.
You have two primitive tools:
1. write_file(path, content)
2. bash(command)

TASK: Build an Event Analytics Service in Python from scratch.
You must implement:
- storage.py: SQLite event storage with schema (id INTEGER PRIMARY KEY, tenant_id TEXT, event_type TEXT, val REAL, timestamp REAL). Methods: init_db(db_path), insert_event(db_path, tenant_id, event_type, val), get_events(db_path, tenant_id=None).
- rate_limiter.py: TokenBucketLimiter(capacity: int, refill_rate: float) with allow_request(tenant_id: str) -> bool.
- analytics.py: compute_metrics(events: list) returning dict with 'total_count', 'mean_val', 'max_val'.
- test_suite.py: pytest test functions testing all components thoroughly.

Think through the problem step by step before outputting the code files.
Format your output as:
<think>
[Your full reasoning process]
</think>
```python:storage.py
[code]
```
```python:rate_limiter.py
[code]
```
```python:analytics.py
[code]
```
```python:test_suite.py
[code]
```
"""

SYSTEM_PROMPT_GRUG = """You are an autonomous AI coding agent following the minimalist Pi Agent / Oh My Pi philosophy.
Reason using dense Grug Speech inside <think> tags.

TASK: Build an Event Analytics Service in Python from scratch.
You must implement:
- storage.py: SQLite event storage with schema (id INTEGER PRIMARY KEY, tenant_id TEXT, event_type TEXT, val REAL, timestamp REAL). Methods: init_db(db_path), insert_event(db_path, tenant_id, event_type, val), get_events(db_path, tenant_id=None).
- rate_limiter.py: TokenBucketLimiter(capacity: int, refill_rate: float) with allow_request(tenant_id: str) -> bool.
- analytics.py: compute_metrics(events: list) returning dict with 'total_count', 'mean_val', 'max_val'.
- test_suite.py: pytest test functions testing all components thoroughly.

Output format:
<think>
Domain: coding
Goal: [State goal]
Invariants: [Conserved properties: sqlite schema, token bucket math, metrics, test assertions]
Action: [Sequence of files to write]
Done.
</think>
```python:storage.py
[code]
```
```python:rate_limiter.py
[code]
```
```python:analytics.py
[code]
```
```python:test_suite.py
[code]
```
"""

USER_REQUEST = "Build the Event Analytics Service from scratch with storage, rate limiting, analytics, and pytest test suite."

def extract_files_and_think(text):
    think_trace = ""
    if "<think>" in text and "</think>" in text:
        think_trace = text.split("<think>")[1].split("</think>")[0].strip()
    
    files = {}
    pattern = r'```python:([a-zA-Z0-9_\.]+)\n([\s\S]*?)```'
    matches = re.findall(pattern, text)
    for fname, code in matches:
        files[fname] = code.strip()
        
    return think_trace, files

def test_code_with_pytest(target_dir):
    try:
        res = subprocess.run(
            ["pytest", "-v", "test_suite.py"],
            cwd=target_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )
        return res.returncode == 0, res.stdout + "\n" + res.stderr
    except Exception as e:
        return False, str(e)

def main():
    print("="*80)
    print("  PI AGENT / OH MY PI APP CODING BENCHMARK: NORMAL VS GRUG REASONING")
    print(f"  Model: {MODEL_ID}")
    print("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print("Loading DeepSeek-R1-Distill-Qwen-7B in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print("Model loaded successfully!")
    
    benchmark_results = {}
    
    experiments = [
        ("Normal_Discursive", SYSTEM_PROMPT_NORMAL, "/content/pi_agent_app_benchmark/normal"),
        ("Grug_Invariant", SYSTEM_PROMPT_GRUG, "/content/pi_agent_app_benchmark/grug")
    ]
    
    for name, sys_prompt, out_dir in experiments:
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'#'*80}")
        print(f"  RUNNING CONDITION: {name}")
        print(f"{'#'*80}")
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": USER_REQUEST}
        ]
        
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        
        print(f"Prompt tokens: {input_len}. Generating code...")
        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        t1 = time.time()
        duration = t1 - t0
        
        gen_tokens = out_ids[0][input_len:]
        output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        num_tokens = len(gen_tokens)
        tps = num_tokens / duration if duration > 0 else 0
        
        think_trace, files = extract_files_and_think(output_text)
        think_words = len(think_trace.split()) if think_trace else 0
        think_tokens = len(tokenizer.encode(think_trace)) if think_trace else 0
        
        print(f"Generation completed in {duration:.2f}s ({num_tokens} tokens, {tps:.1f} tok/s)")
        print(f"Thinking trace length: {think_words} words ({think_tokens} tokens)")
        print(f"Extracted {len(files)} files: {list(files.keys())}")
        
        # Write files
        for fname, code in files.items():
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"  -> Wrote {fpath} ({len(code.splitlines())} lines)")
            
        # Execute pytest
        tests_passed, test_log = test_code_with_pytest(out_dir)
        print(f"Pytest Execution: {'PASSED' if tests_passed else 'FAILED'}")
        
        benchmark_results[name] = {
            "duration_sec": round(duration, 2),
            "total_tokens": num_tokens,
            "tps": round(tps, 1),
            "think_words": think_words,
            "think_tokens": think_tokens,
            "files_generated": list(files.keys()),
            "tests_passed": tests_passed,
            "test_output_snippet": test_log[:400],
            "think_trace_sample": think_trace[:500] + ("..." if len(think_trace) > 500 else ""),
            "full_think_trace": think_trace,
            "full_output": output_text
        }
        
    # Save results
    results_path = os.path.join(BASE_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*80)
    print("  EXPERIMENT SUMMARY")
    print("="*80)
    for name, data in benchmark_results.items():
        print(f"Condition: {name}")
        print(f"  - Think Tokens: {data['think_tokens']} tokens ({data['think_words']} words)")
        print(f"  - Total Tokens: {data['total_tokens']} tokens")
        print(f"  - Duration: {data['duration_sec']}s")
        print(f"  - Files Generated: {len(data['files_generated'])}")
        print(f"  - Pytest Passed: {data['tests_passed']}")
    print("="*80)

if __name__ == "__main__":
    main()
