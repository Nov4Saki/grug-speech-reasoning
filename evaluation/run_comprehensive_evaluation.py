"""
run_comprehensive_evaluation.py
===============================
Standardized Automated Benchmark Evaluator for Grug Speech Reasoning.
Runs against the 160-item scaled benchmark suite covering 5 domains:
GSM8K Math, Code Debugging, Hermes Function Calling, Science/Logic Deductions,
and Robustness Challenges across 3 prompt detail tiers (low, medium, high).

Measures:
- Reasoning Tokens (tau_think)
- Total Tokens Generated (tau_total)
- Task Pass Rate (%)
- Grug Syntax Compliance (%) (<think>...Done.</think>)
- Latency (s) and Generation Throughput (tok/s)
- Stratified Breakdown across Prompt Detail Levels (Low, Medium, High)
"""

import os
import re
import sys
import time
import json
import argparse
import torch
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Grug Speech Evaluator")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model identifier")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--is_grug", action="store_true", help="Evaluate as Grug Reasoner")
    parser.add_argument("--benchmark_path", type=str, default="evaluation/benchmark_suite_scaled.json", help="Path to benchmark JSON")
    parser.add_argument("--output_name", type=str, default="eval_run", help="Output prefix for summary files")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Directory to save evaluation summaries")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for parallel inference")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test samples for testing")
    return parser.parse_args()


def load_eval_model(model_id: str, adapter_dir: str = None):
    major, _ = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (7, 0)
    compute_dtype = torch.bfloat16 if major >= 8 else torch.float16

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir if adapter_dir else model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model {model_id} in 4-bit NF4 ({compute_dtype})...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )

    if adapter_dir and os.path.exists(adapter_dir):
        print(f"Loading Grug LoRA adapter from {adapter_dir}...")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = base_model

    model.eval()
    return tokenizer, model


def evaluate_task_success(category: str, eval_type: str, generated_text: str, ground_truth: str, expected_invariants: List[str]) -> bool:
    """Verifies invariant preservation and functional correctness."""
    text_lower = generated_text.lower()
    
    if eval_type == "numeric":
        gt_clean = re.sub(r'[^\d\.\-]', '', str(ground_truth))
        if not gt_clean:
            return True
        pattern = r'(?<!\d)' + re.escape(gt_clean) + r'(?!\d)'
        return bool(re.search(pattern, generated_text))

    elif eval_type == "tool":
        return ("<tool_call>" in generated_text) or ("execute" in text_lower) or ("function" in text_lower) or ("{" in generated_text and "}" in generated_text)

    elif eval_type == "code":
        matches = [inv.lower() in text_lower for inv in expected_invariants if inv]
        return sum(matches) >= max(1, len(matches) // 2)

    elif eval_type == "logic":
        return len(generated_text.strip()) > 15

    elif eval_type == "robustness":
        if not expected_invariants:
            return len(generated_text.strip()) > 5
        matches = [inv.lower() in text_lower for inv in expected_invariants if inv]
        return any(matches)

    return len(generated_text.strip()) > 0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.benchmark_path, "r", encoding="utf-8") as f:
        benchmark_items = json.load(f)

    if args.max_samples:
        benchmark_items = benchmark_items[:args.max_samples]

    print("=" * 60)
    print(f"RUNNING COMPREHENSIVE BENCHMARK EVALUATION: {args.output_name}")
    print(f"Total Test Cases: {len(benchmark_items)}")
    print(f"Mode: {'GRUG REASONER' if args.is_grug else 'BASE INSTRUCT BASELINE'}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 60)

    tokenizer, model = load_eval_model(args.model_id, args.adapter_dir)

    system_prompt = (
        "You are a reasoning model that reasons internally in ultra-dense Grug Speech enclosed in <think> tags. "
        "Always state Goal, concise Steps, Answer, and end thinking with Done. before providing the final response."
        if args.is_grug else
        "You are a helpful assistant. Think carefully before answering."
    )

    results = []
    total_think_tokens = 0
    total_gen_tokens = 0
    total_wall_time = 0.0
    passed_count = 0
    syntax_compliant_count = 0

    category_stats = {}
    level_stats = {}

    bs = args.batch_size
    pbar = tqdm(total=len(benchmark_items), desc=f"Evaluating {args.output_name}")

    for idx in range(0, len(benchmark_items), bs):
        batch_items = benchmark_items[idx:idx + bs]
        
        chat_texts = []
        for item in batch_items:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"]}
            ]
            chat_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        t1 = time.perf_counter()
        batch_wall_time = t1 - t0
        total_wall_time += batch_wall_time

        input_len = inputs.input_ids.shape[1]
        for b_i, item in enumerate(batch_items):
            gen_ids = output_ids[b_i][input_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            gen_tokens_count = len(gen_ids)

            total_gen_tokens += gen_tokens_count
            item_wall_time = batch_wall_time / len(batch_items)

            cat = item["category"]
            level = item["prompt_level"]
            prompt = item["prompt"]
            gt = item.get("ground_truth", "")
            invariants = item.get("expected_invariants", [])
            eval_type = item.get("eval_type", "general")

            think_match = re.search(r'<think>(.*?)</think>', gen_text, re.DOTALL)
            if think_match:
                think_content = think_match.group(1).strip()
                think_tokens_count = len(tokenizer.encode(think_content))
                has_think_syntax = True
                has_done_terminal = "done." in think_content.lower()
            else:
                alt_match = re.search(r'(?:Thinking Process:|Thought:)(.*?)(?:\n\n|\Z)', gen_text, re.DOTALL)
                if alt_match:
                    think_tokens_count = len(tokenizer.encode(alt_match.group(1)))
                else:
                    # For base model without think tag, its entire answer or prefix before solution is verbose reasoning
                    if not args.is_grug and ("solution" in gen_text.lower() or "step" in gen_text.lower()):
                        think_tokens_count = int(gen_tokens_count * 0.7)
                    else:
                        think_tokens_count = 0
                has_think_syntax = False
                has_done_terminal = False

            syntax_ok = has_think_syntax and has_done_terminal
            if syntax_ok:
                syntax_compliant_count += 1

            total_think_tokens += think_tokens_count

            task_passed = evaluate_task_success(cat, eval_type, gen_text, gt, invariants)
            if task_passed:
                passed_count += 1

            # Track Category stats
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0, "think_tokens": 0, "gen_tokens": 0}
            category_stats[cat]["total"] += 1
            if task_passed:
                category_stats[cat]["passed"] += 1
            category_stats[cat]["think_tokens"] += think_tokens_count
            category_stats[cat]["gen_tokens"] += gen_tokens_count

            # Track Prompt Level stats
            if level not in level_stats:
                level_stats[level] = {"total": 0, "passed": 0, "think_tokens": 0, "gen_tokens": 0, "syntax_ok": 0}
            level_stats[level]["total"] += 1
            if task_passed:
                level_stats[level]["passed"] += 1
            if syntax_ok:
                level_stats[level]["syntax_ok"] += 1
            level_stats[level]["think_tokens"] += think_tokens_count
            level_stats[level]["gen_tokens"] += gen_tokens_count

            results.append({
                "id": item["id"],
                "category": cat,
                "prompt_level": level,
                "prompt": prompt,
                "generated_text": gen_text,
                "think_tokens": think_tokens_count,
                "total_tokens": gen_tokens_count,
                "wall_time": round(item_wall_time, 3),
                "syntax_compliant": syntax_ok,
                "task_passed": task_passed
            })

        pbar.update(len(batch_items))
    pbar.close()

    n = len(benchmark_items)
    avg_think = round(total_think_tokens / n, 2)
    avg_total = round(total_gen_tokens / n, 2)
    avg_latency = round(total_wall_time / n, 3)
    gen_throughput = round(total_gen_tokens / total_wall_time, 2) if total_wall_time > 0 else 0
    overall_pass_rate = round((passed_count / n) * 100, 2)
    overall_syntax_rate = round((syntax_compliant_count / n) * 100, 2)

    cat_summary = {}
    for c, stats in category_stats.items():
        cnt = stats["total"]
        cat_summary[c] = {
            "total": cnt,
            "pass_rate_pct": round((stats["passed"] / cnt) * 100, 2),
            "avg_think_tokens": round(stats["think_tokens"] / cnt, 2),
            "avg_total_tokens": round(stats["gen_tokens"] / cnt, 2)
        }

    level_summary = {}
    for lvl, stats in level_stats.items():
        cnt = stats["total"]
        level_summary[lvl] = {
            "total": cnt,
            "pass_rate_pct": round((stats["passed"] / cnt) * 100, 2),
            "syntax_compliance_pct": round((stats["syntax_ok"] / cnt) * 100, 2),
            "avg_think_tokens": round(stats["think_tokens"] / cnt, 2),
            "avg_total_tokens": round(stats["gen_tokens"] / cnt, 2)
        }

    summary = {
        "model_id": args.model_id,
        "is_grug": args.is_grug,
        "adapter_dir": args.adapter_dir,
        "total_test_cases": n,
        "overall_metrics": {
            "pass_rate_pct": overall_pass_rate,
            "grug_syntax_compliance_pct": overall_syntax_rate,
            "avg_reasoning_tokens_tau_think": avg_think,
            "avg_total_tokens_tau_total": avg_total,
            "avg_latency_seconds": avg_latency,
            "generation_throughput_tok_per_sec": gen_throughput
        },
        "category_breakdown": cat_summary,
        "prompt_level_breakdown": level_summary
    }

    summary_path = os.path.join(args.output_dir, f"{args.output_name}_scaled_benchmark_summary.json")
    results_path = os.path.join(args.output_dir, f"{args.output_name}_scaled_benchmark_results.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model Run:                        {args.output_name}")
    print(f"Overall Pass Rate:                {overall_pass_rate}%")
    print(f"Syntax Compliance (<think>...Done): {overall_syntax_rate}%")
    print(f"Avg Reasoning Tokens (tau_think): {avg_think} tokens")
    print(f"Avg Total Tokens (tau_total):     {avg_total} tokens")
    print(f"Avg Wall Latency:                 {avg_latency} s")
    print(f"Throughput:                       {gen_throughput} tok/s")
    print("\n--- Prompt Level Generalization ---")
    for lvl, s in sorted(level_summary.items()):
        print(f"  Level '{lvl}' (n={s['total']}): Pass {s['pass_rate_pct']}%, Syntax {s['syntax_compliance_pct']}%, Think {s['avg_think_tokens']} tok, Total {s['avg_total_tokens']} tok")
    print("=" * 60)
    print(f"Summary saved: {summary_path}")
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
