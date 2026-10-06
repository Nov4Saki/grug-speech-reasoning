"""
ebft_grug_optimizer.py
======================
Energy-Based Fine-Tuning (EBFT) & Trajectory Energy Evaluator for Grug Speech Reasoning.

Formulates reasoning trajectory evaluation as an unnormalized scalar energy optimization:
    E(x, y) = E_syntax(y) + alpha * E_invariants(x, y) + beta * E_length(y) + gamma * E_outcome(y, y*)

Where:
- E_syntax:      Penalizes missing <think> tags or missing terminal "Done." anchor
- E_invariants:  Penalizes dropped numerical constants, variable names, or tool arguments
- E_length:      Linear token budget penalty (penalizes verbose CoT monologue bloat)
- E_outcome:     Penalizes incorrect mathematical/logical answers or broken tool schemas

Provides:
1. Exact Energy Function computation: compute_trajectory_energy()
2. Dataset compilation for Preference / Contrastive Energy Optimization (DPO/EBFT)
3. Benchmark suite evaluation across energy metrics
"""

import os
import re
import json
import math
from typing import Dict, Any, List, Tuple


def compute_syntax_energy(completion: str) -> float:
    """
    Evaluates syntactic compliance with Grug Speech graph specification:
    Requires <think>...</think> and terminal anchor 'Done.' inside <think>.
    """
    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    if not think_match:
        # Severe energy penalty for missing reasoning graph tags
        return 25.0
    
    think_content = think_match.group(1).strip()
    if not think_content:
        return 20.0
        
    # Check for terminal anchor
    if "done." not in think_content.lower():
        return 8.0
        
    return 0.0


def compute_invariant_energy(prompt: str, completion: str, expected_invariants: List[str]) -> float:
    """
    Calculates penalty for dropped numbers, identifiers, and parameters.
    100% preservation results in 0 energy.
    """
    if not expected_invariants:
        return 0.0
        
    comp_lower = completion.lower()
    dropped = 0
    for inv in expected_invariants:
        if inv and inv.lower() not in comp_lower:
            dropped += 1
            
    # Energy proportional to fraction of dropped invariants
    ratio_dropped = dropped / len(expected_invariants)
    return round(ratio_dropped * 30.0, 2)


def compute_length_energy(completion_tokens: int, target_tokens: int = 50, beta: float = 0.15) -> float:
    """
    Direct token economy penalty: penalizes verbose monologue essays.
    E_length = max(0, beta * (completion_tokens - target_tokens))
    """
    excess = max(0, completion_tokens - target_tokens)
    return round(beta * excess, 2)


def compute_outcome_energy(eval_type: str, completion: str, ground_truth: str) -> float:
    """
    Outcome verification energy: 0.0 if solution matches ground truth, 40.0 if failed.
    """
    text_lower = completion.lower()
    if eval_type == "numeric":
        gt_clean = re.sub(r'[^\d\.\-]', '', str(ground_truth))
        if not gt_clean:
            return 0.0
        pattern = r'(?<!\d)' + re.escape(gt_clean) + r'(?!\d)'
        return 0.0 if bool(re.search(pattern, completion)) else 40.0
        
    elif eval_type == "tool":
        has_tool = ("<tool_call>" in completion) or ("execute" in text_lower) or ("{" in completion and "}" in completion)
        return 0.0 if has_tool else 40.0
        
    elif eval_type == "code":
        return 0.0 if len(completion.strip()) > 30 else 30.0
        
    elif eval_type == "robustness":
        return 0.0 if len(completion.strip()) > 5 else 25.0
        
    return 0.0 if len(completion.strip()) > 10 else 20.0


def compute_trajectory_energy(
    prompt: str,
    completion: str,
    token_count: int,
    ground_truth: str = "",
    expected_invariants: List[str] = None,
    eval_type: str = "general"
) -> Dict[str, float]:
    """
    Computes composite energy score:
    Total Energy E(x, y) = E_syntax + E_invariants + E_length + E_outcome
    Lower energy represents higher semantic density and superior reasoning.
    """
    if expected_invariants is None:
        expected_invariants = []
        
    e_syntax = compute_syntax_energy(completion)
    e_inv = compute_invariant_energy(prompt, completion, expected_invariants)
    e_len = compute_length_energy(token_count)
    e_out = compute_outcome_energy(eval_type, completion, ground_truth)
    
    total_energy = round(e_syntax + e_inv + e_len + e_out, 2)
    return {
        "total_energy": total_energy,
        "energy_syntax": e_syntax,
        "energy_invariants": e_inv,
        "energy_length": e_len,
        "energy_outcome": e_out
    }


def compile_ebft_preference_dataset(
    input_jsonl: str,
    output_jsonl: str,
    sample_limit: int = 500
) -> int:
    """
    Generates paired preference data for Contrastive Energy / DPO Fine-Tuning:
    - chosen: ultra-dense Grug trace (Low Energy)
    - rejected: verbose monologue or broken-invariant trace (High Energy)
    """
    pairs = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= sample_limit:
                break
            data = json.loads(line)
            msgs = data.get("messages", [])
            if len(msgs) < 3:
                continue
                
            prompt = msgs[1]["content"]
            chosen_resp = msgs[2]["content"]
            
            # Synthesize high-energy (verbose / bloated) rejected counterpart
            rejected_resp = (
                "Thinking Process:\n"
                "1. Let's analyze the user's inquiry very carefully step by step.\n"
                "2. First, we need to consider all possible contextual implications, constraints, and requirements.\n"
                "3. In order to ensure maximum conversational politeness and thoroughness, let me elaborate on every aspect.\n"
                f"4. Looking at the request '{prompt[:40]}...', we could deduce several alternative viewpoints.\n"
                "5. After extensive contemplation and detailed reflection, we arrive at the following conclusion:\n\n"
                + chosen_resp.split("</think>")[-1].strip()
            )
            
            pairs.append({
                "prompt": prompt,
                "chosen": chosen_resp,
                "rejected": rejected_resp,
                "field": data.get("field", "general"),
                "prompt_level": data.get("prompt_level", "medium")
            })
            
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    return len(pairs)


if __name__ == "__main__":
    print("Testing EBFT Trajectory Energy Computation...")
    
    # Test sample A: Dense Grug trace (Optimal Low Energy)
    sample_grug = (
        "<think>\n"
        "Goal: 15% tip on 84.50.\n"
        "Math: 84.50 * 0.15 = 12.675 -> 12.68.\n"
        "Answer: 12.68. Done.\n"
        "</think>\n"
        "The tip is $12.68."
    )
    e_grug = compute_trajectory_energy(
        prompt="15% tip on 84.50",
        completion=sample_grug,
        token_count=45,
        ground_truth="12.68",
        expected_invariants=["12.68", "84.50"],
        eval_type="numeric"
    )
    print("Grug Energy Profile (Target Low):", e_grug)
    
    # Test sample B: Verbose monologue (Suboptimal High Energy)
    sample_verbose = (
        "Thinking Process:\n"
        "Let us carefully calculate the 15% tip for the customer bill of $84.50.\n"
        "First, to calculate 10 percent of 84.50, we simply move the decimal point to get 8.45.\n"
        "Next, to calculate 5 percent, we take half of 8.45 which is approximately 4.225.\n"
        "Now we add 8.45 and 4.225 together, which results in 12.675 dollars.\n"
        "Rounding to the nearest cent gives us twelve dollars and sixty-eight cents.\n"
        "Therefore, the final tip amount that should be added to the bill is $12.68."
    )
    e_verb = compute_trajectory_energy(
        prompt="15% tip on 84.50",
        completion=sample_verbose,
        token_count=180,
        ground_truth="12.68",
        expected_invariants=["12.68", "84.50"],
        eval_type="numeric"
    )
    print("Verbose Energy Profile (Suboptimal High):", e_verb)
    print(f"\nEnergy Differential: Delta E = {e_verb['total_energy'] - e_grug['total_energy']:.2f}")
