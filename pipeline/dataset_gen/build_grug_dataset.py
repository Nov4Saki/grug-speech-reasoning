import json
import os
import re
import random
from datasets import load_dataset
from transformers import AutoTokenizer

random.seed(42)

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')

SYSTEM_PROMPT_OPTIMIZE = (
    "You are a Grug Speech Optimizer. Convert the verbose speech or reasoning into grug speech—an ultra-terse, "
    "token-compressed internal reasoning style (as seen in GPT-5.6). "
    "Rules:\n"
    "1. Strip all conversational fluff, filler words, polite padding, and redundant grammar.\n"
    "2. Use telegraphic shorthand, short imperative verbs, colons, and arrows (->).\n"
    "3. Keep 100% of facts, numbers, equations, logical steps, constraints, and final conclusions intact.\n"
    "4. Maximum semantic density with minimum tokens."
)

SYSTEM_PROMPT_SOLVE = (
    "You are a reasoning model that reasons in Grug Speech. When given a problem or question, think and answer in "
    "ultra-terse, token-compressed telegraphic shorthand (GPT-5.6 grug style). "
    "Retain every critical step and number, but eliminate all filler words and verbosity. State the final answer clearly."
)

def compress_gsm8k(question, answer_raw):
    # Parse lines
    clean_raw = re.sub(r'<<.*?>>', '', answer_raw)
    final_match = re.search(r'####\s*([^\n]+)', clean_raw)
    final_ans = final_match.group(1).strip() if final_match else ""
    
    # Process reasoning lines
    lines = [l.strip() for l in clean_raw.split('\n') if l.strip() and not l.startswith('####')]
    
    # Convert lines to terse grug statements
    grug_lines = []
    # Identify goal
    # Strip common leading questions
    q_simple = question
    q_match = re.search(r'(?:How many|How much|What is|Calculate|Find)\s+([^?]+)', question, re.IGNORECASE)
    if q_match:
        goal = q_match.group(1).strip()
        # truncate goal if too long
        if len(goal.split()) > 7:
            goal = " ".join(goal.split()[:7])
        grug_lines.append(f"Goal: {goal}.")
    else:
        grug_lines.append(f"Goal: solve target.")
        
    for line in lines:
        # Simplify sentence
        # look for equations like A + B = C
        eq_match = re.findall(r'(\d+(?:\.\d+)?\s*[\+\-\*\/\%]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/\%]\s*\d+(?:\.\d+)?)*\s*=\s*\$?\d+(?:\.\d+)?)', line)
        if eq_match:
            # extract key subject
            words = line.split('=')[0].split()
            keywords = [w for w in words if w.lower() not in ['so', 'there', 'are', 'then', 'she', 'he', 'they', 'it', 'is', 'was', 'the', 'a', 'an', 'in', 'to', 'of', 'for', 'from', 'by', 'that', 'this', 'will', 'would', 'could', 'should', 'since', 'after', 'before'] and not re.match(r'[\d\+\-\*\/\(\)\$]', w)]
            subject = " ".join(keywords[:3]) if keywords else "Step"
            eqs = " | ".join(eq_match)
            grug_lines.append(f"{subject}: {eqs}.")
        else:
            # Terse summary of text line
            condensed = re.sub(r'\b(first|second|next|then|finally|in order to|we need to|we can see that|therefore|as a result|so|it follows that|notice that)\b', '', line, flags=re.IGNORECASE)
            condensed = " ".join(condensed.split())
            if len(condensed.split()) > 10:
                condensed = " ".join(condensed.split()[:10])
            if condensed:
                grug_lines.append(f"{condensed}.")
                
    grug_lines.append("Done.")
    grug_lines.append(f"Answer: {final_ans}")
    return "\n".join(grug_lines)

print("GSM8K converter defined.")
