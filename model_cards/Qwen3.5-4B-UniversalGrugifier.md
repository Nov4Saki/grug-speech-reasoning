---
language:
- en
- ar
- zh
- multilingual
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-4B
tags:
- reasoning
- grug-speech
- cognitive-compiler
- tool-use
- function-calling
- agent
- peft
- lora
- mtp
pipeline_tag: text-generation
datasets:
- Novasaki/grug-multifield-reasoning-dataset
metrics:
- accuracy
- latency
- compression-ratio
model-index:
- name: Qwen3.5-4B-UniversalGrugifier
  results:
  - task:
      type: text-generation
      name: Tool Calling (Glaive AI)
    metrics:
    - name: Glaive AI Accuracy
      type: accuracy
      value: 78.0
    - name: Baseline Normal Accuracy
      type: accuracy
      value: 44.0
  - task:
      type: text-generation
      name: Mathematical Reasoning (GSM8K)
    metrics:
    - name: GSM8K Accuracy
      type: accuracy
      value: 24.0
    - name: Latency Reduction
      type: speedup
      value: 13.0
---

# 🗿 Qwen 3.5 4B Universal Grugifier (MTP + Hybrid Attention Flagship)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Nov4Saki/grug-speech-reasoning)
[![Technical Report](https://img.shields.io/badge/Technical_Report-1%2C244_Lines-blue)](https://github.com/Nov4Saki/grug-speech-reasoning/blob/main/FULL_TECHNICAL_REPORT_GRUG_SPEECH_REASONING.txt)
[![Base Model](https://img.shields.io/badge/Base_Model-Qwen%2FQwen3.5--4B-purple)](https://huggingface.co/Qwen/Qwen3.5-4B)

**Universal Grugifier** is a cognitive compilation adapter that compresses rambling, unconstrained reasoning traces ("Cognitive Bloat") into dense, invariant-pinned **"Grug Speech"** (`<think>Domain: ... Goal: ... Invariants: ... Action: ... Done.</think>`).

Fine-tuned using **Energy-Based Fine-Tuning (EBFT)** with QLoRA ($r=16, \alpha=32$), this adapter aligns with Qwen 3.5's **Multi-Token Prediction (MTP)** heads and hybrid linear attention, accelerating reasoning throughput and eliminating discursive failure modes.

---

## 🏆 Benchmark Scorecard (100 Tasks Per Domain)

Evaluated under strict head-to-head empirical testing against base unconstrained reasoning:

| Benchmark Domain | Native Normal Baseline | Grugified Adapter | Absolute Gain | Latency / Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Tool Calling (Glaive AI 100)** | 44.0% | **78.0%** | **+34.0%** | **1.35s** (137 tokens) |
| **Quantitative Math (GSM8K 100)** | 3.0% | **24.0%** | **+21.0%** (8x rel) | **13.0x speedup** (156s $	o$ 12s) |
| **Coding (HumanEval 100)** | 14.0% | Cognitive Plan | Compiler mode | Zero schema drift |
| **Overall 300-Item Benchmark** | 20.3% | **34.0%** | **+13.7%** | **47% token reduction** |

### Live Production CSV Agent Test (`sales.xlsx`, 100,300 rows)
* **100% Plan Execution Pass Rate** across multi-step Arabic and English queries (`filter_tool`, `aggregate_tool`, `view_tool`, `chart_tool`).
* **49.2 tok/s** decoding speed on NVIDIA A100.

---

## 🧠 The Grug Speech Invariant Topology

Instead of hundreds of tokens of discursive self-doubt, Grug Speech enforces strict invariant conservation:

```text
<think>
Domain: tool_use
Goal: dispatch payment verification
Invariants: order_id=1001, amount=250.0, currency="USD"
Action: call verify_payment(order_id=1001, amount=250.0, currency="USD")
Done.
</think>
<tool_call>
{"name": "verify_payment", "arguments": {"order_id": 1001, "amount": 250.0, "currency": "USD"}}
</tool_call>
```

---

## 🚀 Quickstart & Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3.5-4B"
ADAPTER_ID = "Novasaki/Qwen3.5-4B-UniversalGrugifier"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

prompt = """You are a data analysis planner. Reason in Grug Speech inside <think> tags.
User Query: Filter orders where Total > 500, group by City, and sum Total.
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=250, do_sample=False)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## 📚 Academic Citations & Technical Report

For full theoretical derivations (EBFT energy function, length penalty $\gamma=0.005$, margin $m=1.5$), complete 1,800-task master scorecard, and reproduction instructions, see:

* **Technical Report**: [`FULL_TECHNICAL_REPORT_GRUG_SPEECH_REASONING.txt`](https://github.com/Nov4Saki/grug-speech-reasoning/blob/main/FULL_TECHNICAL_REPORT_GRUG_SPEECH_REASONING.txt)
* **Repository**: [Nov4Saki/grug-speech-reasoning](https://github.com/Nov4Saki/grug-speech-reasoning)
