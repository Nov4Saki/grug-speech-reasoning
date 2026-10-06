---
language:
- en
- multilingual
license: apache-2.0
library_name: peft
base_model: HuggingFaceTB/SmolLM2-1.7B-Instruct
tags:
- reasoning
- grug-speech
- tool-use
- function-calling
- edge-ai
- peft
- lora
pipeline_tag: text-generation
datasets:
- Novasaki/grug-multifield-reasoning-dataset
metrics:
- accuracy
- latency
model-index:
- name: SmolLM2-1.7B-UniversalGrugifier
  results:
  - task:
      type: text-generation
      name: Tool Calling (Glaive AI)
    metrics:
    - name: Glaive AI Accuracy
      type: accuracy
      value: 95.0
    - name: Baseline Normal Accuracy
      type: accuracy
      value: 5.0
---

# 🗿 SmolLM2 1.7B Universal Grugifier (Edge Tool-Use Sensation)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Nov4Saki/grug-speech-reasoning)
[![Technical Report](https://img.shields.io/badge/Technical_Report-1%2C244_Lines-blue)](https://github.com/Nov4Saki/grug-speech-reasoning/blob/main/FULL_TECHNICAL_REPORT_GRUG_SPEECH_REASONING.txt)
[![Base Model](https://img.shields.io/badge/Base_Model-HuggingFaceTB%2FSmolLM2--1.7B--Instruct-purple)](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

**SmolLM2 1.7B Universal Grugifier** delivers the most dramatic empirical breakthrough of the entire research sprint: surging tool-calling reliability from **5.0% to 95.0%** (a **19.0x relative improvement**)!

---

## 🏆 Benchmark Scorecard (100 Tasks Per Domain)

| Benchmark Domain | Native Normal Baseline | Grugified Adapter | Absolute Gain | Relative Surge |
| :--- | :---: | :---: | :---: | :---: |
| **Tool Calling (Glaive AI 100)** | 5.0% | **95.0%** | **+90.0%** | **19.0x Surge** 🚀 |
| **Quantitative Math (GSM8K 100)** | 0.0% | **9.0%** | **+9.0%** | Non-zero logic |
| **Coding (HumanEval 100)** | 8.0% | **11.0%** | **+3.0%** | 1.38x |
| **Overall 300-Item Benchmark** | 4.3% | **38.3%** | **+34.0%** | **8.9x Surge** |

---

## 💡 Why SmolLM2 Surged from 5% to 95% on Tools
Compact models (< 2B) lack the working memory to maintain both conversational self-debate and strict JSON schemas. Under native unconstrained prompting, SmolLM2 outputs rambling natural language, exhausting context before emitting valid JSON.

Grug Speech pins down the exact target invariants:
```text
<think>
Domain: tool_use
Goal: get weather
Invariants: city="Tokyo", unit="celsius"
Action: emit get_current_weather
Done.
</think>
<tool_call>
{"name": "get_current_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}
</tool_call>
```
Result: **95 out of 100 tool tasks passed successfully**.

---

## 🚀 Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
ADAPTER_ID = "Novasaki/SmolLM2-1.7B-UniversalGrugifier"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
```

---

## 📚 Technical Report & Full Citation
* Full Report: [`FULL_TECHNICAL_REPORT_GRUG_SPEECH_REASONING.txt`](https://github.com/Nov4Saki/grug-speech-reasoning/blob/main/FULL_TECHNICAL_REPORT_GRUG_SPEECH_REASONING.txt)
* GitHub: [Nov4Saki/grug-speech-reasoning](https://github.com/Nov4Saki/grug-speech-reasoning)
