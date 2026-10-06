---
language:
- en
- ar
- zh
- multilingual
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- reasoning
- grug-speech
- cognitive-compiler
- tool-use
- function-calling
- peft
- lora
pipeline_tag: text-generation
datasets:
- Novasaki/grug-multifield-reasoning-dataset
metrics:
- accuracy
- latency
- compression-ratio
---

# 🗿 Qwen 2.5 3B Universal Grugifier (Foundational Compiler)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Nov4Saki/grug-speech-reasoning)
[![Technical Report](https://img.shields.io/badge/Technical_Report-1%2C244_Lines-blue)](https://github.com/Nov4Saki/grug-speech-reasoning/blob/main/FULL_TECHNICAL_REPORT_GRUG_SPEECH_REASONING.txt)
[![Base Model](https://img.shields.io/badge/Base_Model-Qwen%2FQwen2.5--3B--Instruct-purple)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

The foundational **Universal Grugifier** adapter for `Qwen/Qwen2.5-3B-Instruct`. Converts unconstrained conversational reasoning into structured, invariant-pinned cognitive plans.

---

## 🏆 Key Performance Metrics

* **GSM8K Arithmetic Accuracy**: **38.0%** (100 problems) with sub-second execution (**0.91s**).
* **Tool Calling (Glaive AI)**: **46.0%** deterministic schema compliance.
* **GGUF Quantization Available**:
  * `Qwen2.5-3B-GrugSpeech-Q4_K_M.gguf` (1.93 GB)
  * `Qwen2.5-3B-GrugSpeech-Q8_0.gguf` (3.29 GB)

---

## 🚀 Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_ID = "Novasaki/Qwen2.5-3B-UniversalGrugifier"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
```

---

## 📚 Technical Report & Codebase
* Full Report: [`FULL_TECHNICAL_REPORT_GRUG_SPEECH_REASONING.txt`](https://github.com/Nov4Saki/grug-speech-reasoning/blob/main/FULL_TECHNICAL_REPORT_GRUG_SPEECH_REASONING.txt)
* GitHub: [Nov4Saki/grug-speech-reasoning](https://github.com/Nov4Saki/grug-speech-reasoning)
