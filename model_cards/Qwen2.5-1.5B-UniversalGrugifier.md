---
language:
- en
- ar
- zh
- multilingual
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- reasoning
- grug-speech
- edge-ai
- compact-llm
- peft
- lora
pipeline_tag: text-generation
datasets:
- Novasaki/grug-multifield-reasoning-dataset
---

# 🗿 Qwen 2.5 1.5B Universal Grugifier (Ultra-Compact Edge Compiler)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Nov4Saki/grug-speech-reasoning)
[![Technical Report](https://img.shields.io/badge/Technical_Report-1%2C244_Lines-blue)](https://github.com/Nov4Saki/grug-speech-reasoning/blob/main/FULL_TECHNICAL_REPORT_GRUG_SPEECH_REASONING.txt)

Ultra-compact cognitive compilation adapter designed for constrained edge environments (< 2 GB VRAM). Distills reasoning down to immediate structural invariants, preventing context overflow on low-memory edge devices.

---

## 🚀 Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "Novasaki/Qwen2.5-1.5B-UniversalGrugifier"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
```

* GitHub: [Nov4Saki/grug-speech-reasoning](https://github.com/Nov4Saki/grug-speech-reasoning)
