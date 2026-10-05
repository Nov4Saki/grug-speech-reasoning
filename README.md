# Grug Speech Reasoning Distillation Pipeline & Cross-Model Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Novasaki-orange)](https://huggingface.co/Novasaki)
[![Models: GGUF](https://img.shields.io/badge/Quantization-Q8__0%20%7C%20Q4__K__M-blue)](https://huggingface.co/Novasaki)

An end-to-end framework to compress long-form Chain-of-Thought (CoT) reasoning into ultra-terse, telegraphic **"Grug Speech"** internal monologue (inspired by OpenAI GPT-5.6 *Sol* / *Terra* reasoning traces). Evaluated across **Alibaba Qwen 3.5** and **Google Gemma 4** model families.

---

## 🚀 Key Highlights & Architectural Motivation

* **10.0x Reduction in Multi-Turn Agent Context ($O(N^2)$ Penalty):**  
  In iterative tool-calling trajectories (e.g. 40 turns), standard verbose reasoning forces **492,000 cumulative input tokens** through self-attention. Grug Speech slashes this to **49,200 tokens**, eliminating context bloat and needle-in-a-haystack attention drift.
* **100% Invariant Conservation:**  
  Automated quality gates verify that mathematical equations, variable names, database identifiers, and causal conclusions remain 100% intact.
* **Multi-Field Hierarchical Dataset & Generalization Hardening:**  
  Curated across 22 sub-sections in 5 domains (Language, Coding, Tool Use, Roleplay, Math/Logic) and hardened against casual conversational over-thinking, typos/noisy input, and defensive AI safety boundaries.
* **Cross-Family Deployment-Ready GGUF:**  
  Merged in `bfloat16` and compiled to **`Q8_0`** and **`Q4_K_M`** for instant local execution via Ollama and `llama.cpp`.

---

## 📦 Model Zoo & Hugging Face Hub Repositories

| Model | Architecture | GGUF Binaries Available | Min VRAM | Hugging Face Repository |
| :--- | :--- | :--- | :--- | :--- |
| **MiniCPM 5 2B Grug Native** | 42 Layers (Llama Causal LM) | `MiniCPM5-2B-GrugSpeech-Q4_K_M.gguf` (1.5 GB)<br>`MiniCPM5-2B-GrugSpeech-Q8_0.gguf` (2.5 GB) | ~2.1 GB | [`Novasaki/MiniCPM5-2B-GrugSpeech-Native`](https://huggingface.co/Novasaki/MiniCPM5-2B-GrugSpeech-Native) |
| **Qwen 3.5 4B Grug (Standard)** | 32 Layers (Causal LM) | `Qwen3.5-4B-GrugSpeech-Q4_K_M.gguf` (2.6 GB)<br>`Qwen3.5-4B-GrugSpeech-Q8_0.gguf` (4.2 GB) | ~3.4 GB | [`Novasaki/Qwen3.5-4B-GrugSpeech-Native`](https://huggingface.co/Novasaki/Qwen3.5-4B-GrugSpeech-Native) |
| **Qwen 3.5 4B Grug (Native MTP)** | 33 Layers (MTP NextN) | `Qwen3.5-4B-GrugSpeech-MTP-Q4_K_M.gguf` (2.64 GB)<br>`Qwen3.5-4B-GrugSpeech-MTP-Q8_0.gguf` (4.61 GB) | ~3.6 GB | [`Novasaki/Qwen3.5-4B-GrugSpeech-Native`](https://huggingface.co/Novasaki/Qwen3.5-4B-GrugSpeech-Native) |
| **Qwen 3.5 2B Grug** | 24 Layers (Causal LM) | `Qwen3.5-2B-GrugSpeech-Q4_K_M.gguf` (1.2 GB)<br>`Qwen3.5-2B-GrugSpeech-Q8_0.gguf` (1.9 GB) | ~1.8 GB | [`Novasaki/Qwen3.5-2B-GrugSpeech-Q8`](https://huggingface.co/Novasaki/Qwen3.5-2B-GrugSpeech-Q8) |
| **Gemma 4 E2B Grug Native** | 35 Layers (Conditional) | `Gemma4-E2B-GrugSpeech-Q4_K_M.gguf` (3.2 GB)<br>`Gemma4-E2B-GrugSpeech-Q8_0.gguf` (4.7 GB) | ~4.0 GB | [`Novasaki/Gemma-4-E2B-GrugSpeech-Native`](https://huggingface.co/Novasaki/Gemma-4-E2B-GrugSpeech-Native) |

**Published Datasets on Hugging Face:**
* Hierarchical Multi-Field Dataset: [`Novasaki/grug-multifield-reasoning-dataset`](https://huggingface.co/datasets/Novasaki/grug-multifield-reasoning-dataset)
* Seed Distillation Dataset: [`Novasaki/grug-speech-reasoning`](https://huggingface.co/datasets/Novasaki/grug-speech-reasoning)

---

## 📊 Benchmark & Compute Reduction Graphs

```text
GRAPH 1: PER-TASK REASONING TOKEN GENERATION (VERBOSE vs. GRUG SPEECH)
Scale: 1 block = ~4 tokens | Legend: [#] Verbose CoT  [*] Grug Speech CoT
--------------------------------------------------------------------------------
OpenBookQA (Bio)   | Verbose (127 tok): ###############################
                   | Grug    ( 51 tok): ************ (-59.8%)
                   |
MMLU (Math)        | Verbose (149 tok): #####################################
                   | Grug    ( 61 tok): *************** (-59.1%)
                   |
Systems Arch       | Verbose (128 tok): ################################
                   | Grug    ( 63 tok): *************** (-50.8%)
                   |
Hermes Tool Use    | Verbose (182 tok): #############################################
                   | Grug    ( 38 tok): ********* (-79.1%)
                   |
Python Debug       | Verbose (140 tok): ###################################
                   | Grug    ( 35 tok): ******** (-75.0%)
                   |
Server Power Math  | Verbose (115 tok): ############################
                   | Grug    ( 39 tok): ********* (-66.1%)

GRAPH 2: AGENT CONTEXT COMPOUNDING OVER 40 TURNS (CUMULATIVE INPUT LOAD)
Turn 10 | Verbose ( 30.0k tok): ###          | Grug ( 3.1k tok): *
Turn 20 | Verbose (120.0k tok): ############ | Grug (12.3k tok): *
Turn 30 | Verbose (270.0k tok): ############ | Grug (27.6k tok): **
Turn 40 | Verbose (492.0k tok): ############ | Grug (49.2k tok): **** (10x Savings)
```

---

## 📁 Repository Directory Structure

```text
.
├── README.md                          <- Main project documentation & benchmark overview
├── LICENSE                            <- MIT License
├── requirements.txt                   <- Python environment dependencies
├── .gitignore                         <- Excludes heavy weights, safetensors, and checkpoints
│
├── pipeline/                          <- End-to-end dataset distillation engines
│   ├── run_grugification_pipeline.py  <- Teacher extraction -> 2B Grugifier -> Invariant filter
│   ├── build_multifield_grug_dataset.py <- 22-subsection hierarchical dataset builder
│   ├── dataset_hardening_engine.py    <- Hardening: casual chat, noisy prompts, safety defense
│   ├── test_pipeline_sample.py        <- Smoke test validation runner
│   └── dataset_gen/                   <- Underlying seed data generation modules
│
├── training/                          <- Cross-family QLoRA fine-tuning scripts
│   ├── train_qwen_grug.py             <- Qwen 3.5 2B trainer
│   ├── train_qwen3.5_4b_grug.py       <- Qwen 3.5 4B native trainer
│   ├── train_gemma4_grug.py           <- Gemma 4 E2B trainer (with clippable linear regex fix)
│   └── train_minicpm5_2b_grug.py      <- MiniCPM5 2B trainer (QLoRA 8-bit, 42 layers)
│
├── quantization_and_export/           <- LoRA weight merge & GGUF compilation
│   ├── merge_and_export_gguf.py       <- bfloat16 adapter merger & llama.cpp export (Q8_0, Q4_K_M)
│   ├── export_gemma4_gguf.py          <- Gemma 4 specific GGUF conversion script
│   └── export_minicpm5_gguf.py        <- MiniCPM5 specific GGUF conversion & quantization script
│
├── evaluation/                        <- Evaluation & validation harnesses
│   ├── evaluate_grug_model.py         <- Benchmark evaluation (GSM8K, ARC, systems)
│   ├── evaluate_qwen3.5_4b_multifield.py <- Qwen multi-field domain validation
│   └── evaluate_gemma4_multifield.py  <- Gemma 4 multi-field domain validation
│
├── datasets/                          <- Curated datasets (metadata & jsonl files only)
│   ├── multifield/                    <- 22-subsection + hardened dataset
│   └── seed/                          <- Initial converted reasoning dataset
│
├── reports_and_decisions/             <- Comprehensive research documentation
│   ├── grug_speech_full_technical_report.txt <- Comprehensive technical report with full ASCII graphs
│   ├── grug_speech_technical_report.md       <- Formatted markdown report with mermaid diagrams
│   ├── grug_benchmark_data_points.json       <- Raw benchmark numbers for graph plotting
│   ├── AUDIT_REPORT.md                       <- Subagent dataset audit report
│   ├── PROMPT_DECISIONS_AND_INVARIANTS.md    <- In-depth prompt design & invariant conservation
│   └── subagents/
│       └── dataset_reviewer_agent.md         <- System prompt & architecture of reviewer subagent
│
└── deployment/                        <- Production deployment configs
    ├── Modelfile.qwen3.5_4b           <- Ollama Modelfile for Qwen 3.5 4B
    ├── Modelfile.gemma4_e2b           <- Ollama Modelfile for Gemma 4 E2B
    ├── Modelfile.minicpm5_2b          <- Ollama Modelfile for MiniCPM5 2B
    └── run_llama_cli.sh               <- Shell script for llama-cli inference
```

---

## ⚡ Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/<your-username>/grug-speech-reasoning.git
cd grug-speech-reasoning
pip install -r requirements.txt
```

### 2. Run Dataset Distillation & Hardening
```bash
# Generate the multi-field dataset across 22 subsections
python pipeline/build_multifield_grug_dataset.py

# Inject casual conversation, noisy input, and safety boundary defenses
python pipeline/dataset_hardening_engine.py
```

### 3. Fine-Tune Models (QLoRA)
```bash
# Fine-tune Qwen 3.5 4B
python training/train_qwen3.5_4b_grug.py

# Fine-tune Google Gemma 4 E2B
python training/train_gemma4_grug.py
```

### 4. Merge Adapters & Compile GGUF
```bash
python quantization_and_export/merge_and_export_gguf.py
```

### 5. Local Execution via llama.cpp or Ollama
```bash
# Run with llama-cli
./deployment/run_llama_cli.sh path/to/Qwen3.5-4B-GrugSpeech-Q4_K_M.gguf

# Run with Ollama
ollama create qwen-grug -f deployment/Modelfile.qwen3.5_4b
ollama run qwen-grug
```

---

## 📜 Prompt Decisions & Invariant Rules

See [`reports_and_decisions/PROMPT_DECISIONS_AND_INVARIANTS.md`](reports_and_decisions/PROMPT_DECISIONS_AND_INVARIANTS.md) for full documentation on:
* **The Grug Grammar:** Telegraphic clauses, relational operators (`->`, `|`, `:`), terminal anchors (`Done.`).
* **Automated Invariant Gates:** Quantitative constant conservation ($\forall c \in \mathcal{C}$), identifier retention, and compression thresholding ($\rho \ge 2.0$).
* **Subagent Audit Findings:** Anti-overthinking rules for casual dialogue, typo resistance, and in-thought safety boundary defense.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
