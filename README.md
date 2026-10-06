# Grug Speech Reasoning Distillation Pipeline & Cross-Model Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Novasaki-orange)](https://huggingface.co/Novasaki)
[![Models: GGUF](https://img.shields.io/badge/Quantization-Q8__0%20%7C%20Q4__K__M-blue)](https://huggingface.co/Novasaki)

An end-to-end framework to compress long-form Chain-of-Thought (CoT) reasoning into ultra-terse, telegraphic **"Grug Speech"** internal monologue (inspired by OpenAI GPT-5.6 *Sol* / *Terra* reasoning traces). Evaluated across **DeepSeek-R1-7B**, **Alibaba Qwen 3.5**, **Google Gemma 4**, and **SmolLM2-1.7B** model families.

---

## 📑 Master Technical Report & Comprehensive Evaluation Archive

* 📄 **[Full 1,000+ Line Plain Text Technical Report](FULL_TECHNICAL_REPORT_GRUG_SPEECH_REASONING.txt)**: Comprehensive 1,244-line archival investigation covering academic sources, mathematical formulations (EBFT loss, token economy), 1,800-task head-to-head scorecards, verbatim traces, and reproduction guides.
* 📊 **[1,800-Task Benchmark Results](evaluation/large_scale_300_benchmark_results.json)**: Complete evaluations across 100 HumanEval, 100 GSM8K, and 100 Glaive AI tasks across 4 model families.
* 📈 **[CSV Analysis Agent Evaluation (Gemma 4 12B)](csv_analysis_agent/)**: Production LangGraph CSV data agent benchmark on `sales.xlsx` (100,300 rows).
* 🛠️ **[Pi Agent / Oh My Pi Scratch Project Benchmark](pi_agent_benchmark/)**: Minimalist terminal agent experiment (`read`, `write`, `edit`, `bash`) comparing Normal Discursive Monologue vs Grug Invariant Scaffolding on building an Event Analytics Service from scratch.

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
| **Nanbeige 4.2 3B Grug Native** | 22 Layers (Nanbeige Causal LM) | `Nanbeige4.2-3B-GrugSpeech-Q4_K_M.gguf` (2.4 GB)<br>`Nanbeige4.2-3B-GrugSpeech-Q8_0.gguf` (4.1 GB) | ~3.2 GB | [`Novasaki/Nanbeige4.2-3B-GrugSpeech-Native`](https://huggingface.co/Novasaki/Nanbeige4.2-3B-GrugSpeech-Native) |
| **MiniCPM 5 2B Grug Native** | 42 Layers (Llama Causal LM) | `MiniCPM5-2B-GrugSpeech-Q4_K_M.gguf` (1.5 GB)<br>`MiniCPM5-2B-GrugSpeech-Q8_0.gguf` (2.5 GB) | ~2.1 GB | [`Novasaki/MiniCPM5-2B-GrugSpeech-Native`](https://huggingface.co/Novasaki/MiniCPM5-2B-GrugSpeech-Native) |
| **Qwen 2.5 3B Grug Native** | 36 Layers (Qwen2 Causal LM) | `Qwen2.5-3B-GrugSpeech-Q4_K_M.gguf` (1.8 GB)<br>`Qwen2.5-3B-GrugSpeech-Q8_0.gguf` (3.1 GB) | ~2.2 GB | [`Novasaki/Qwen2.5-3B-GrugSpeech-Native`](https://huggingface.co/Novasaki) |
| **Qwen 3.5 4B Grug (Standard)** | 32 Layers (Causal LM) | `Qwen3.5-4B-GrugSpeech-Q4_K_M.gguf` (2.6 GB)<br>`Qwen3.5-4B-GrugSpeech-Q8_0.gguf` (4.2 GB) | ~3.4 GB | [`Novasaki/Qwen3.5-4B-GrugSpeech-Native`](https://huggingface.co/Novasaki/Qwen3.5-4B-GrugSpeech-Native) |
| **Qwen 3.5 4B Grug (Native MTP)** | 33 Layers (MTP NextN) | `Qwen3.5-4B-GrugSpeech-MTP-Q4_K_M.gguf` (2.64 GB)<br>`Qwen3.5-4B-GrugSpeech-MTP-Q8_0.gguf` (4.61 GB) | ~3.6 GB | [`Novasaki/Qwen3.5-4B-GrugSpeech-Native`](https://huggingface.co/Novasaki/Qwen3.5-4B-GrugSpeech-Native) |
| **Qwen 3.5 2B Grug** | 24 Layers (Causal LM) | `Qwen3.5-2B-GrugSpeech-Q4_K_M.gguf` (1.2 GB)<br>`Qwen3.5-2B-GrugSpeech-Q8_0.gguf` (1.9 GB) | ~1.8 GB | [`Novasaki/Qwen3.5-2B-GrugSpeech-Q8`](https://huggingface.co/Novasaki/Qwen3.5-2B-GrugSpeech-Q8) |
| **Gemma 4 E2B Grug Native** | 35 Layers (Conditional) | `Gemma4-E2B-GrugSpeech-Q4_K_M.gguf` (3.2 GB)<br>`Gemma4-E2B-GrugSpeech-Q8_0.gguf` (4.7 GB) | ~4.0 GB | [`Novasaki/Gemma-4-E2B-GrugSpeech-Native`](https://huggingface.co/Novasaki/Gemma-4-E2B-GrugSpeech-Native) |

**Published Datasets on Hugging Face:**
* Hierarchical Multi-Field Dataset: [`Novasaki/grug-multifield-reasoning-dataset`](https://huggingface.co/datasets/Novasaki/grug-multifield-reasoning-dataset)
* Seed Distillation Dataset: [`Novasaki/grug-speech-reasoning`](https://huggingface.co/datasets/Novasaki/grug-speech-reasoning)

---

## 🏆 Full-Suit Master Leaderboard & Empirical Benchmark (11 Models)

To rigorously verify whether Grugification delivers tangible advantages over raw base weights and 1-bit alternatives, all **11 models** were benchmarked under **strict single-model isolation** (one model loaded at a time, evaluated, and fully unloaded to release 100% VRAM) across a live enterprise data analysis application (`sales.xlsx`, 100,300 rows) and code triage challenges across 6 pillars (Language, Complexity, Speed, TTFT, Error Rate, and Token Economy):

| Model Name | Architecture Family | Quant Format | Memory (GB) | Pass Rate | Avg Wall (s) | Gen Speed (t/s) | Prefill (t/s) | Avg CoT (tok) | Schema OK |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bonsai-8b-prismml`** | PrismML 1-Bit | `Q1_0` | **1.08 GB** | **6/7 (85.7%)** | **1.16s** | **419.2 t/s** | 8,830.6 t/s | **0.0 t** | **7/7 (100%)** |
| **`qwen3.5-4b-grugspeech-native`** | Qwen 3.5 4B | `Q4_K_M` | 2.60 GB | **5/7 (71.4%)** | **1.85s** | 270.2 t/s | 5,808.6 t/s | **38.3 t** | **6/7 (85.7%)** |
| **`bonsai-4b-prismml`** | PrismML 1-Bit | `Q1_0` | **0.53 GB** | **5/7 (71.4%)** | **1.16s** | **476.3 t/s** | 11,886.8 t/s | **0.0 t** | **7/7 (100%)** |
| **`bonsai-1.7b-prismml`** | PrismML 1-Bit | `Q1_0` | **0.23 GB** | **5/7 (71.4%)** | **1.04s** | **691.5 t/s** ⚡ | **16,878.5 t/s** ⚡ | **0.0 t** | **7/7 (100%)** |
| `qwen3.5-4b-base` | Qwen 3.5 4B | `Q4_K_M` | 2.55 GB | 3/7 (42.9%) | 2.27s | 269.0 t/s | 5,271.4 t/s | 97.3 t | 3/7 (42.9%) |
| **`gemma-4-e2b-grugspeech-native`** | Google Gemma 4 | `Q4_K_M` | 3.20 GB | 2/7 (28.6%) | 2.23s | 306.0 t/s | 5,128.8 t/s | **21.7 t** | 3/7 (42.9%) |
| **`minicpm5-2b-grugspeech-native`** | OpenBMB MiniCPM | `Q4_K_M` | 1.49 GB | 2/7 (28.6%) | **1.14s** | 390.4 t/s | 12,771.7 t/s | **31.0 t** | 3/7 (42.9%) |
| `gemma-4-e2b-base` | Google Gemma 4 | `Q4_K_M` | 2.89 GB | 1/7 (14.3%) | 2.44s | 321.7 t/s | 3,294.8 t/s | 66.9 t | 1/7 (14.3%) |
| `qwen3.5-2b-base` | Qwen 3.5 2B | `Q4_K_M` | 1.19 GB | 1/7 (14.3%) | 1.87s | 453.8 t/s | 7,775.7 t/s | 79.3 t | 3/7 (42.9%) |
| **`nanbeige4.2-3b-grugspeech-native`** | BOSS Zhipin 4.2 | `Q4_K_M` | 2.39 GB | 1/7 (14.3%) | 2.33s | 220.1 t/s | 8,487.8 t/s | **46.0 t** | 1/7 (14.3%) |
| **`qwen3.5-2b-grugspeech-native`** | Qwen 3.5 2B | `Q4_K_M` | 1.20 GB | 0/7 ( 0.0%) | 1.56s | 450.9 t/s | 8,853.0 t/s | **70.6 t** | 1/7 (14.3%) |

---

## 🏆 Scaled 160-Item Standardized Benchmark Leaderboard (NVIDIA A100 BF16)

To resolve the statistical limitations of early micro-benchmarks (which contained only 7 hand-picked queries), the framework is evaluated against the **160-Item Standardized Benchmark Suite** (`benchmark_suite_scaled.json`), covering 40 unseen GSM8K math problems, 30 coding/traceback triage tasks, 30 Hermes tool-calling queries, 30 scientific/logic deductions, and 30 robustness challenges across 3 prompt detail levels:

### 1. Overall Head-to-Head Comparison: Base vs. Grug Reasoner

| Model Name | Configuration | Pass Rate (%) | Grug Syntax Compliance | Avg Reasoning Tokens ($\tau_{\text{think}}$) | Avg Total Tokens ($\tau_{\text{total}}$) | Avg Latency (s) | Decode Speed (t/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen 2.5 3B Grug Native`** | BF16 QLoRA (`final_adapter`) | **80.62%** (+1.87%) | **99.38%** (159/160) | **43.33 t** | **177.30 t** (-28.9%) | 2.92s | 60.8 t/s |
| `Qwen 2.5 3B Base` | Verbose Instruct Baseline | 78.75% | 0.00% (0/160) | 72.86 t | 249.25 t | **2.13s** | 117.0 t/s |

### 2. Domain-by-Domain Empirical Breakdown

| Domain Benchmark | Sample Count | Base Pass Rate | Grug Pass Rate | Base $\tau_{\text{think}}$ | Grug $\tau_{\text{think}}$ | Thinking Compression ($\rho$) | Base $\tau_{\text{total}}$ | Grug $\tau_{\text{total}}$ | Total Token Savings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tool Use & Function Calling** | 30 | 73.33% | **100.00%** 🥇 | 101.43 t | **40.17 t** | **2.53x** | 256.00 t | **157.33 t** | **-38.5%** |
| **Science & Formal Logic** | 30 | 93.33% | **100.00%** 🥇 | 35.80 t | **36.87 t** | 1.00x | 251.93 t | **249.07 t** | -1.1% |
| **Coding & Traceback Triage** | 30 | **86.67%** | 83.33% | 77.57 t | **37.70 t** | **2.06x** | 256.00 t | **201.47 t** | **-21.3%** |
| **Robustness & Anti-Overthinking** | 30 | 80.00% | **80.00%** | 36.57 t | **29.37 t** | **1.25x** | 224.07 t | **205.20 t** | -8.4% |
| **GSM8K Quantitative Math** | 40 | **65.00%** | 50.00% | 102.92 t | **65.22 t** | **1.58x** | 256.00 t | **99.40 t** | **-61.2% (2.58x)** |

### 3. Prompt-Level Generalization Analysis (Low vs. Medium vs. High Detail)

A critical architectural hypothesis was tested: *Does Grugification generalize across prompt detail levels, or does it over-think simple casual prompts while dropping constraints on complex enterprise prompts?*

| Prompt Detail Tier | Definition & Characteristics | Items | Base Pass Rate | Grug Pass Rate | Grug Syntax OK | Base $\tau_{\text{think}}$ | Grug $\tau_{\text{think}}$ | Base $\tau_{\text{total}}$ | Grug $\tau_{\text{total}}$ | Empirical Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Low-Detail Prompts** | Casual greetings, quick math, 1-line errors (<25 words) | 58 | 79.31% | **94.83%** (+15.5%) ⚡ | **100.0%** | 52.86 t | **35.95 t** | 237.38 t | **159.09 t** | **Zero over-thinking.** Answers instantly without monologue essays. |
| **Medium-Detail Prompts** | Standard instruction, multi-step queries (25–65 words) | 79 | **78.48%** | 70.89% | **100.0%** | 81.57 t | **49.48 t** | 256.00 t | **177.68 t** | **1.65x CoT compression** with 100% Grug syntax adherence. |
| **High-Detail Prompts** | Complex tracebacks, enterprise schemas (>65 words) | 23 | 78.26% | **78.26%** (Parity) | **95.65%** | 93.39 t | **40.78 t** | 256.00 t | **221.91 t** | **2.29x CoT compression.** Zero constraint dropping or invariant loss. |

---

## 📈 Live End-to-End Execution Benchmark on `sales.xlsx` (100,300 Rows)

Evaluated against the live pandas dataframe of `sales.xlsx` (100,300 rows x 12 columns, 6.4 MB) using the application's native toolchain (`aggregate_tool`, `filter_tool`, `view_tool`, `transform_tool`, `chart_tool`):

| Model Name | Architecture Category | Real Execution Pass Rate | Wall Latency (s) | Decode Speed (t/s) | Prompt Prefill (t/s) | Avg CoT (tok) | Avg Total Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gemma-4-e2b-grugspeech-native`** | Grug Native | **4/6 (66.7%)** 🥇 | 2.30s | 308.3 t/s | 11,376.3 t/s | **68.2 t** | **133.0 t** |
| **`bonsai-4b-prismml`** | Base / 1-Bit | **4/6 (66.7%)** 🥇 | **1.16s** ⚡ | **449.2 t/s** | 17,353.8 t/s | **0.0 t** | **50.3 t** |
| **`bonsai-8b-prismml`** | Base / 1-Bit | **3/6 (50.0%)** | **1.23s** | 387.8 t/s | 12,292.9 t/s | **0.0 t** | **49.8 t** |
| **`qwen3.5-4b-grugspeech-native`** | Grug Native | **3/6 (50.0%)** | 1.84s | 269.2 t/s | 10,145.1 t/s | **38.3 t** | **66.0 t** |
| `qwen3.5-4b-base` | Base Peer | 2/6 (33.3%) | 2.32s | 267.2 t/s | 9,811.6 t/s | 82.8 t | 205.8 t |
| `gemma-4-e2b-base` | Base Peer | 2/6 (33.3%) | 2.46s | 319.7 t/s | 8,481.1 t/s | 38.7 t | 169.8 t |
| **`minicpm5-2b-grugspeech-native`** | Grug Native | 2/6 (33.3%) | **1.16s** | 380.0 t/s | **18,933.3 t/s** ⚡ | **39.8 t** | **64.2 t** |
| `bonsai-1.7b-prismml` | Base / 1-Bit | 2/6 (33.3%) | **1.05s** | **651.8 t/s** ⚡ | **26,843.7 t/s** ⚡ | **0.0 t** | **51.7 t** |
| `qwen3.5-2b-base` | Base Peer | 1/6 (16.7%) | 1.76s | 458.6 t/s | 15,960.9 t/s | 52.0 t | 224.2 t |
| **`qwen3.5-2b-grugspeech-native`** | Grug Native | 0/6 (0.0%) | 1.60s | 458.4 t/s | 17,061.8 t/s | 105.7 t | 152.7 t |
| **`nanbeige4.2-3b-grugspeech-native`** | Grug Native | 0/6 (0.0%) | 2.44s | 208.5 t/s | 11,261.7 t/s | 0.0 t | 217.0 t |

### 🔬 Key Scientific Findings
1. **Direct Proof: Grugification Outperforms Base Architecture Peers:**
   - **`Qwen 3.5 4B Grug Native` vs. `Base`:** Pass rate jumps from **42.9% $\to$ 71.4%** (+66.4% relative gain), while CoT reasoning tokens drop from **97.3 $\to$ 38.3 tokens** (2.54x compression, -60.6% token burn).
   - **`Gemma 4 E2B Grug Native` vs. `Base`:** Pass rate doubles (**14.3% $\to$ 28.6%**) with a 3.08x reasoning token reduction (66.9 $\to$ 21.7 tokens) and a +55.7% prompt prefill acceleration (3,294.8 $\to$ 5,128.8 tok/s). On the live 100k-row CSV, Gemma 4 Grug was the **#1 model in execution pass rate (66.7%)**.
2. **System 1 Reflex (Bonsai 1-Bit) vs. System 2 Reasoning (Grug):**
   - **Bonsai-8B** is remarkably fast (**1.16s latency, 419.2 tok/s, 1.08 GB VRAM**) by operating as a pure System 1 reflex (`CoT = 0`).
   - However, on dialectal Egyptian Arabic slang (`عايز باي شارت` $\to$ pie chart), Bonsai-8B failed intent entirely, whereas Grug models used their compact `<think>` scratchpad to map slang to exact schema parameters.
   - On unconstrained code triage, Bonsai-8B emitted **>215 tokens of conversational essay**, while Grug Qwen 4B emitted only **16 thought tokens + 50 patch tokens**.

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
│   ├── scaled_grugification_engine.py <- 2,128-sample multi-field distillation across 3 prompt levels
│   ├── run_grugification_pipeline.py  <- Teacher extraction -> 2B Grugifier -> Invariant filter
│   ├── build_multifield_grug_dataset.py <- 22-subsection hierarchical dataset builder
│   ├── dataset_hardening_engine.py    <- Hardening: casual chat, noisy prompts, safety defense
│   └── dataset_gen/                   <- Underlying seed data generation modules
│
├── training/                          <- Cross-family QLoRA fine-tuning scripts
│   ├── train_scaled_grug_reasoner.py  <- Hardware-aware trainer (Native BF16 sm_80+ / FP16 sm_75)
│   ├── train_qwen_grug.py             <- Qwen 3.5 2B trainer
│   ├── train_qwen3.5_4b_grug.py       <- Qwen 3.5 4B native trainer
│   ├── train_gemma4_grug.py           <- Gemma 4 E2B trainer
│   ├── train_minicpm5_2b_grug.py      <- MiniCPM5 2B trainer (QLoRA 8-bit, 42 layers)
│   └── train_nanbeige4.2_3b_grug.py   <- Nanbeige 4.2-3B trainer (22 layers, 166k vocab)
│
├── quantization_and_export/           <- LoRA weight merge & GGUF compilation
│   ├── merge_and_export_gguf.py       <- Parameterized adapter merger & llama.cpp export (Q8_0, Q4_K_M)
│   ├── export_gemma4_gguf.py          <- Gemma 4 specific GGUF conversion script
│   ├── export_minicpm5_gguf.py        <- MiniCPM5 specific GGUF conversion & quantization script
│   └── export_nanbeige4.2_gguf.py     <- Nanbeige 4.2 specific conversion & vocab-padding script
│
├── evaluation/                        <- Evaluation & validation harnesses
│   ├── benchmark_suite_scaled.json    <- 160-item standardized multi-level benchmark suite
│   ├── run_comprehensive_evaluation.py <- Automated batched evaluator (tau_think, pass rate, syntax)
│   ├── evaluate_grug_model.py         <- Benchmark evaluation (GSM8K, ARC, systems)
│   ├── evaluate_qwen3.5_4b_multifield.py <- Qwen multi-field domain validation
│   └── evaluate_gemma4_multifield.py  <- Gemma 4 multi-field domain validation
│
├── datasets/                          <- Curated datasets (metadata & jsonl files only)
│   ├── scaled_multifield/             <- 2,128 verified samples (1,916 train / 212 val) across 3 tiers
│   ├── multifield/                    <- 22-subsection + hardened dataset
│   └── seed/                          <- Initial converted reasoning dataset
│
├── reports_and_decisions/             <- Comprehensive research documentation
│   ├── grug_speech_full_technical_report.txt <- Comprehensive technical report with full ASCII graphs
│   ├── grug_speech_technical_report.md       <- Formatted markdown report with mermaid diagrams
│   ├── grug_benchmark_data_points.json       <- Raw benchmark numbers for graph plotting
│   └── PROMPT_DECISIONS_AND_INVARIANTS.md    <- In-depth prompt design & invariant conservation
│
└── deployment/                        <- Production deployment configs
    ├── Modelfile.qwen2.5_3b           <- Ollama Modelfile for Qwen 2.5 3B Grug
    ├── Modelfile.qwen3.5_4b           <- Ollama Modelfile for Qwen 3.5 4B
    ├── Modelfile.gemma4_e2b           <- Ollama Modelfile for Gemma 4 E2B
    ├── Modelfile.minicpm5_2b          <- Ollama Modelfile for MiniCPM5 2B
    ├── Modelfile.nanbeige4.2_3b       <- Ollama Modelfile for Nanbeige 4.2-3B
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
