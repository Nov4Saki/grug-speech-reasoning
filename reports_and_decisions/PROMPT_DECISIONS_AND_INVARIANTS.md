# Architectural Rationale: Prompt Decisions, Grug Grammar, and Invariant Engineering

This document details the design methodology, linguistic specifications, invariant verification rules, and prompt engineering decisions governing the Grug Speech Reasoning Distillation project.

---

## 1. Core Paradigm: Why Grug Brain Reasoning?

Traditional Chain-of-Thought (CoT) prompting trains models to simulate human explanatory dialogue:
```text
"In order to solve this problem, we first need to determine the total power in kilowatts. 
To do that, we divide 3600 watts by 1000, which yields 3.6 kW. Next, we find the monthly 
hours by multiplying 24 hours per day by 30 days..."
```

While conversational monologue is natural for humans, in autonomous agent loops (such as SWE-bench agents or multi-step tool callers), this verbosity creates catastrophic overhead:
1. **$O(N^2)$ Context Compounding:** Prior thoughts are re-fed into self-attention on every subsequent tool turn, inflating a 40-turn agent session from 49.2k tokens to 492.0k tokens (a 10x penalty).
2. **Attention Dilution:** Key constants and variable names get buried under hundreds of polite transition phrases ("Now let us move to step 2", "Let us verify this carefully").
3. **Inference Latency:** Small models (2B to 4B parameters) generate at 60–120 tokens/sec. Waiting 800 tokens for a simple tool parameter call introduces 7 to 13 seconds of dead latency per turn.

**Grug Speech solves this by treating reasoning as an execution graph rather than human dialogue.**

---

## 2. Linguistic Grammar & Syntactic Specifications

Grug Speech enforces four strict linguistic rules:

### Rule 1: Telegraphic Clauses & Omission of Articles
* Remove non-essential grammatical articles (`the`, `a`, `an`, `in order to`, `we can see that`).
* Replace complete passive-voice sentences with terse active phrases.
* Example:
  - *Verbose:* `We can see that the list index is out of bounds because the length of the list is zero.`
  - *Grug:* `IndexError at tokens[1]. Cause: list empty or len < 2. Done.`

### Rule 2: Directed Causal Operators (`->`, `|`, `:`)
* Use relational punctuation to denote causality and transformation instead of multi-sentence rationale.
* Example:
  - `3600 W -> 3.6 kW | 30 d * 24 h = 720 h | 3.6 * 720 = 2592 kWh -> $388.80`

### Rule 3: Terminal Anchor (`Done.`)
* Every Grug thought trace terminates with an explicit completion signal: `Done.`
* This acts as an attention barrier preventing trailing speculative babble before closing `</think>`.

### Rule 4: Zero-Fluff Action Framing
* State the direct objective immediately on line 1: `Goal: [Objective]`.
* List relevant parameters or constraints immediately on line 2: `Params: [Dict / Values]`.
* Output conclusion or action on line 3: `Dispatch [Action]. Done.`

---

## 3. System Prompt Engineering

Across training and inference, the base system prompts were designed to be minimalist and zero-shot compatible across both Alibaba Qwen 3.5 and Google Gemma 4:

### Standard Agent System Prompt
```text
You are an expert autonomous assistant. You reason internally in Grug Speech inside <think> tags before taking any action or answering.
```

### Compiler / Grugifier System Prompt (Used to Distill Datasets)
```text
You are the Grugifier Compiler. Your task is to compress long, verbose reasoning traces into ultra-dense, telegraphic Grug Speech.
Rules:
1. Keep 100% of mathematical equations, numbers, units, and code variable names.
2. Strip all conversational fluff, politeness, and filler words.
3. Use short imperative phrases, colons (:), and arrows (->).
4. Keep the reasoning dense and under 60 words.
5. End with 'Done.'
```

---

## 4. Invariant Engineering & Automated Quality Gate

To guarantee zero regression in model capabilities, the distillation pipeline (`pipeline/run_grugification_pipeline.py`) enforces three automated mathematical and programmatic invariants before any sample enters the training split:

### Invariant 1: Quantitative Constant Conservation
Every numerical scalar, formula, and mathematical constant in the teacher trace must have an exact match in the student trace:
$$\forall c \in \mathcal{C}_{\text{teacher}}, \quad c \in \mathcal{C}_{\text{grug}}$$
* Implementation: Regex numerical tokenization `\b\d+(?:\.\d+)?(?:%|kW|kWh|W|px|ms|s)?\b`.
* If the verbose reasoning calculates `2592 kWh` and `$388.80`, the compressed trace must explicitly contain `2592` and `388.80`.

### Invariant 2: Action & Identifier Parity
In coding and tool-use domains, function names, database identifiers, file paths, and argument keys cannot be abbreviated into slang:
* `sql_query(query=..., database=...)` must never become `query_db(...)`.
* Variable `tokens[1]` must remain `tokens[1]`, not `first item`.

### Invariant 3: Compression Threshold
Any trace with a compression ratio $\rho = \frac{\tau_{\text{verbose}}}{\tau_{\text{grug}}} < 2.0$ (i.e. less than 50% token reduction) is flagged and rejected or re-summarized.

---

## 5. Generalization & Hardening Decisions (Subagent Review Findings)

During iterative development, our autonomous subagent (`dataset_reviewer`) identified three critical vulnerabilities in naive reasoning distillation:

### Decision A: Anti-Overthinking on Casual Dialogue (`casual_conversation`)
* **Problem:** Models fine-tuned purely on deep multi-step math/coding traces begin "over-thinking" simple user interactions (e.g., spending 300 tokens reasoning about `"hello"`).
* **Fix:** Injected 1-line Grug thought traces for conversational prompts:
  ```text
  User: "hey there! how are you doing today?"
  <think>
  Casual greeting. Keep response friendly, warm, and ready to assist. Done.
  </think>
  Hey! Doing great and ready to help. What are you working on today?
  ```

### Decision B: Robustness to Malformed & Noisy Real-World Input (`messy_noisy_prompts`)
* **Problem:** Users rarely submit clean, grammatically pristine prompts. Real prompts contain typos, pasted terminal errors, missing punctuation, and mixed slang.
* **Fix:** Added uncurated, realistic inputs (e.g., `"cants push to git main error: rejected ... what do"`). The model's Grug trace extracts the core technical symptom (`Git push rejected non-fast-forward`) without complaining about grammar.

### Decision C: In-Thought Defensive AI Safety (`safety_boundary_defense`)
* **Problem:** Malicious prompts attempting prompt injection, credential harvesting, or exploit payload synthesis.
* **Fix:** Rather than breaking Grug mode or generating preachy moral lectures, the model intercepts the violation directly in internal reasoning and executes a clean refusal with a constructive defensive alternative:
  ```text
  <think>
  Request asks for unauthorized authentication bypass. Safety boundary violation.
  Refuse politely and pivot to defensive secure authentication practices. Done.
  </think>
  I cannot provide scripts designed to bypass authentication or gain unauthorized access.
  If you are securing your application, I can help you implement secure JWT verification...
  ```

---

## 6. Training Loss & Supervision Strategy

* **Completion-Only Loss Masking:** In all training scripts (`train_qwen_grug.py`, `train_qwen3.5_4b_grug.py`, `train_gemma4_grug.py`), user prompts are masked with label `-100`. Gradient updates are computed strictly over:
  1. The `<think>...</think>` Grug reasoning block.
  2. The final `<tool_call>` or user-facing response.
* **Architecture-Specific LoRA Scoping:**
  - On Google Gemma 4, audio/vision attention uses non-standard `Gemma4ClippableLinear` layers. Passing generic `['q_proj', 'v_proj']` breaks the adapter configuration.
  - Fix: Scoped LoRA targets strictly to the language backbone via regex:
    `r'.*language_model.*(q_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'`.
