import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen3.5-4B"
adapter_dir = "/content/drive/MyDrive/qwen3.5-4b-grug-native/final_adapter"

print("1. Loading Tokenizer & Model...")
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

SYSTEM_PROMPT = (
    "You are Qwen 3.5. You reason internally in Grug Speech—an ultra-terse, high-density telegraphic shorthand "
    "(GPT-5.6 grug brain) enclosed in <think> tags. Keep 100% of facts, code, and math intact without filler words."
)

test_prompts = [
    {
        "field": "Coding & Bug Triage",
        "prompt": "We are getting `IndexError: list index out of range` on line 12 when parsing `tokens[1]`. Explain the root cause and provide the fix."
    },
    {
        "field": "Tool Use & Parameter Calling",
        "prompt": "Call the appropriate tool: 'Query the Postgres database for the top 5 customers with the highest total spend in 2026.'\nAvailable Tool: `sql_query(query: str, database: str)`"
    },
    {
        "field": "Roleplay (Grug Brained Architect)",
        "prompt": "A developer asks: 'Should we replace our 5-line Python script with a Kubernetes CronJob running a custom Docker container with OpenTelemetry sidecars?'"
    },
    {
        "field": "Language & Dense Summarization",
        "prompt": "Summarize this in 2 dense bullets:\n'The mission to Mars launched on Tuesday morning after a 48-hour delay caused by high atmospheric winds. The probe carries spectrometer instruments designed to detect methane emissions in the southern hemisphere crater basins over a 3-year observation window.'"
    },
    {
        "field": "Math & Quantitative Reasoning",
        "prompt": "A server rack consumes 3,600 watts continuously. If electricity costs $0.15 per kilowatt-hour, what is the total cost to run the rack for a 30-day month?"
    }
]

print("\n--- RUNNING MULTI-FIELD INFERENCE ---")
for t in test_prompts:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": t["prompt"]}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.1
        )
    gen = out[0][inputs.input_ids.shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    
    print(f"\n=======================================================")
    print(f"FIELD: {t['field']}")
    print(f"PROMPT: {t['prompt']}")
    print(f"-------------------------------------------------------")
    print(f"GENERATION:\n{text}")
    print(f"=======================================================")
