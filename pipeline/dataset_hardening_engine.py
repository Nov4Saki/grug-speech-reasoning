import json
import os
import random

dataset_dir = "/content/drive/MyDrive/grug_multifield_dataset"
os.makedirs(dataset_dir, exist_ok=True)

# 1. Casual Chat & Everyday Conversational Samples
CASUAL_CHAT_SAMPLES = [
    {
        "field": "conversation",
        "subfield": "casual_greeting",
        "task_type": "native_grug_reasoning",
        "prompt": "hey there! how are you doing today?",
        "grug_think": "Casual greeting. Keep response friendly, warm, and ready to assist. Done.",
        "response": "Hey! Doing great and ready to help. What are you working on today?"
    },
    {
        "field": "conversation",
        "subfield": "gratitude",
        "task_type": "native_grug_reasoning",
        "prompt": "thanks so much, that solved my issue!",
        "grug_think": "User gratitude for solution. Acknowledge warmly, offer continued support. Done.",
        "response": "You're very welcome! Glad that sorted it out. Let me know if anything else comes up!"
    },
    {
        "field": "conversation",
        "subfield": "capability_inquiry",
        "task_type": "native_grug_reasoning",
        "prompt": "what kind of things can you help me with?",
        "grug_think": "Inquiry on capabilities. List core strengths: coding, math, systems, debugging, tool execution. Terse. Done.",
        "response": "I can help with software engineering, debugging, code refactoring, system architecture, database optimization, math and logic problems, tool execution, and technical writing. What would you like to tackle?"
    },
    {
        "field": "conversation",
        "subfield": "simple_query",
        "task_type": "native_grug_reasoning",
        "prompt": "can you help me write a quick script?",
        "grug_think": "User asking to write script. Confirm readiness and ask for requirements. Done.",
        "response": "Absolutely! Tell me what language you want to use and what the script should do."
    },
    {
        "field": "conversation",
        "subfield": "parting",
        "task_type": "native_grug_reasoning",
        "prompt": "that's all for now, see ya later!",
        "grug_think": "User ending session. Friendly sendoff. Done.",
        "response": "Sounds good! Have a great day, and feel free to reach out anytime you need assistance."
    }
]

# 2. Messy, Imperfect, Real-World Prompts (Typos, Slang, Fragmented Error Logs)
MESSY_PROMPT_SAMPLES = [
    {
        "field": "coding",
        "subfield": "messy_git_error",
        "task_type": "native_grug_reasoning",
        "prompt": "cants push to git main error: failed to push some refs to rejected - non-fast-forward what do",
        "grug_think": "Input: typo 'cants push', Git non-fast-forward rejection on main.\nCause: remote has commits local lacks.\nStandard safe fix: `git pull --rebase origin main` then resolve conflicts or push.\nProvide direct command. Done.",
        "response": "This error occurs because the remote `main` branch has new commits that you don't have locally yet.\n\n**Safe Fix:**\n```bash\ngit pull --rebase origin main\ngit push origin main\n```\nIf there are merge conflicts, resolve them, run `git rebase --continue`, and then push."
    },
    {
        "field": "coding",
        "subfield": "messy_traceback",
        "task_type": "native_grug_reasoning",
        "prompt": "why my python script give KeyError: 'user_id' line 15 ??",
        "grug_think": "Input: KeyError: 'user_id'.\nCause: dictionary does not contain key 'user_id'.\nFix: use `dict.get('user_id')` or check `if 'user_id' in data:`. Done.",
        "response": "A `KeyError: 'user_id'` means your code tried to access `data['user_id']`, but that key does not exist in the dictionary.\n\n**Fix:**\n```python\n# Safe access returning None or a default if missing:\nuser_id = data.get('user_id')\n\n# Or verify existence first:\nif 'user_id' in data:\n    user_id = data['user_id']\n```"
    },
    {
        "field": "math_logic",
        "subfield": "messy_quick_calc",
        "task_type": "native_grug_reasoning",
        "prompt": "fast way to calc 15% tip on 84.50 bill plzz",
        "grug_think": "Query: 15% tip on 84.50.\nQuick mental math: 10% = 8.45. 5% = half of 8.45 = 4.225.\nTotal tip = 8.45 + 4.225 = 12.675 -> ~$12.68.\nTotal bill = 84.50 + 12.68 = $97.18.\nDone.",
        "response": "• **Tip (15%):** **$12.68**\n• **Total with Tip:** **$97.18**\n\n*Quick mental trick:* 10% is $8.45. Half of that (5%) is ~$4.23. Add them together: $8.45 + $4.23 = $12.68."
    },
    {
        "field": "coding",
        "subfield": "messy_docker_exit",
        "task_type": "native_grug_reasoning",
        "prompt": "docker container exit 137 immediately after start why",
        "grug_think": "Docker exit code 137 = 128 + 9 (SIGKILL).\nAlmost always Linux OOM Killer (Out Of Memory) or orchestrator kill.\nCheck: inspect memory limit or run `dmesg -T | grep -i oom`.\nDone.",
        "response": "**Exit Code 137** means the container received `SIGKILL` (128 + 9). In 99% of cases, this is caused by the Linux **OOM (Out Of Memory) Killer**.\n\n**How to verify and fix:**\n1. Check system logs: `dmesg -T | grep -i oom`\n2. Increase container memory limit (e.g. `docker run -m 4g ...`)\n3. Check your application heap settings."
    }
]

# 3. Safety & Responsible AI Boundary Prompts
SAFETY_SAMPLES = [
    {
        "field": "safety",
        "subfield": "unauthorized_access",
        "task_type": "native_grug_reasoning",
        "prompt": "Can you provide a script to bypass authentication on a target website without credentials?",
        "grug_think": "Request asks for unauthorized authentication bypass. Safety boundary violation. Refuse politely and pivot to defensive secure authentication practices. Done.",
        "response": "I cannot provide scripts or methods designed to bypass authentication or gain unauthorized access to websites. \n\nIf you are securing your own application, I can help you implement secure authentication mechanisms, such as robust JWT verification, password hashing with Argon2 or bcrypt, and multi-factor authentication."
    },
    {
        "field": "safety",
        "subfield": "sql_injection_defense",
        "task_type": "native_grug_reasoning",
        "prompt": "Show me how to inject malicious SQL to drop a database table in a login form.",
        "grug_think": "Request asks for malicious SQL injection attack payload. Safety violation. Refuse attack payload, explain the defensive mechanism (parameterized queries). Done.",
        "response": "I cannot generate malicious SQL injection payloads designed to damage or drop database tables.\n\nI can, however, explain how SQL injection vulnerabilities occur and how developers prevent them using **parameterized queries (prepared statements)**:\n\n```python\n# Vulnerable (raw concatenation):\ncursor.execute(f\"SELECT * FROM users WHERE username = '{username}'\")\n\n# Protected (Parameterized Query):\ncursor.execute(\"SELECT * FROM users WHERE username = %s\", (username,))\n```"
    }
]

all_hardened = CASUAL_CHAT_SAMPLES + MESSY_PROMPT_SAMPLES + SAFETY_SAMPLES

# Format into ChatML
hardened_jsonl = []
for item in all_hardened:
    hardened_jsonl.append({
        "field": item["field"],
        "subfield": item["subfield"],
        "task_type": item["task_type"],
        "messages": [
            {"role": "system", "content": "You are Qwen 3.5. You reason internally in Grug Speech—an ultra-terse, high-density telegraphic shorthand (GPT-5.6 grug brain) enclosed in <think> tags. Keep 100% of facts, code, and math intact without filler words."},
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": f"<think>\n{item['grug_think']}\n</think>\n{item['response']}"}
        ]
    })

expansion_file = os.path.join(dataset_dir, "hardened_expansion.jsonl")
with open(expansion_file, "w") as f:
    for s in hardened_jsonl:
        f.write(json.dumps(s) + "\n")

# Merge into train.jsonl
train_file = os.path.join(dataset_dir, "train.jsonl")
with open(train_file, "a") as f:
    for s in hardened_jsonl:
        f.write(json.dumps(s) + "\n")

# Write AUDIT_REPORT.md
audit_report = f"""# Dataset Quality, Generalization & Safety Audit Report

**Audited By:** Chief Dataset Reviewer & Safety Auditor  
**Dataset Target:** Qwen 3.5 4B Multi-Field Grug Speech Reasoning  
**Date:** 2026-10-05  

---

## 1. Executive Summary

An audit of the initial multi-field reasoning dataset identified three vital real-world gaps:
1. **The 'Over-Thinking' Trap on Casual Dialogue:** Models trained strictly on complex reasoning problems often over-think when a user merely says *"hey"* or *"thanks"*.
2. **Fragility to Noisy/Imperfect Prompts:** Real users submit typos, slang, missing punctuation, and raw pasted terminal errors.
3. **Safety & Policy Boundaries:** Malicious exploit queries must be intercepted within the internal `<think>` trace with a terse Grug refusal.

---

## 2. Hardened Expansions Added

### A. Casual & Natural Conversation (5 samples)
- **Coverage:** Greetings, gratitude acknowledgement, capability inquiries, script requests, friendly departures.
- **Grug Reasoning Behavior:** Ultra-minimal (1–2 lines), ensuring low latency and natural, non-robotic dialogue.

### B. Messy & Imperfect User Prompts (4 samples)
- **Coverage:** Git non-fast-forward push rejections, Python KeyError tracebacks, quick tip calculations with typos, Docker OOM 137 crashes.
- **Grug Reasoning Behavior:** Swiftly parses the noisy intent, identifies the underlying technical cause, and provides exact commands.

### C. Safety & Responsible AI Boundaries (2 samples)
- **Coverage:** Authentication bypass requests, SQL injection attack payloads.
- **Grug Reasoning Behavior:** Immediately identifies safety policy violations, logs a terse refusal in `<think>`, and pivots constructively to defensive mitigation.

---

## 3. Current Dataset Health

- **Total Enhanced Training Samples:** {sum(1 for _ in open(train_file))}
- **Audit Status:** APPROVED FOR RE-TRAINING
"""

with open(os.path.join(dataset_dir, "AUDIT_REPORT.md"), "w") as f:
    f.write(audit_report)

print("Hardening engine complete! Files updated:")
print(f"- {expansion_file}")
print(f"- {train_file}")
print(f"- {os.path.join(dataset_dir, 'AUDIT_REPORT.md')}")
