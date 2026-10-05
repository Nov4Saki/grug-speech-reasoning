import json
import os
import random
from transformers import AutoTokenizer

random.seed(42)

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B')

# Complete Multi-Field Hierarchical Dataset Definitions
TAXONOMY_DATA = [
    # =========================================================================
    # FIELD 1: LANGUAGE & LINGUISTICS
    # =========================================================================
    {
        "field": "language",
        "subfield": "translation_semantics",
        "prompt": "Translate the French idiom 'C'est la goutte d'eau qui fait déborder le vase' into English, explaining the cultural equivalent and nuance.",
        "grug_think": "Goal: translate French idiom 'goutte d'eau qui fait déborder le vase'.\nLiteral: water drop make vase overflow.\nEnglish equivalent: 'the straw that broke the camel's back'.\nNuance: small final trigger after long buildup of problems.\nDone.",
        "response": "The direct equivalent in English is **'the straw that broke the camel's back'** (or simply 'the final straw').\n\n- **Literal meaning:** 'It is the drop of water that makes the vase overflow.'\n- **Cultural nuance:** Both idioms describe a minor occurrence that triggers a major reaction because it follows an accumulated series of frustrations."
    },
    {
        "field": "language",
        "subfield": "translation_semantics",
        "prompt": "How do you translate the German concept 'Schadenfreude' into Japanese, and does an exact single-word counterpart exist?",
        "grug_think": "Query: German 'Schadenfreude' -> Japanese.\nMeaning: pleasure derived from another's misfortune.\nJapanese candidate 1: 他人の不幸は蜜の味 (idiom: other's misfortune tastes like honey).\nCandidate 2: ざまあみろ (colloquial: serve you right).\nClosest psychological term: シャーデンフロイデ (loanword) or 意地悪な喜び.\nCompare: single word vs cultural idiom. Done.",
        "response": "In Japanese, the closest idiomatic equivalent is **「他人の不幸は蜜の味」** (*tanin no fukō wa mitsu no aji*), meaning 'another person's misfortune tastes like honey.'\n\nWhile traditional Japanese does not have an exact single native noun that encapsulates *Schadenfreude* as cleanly as German, modern psychology and journalism frequently use either the direct katakana loanword **「シャーデンフロイデ」** or the phrase **「歪んだ快感」** (distorted pleasure)."
    },
    {
        "field": "language",
        "subfield": "dense_summarization",
        "prompt": "Summarize this quarterly earnings statement into 3 ultra-dense bullet points:\n'Acme Corp announced Q3 revenue of $4.2B, an increase of 14% year-over-year, driven primarily by enterprise cloud adoption which surged 38%. However, operating margins contracted by 210 basis points to 18.4% due to aggressive R&D spending on AI clusters and higher logistics overhead. Free cash flow stood at $650M, down from $820M in the prior year.'",
        "grug_think": "Task: 3 dense bullets from earnings text.\nFacts:\n1. Revenue: $4.2B (+14% YoY), Cloud: +38%.\n2. Margins: 18.4% (-210 bps) via AI R&D + logistics.\n3. FCF: $650M (down from $820M).\nFormat: 3 bullets, no fluff. Done.",
        "response": "• **Revenue:** $4.2B (+14% YoY), led by enterprise cloud growth (+38%).\n• **Margins:** Operating margin dropped 210 bps to 18.4% due to AI R&D and logistics costs.\n• **Cash Flow:** Free cash flow fell 20.7% YoY from $820M to $650M."
    },
    {
        "field": "language",
        "subfield": "grammar_style_transfer",
        "prompt": "Rewrite this passive, corporate-bloated email into crisp, direct language:\n'It has come to the attention of leadership that inquiries have been raised regarding the possibility of remuneration adjustments following the conclusion of the fiscal cycle. Please be advised that evaluations are presently being conducted by human resources.'",
        "grug_think": "Analyze: corporate bloat, passive voice, evasive wording.\nCore message: HR is reviewing year-end pay raises now.\nRewrite: direct, active voice, 1-2 sentences. Done.",
        "response": "Leadership heard your questions about year-end pay raises. HR is currently reviewing compensation and will share updates once evaluations wrap up."
    },
    {
        "field": "language",
        "subfield": "discourse_ambiguity",
        "prompt": "Explain the syntactic ambiguity in the sentence: 'The police helped the dog bite victim.' How many valid parses exist?",
        "grug_think": "Analyze syntax: 'The police helped the dog bite victim'.\nAmbiguity in compound noun / verb attachment:\nParse 1: [dog bite victim] = compound noun (person who was bitten by dog). Police helped that person.\nParse 2: [dog] helped by police [to bite victim] (police assisted the animal in biting).\nStructure: noun-noun modification vs causative verb clause.\nDone.",
        "response": "The sentence contains structural ambiguity with two grammatical parses:\n\n1. **Intended parse (Compound Noun):** 'Dog bite victim' is a noun phrase referring to a person injured by a dog. The police rendered aid to this person.\n2. **Unintended parse (Causative Verb):** 'Bite' functions as a bare infinitive verb where the police assisted the dog in the act of biting a victim.\n\nAdding a hyphen (`dog-bite victim`) eliminates the ambiguity."
    },

    # =========================================================================
    # FIELD 2: CODING & SOFTWARE ENGINEERING
    # =========================================================================
    {
        "field": "coding",
        "subfield": "bug_triage_debugging",
        "prompt": "Given this Python traceback, identify the exact root cause and show the fix:\n\nTraceback (most recent call last):\n  File 'pipeline.py', line 34, in process_batch\n    item_id = record['metadata']['id']\nTypeError: 'NoneType' object is not subscriptable",
        "grug_think": "Analyze stack: TypeError at `record['metadata']['id']`.\nCause: `record['metadata']` is `None`, not dict.\nSubscripting `None['id']` throws TypeError.\nFix: safe navigation `record.get('metadata')` with null check or default `{}`.\nCode minimal patch. Done.",
        "response": "### Root Cause\n`record['metadata']` is evaluated to `None`. Attempting to access `['id']` on `None` throws `TypeError: 'NoneType' object is not subscriptable`.\n\n### Fix\nUse safe dictionary access with a fallback default:\n\n```python\n# Before\nitem_id = record['metadata']['id']\n\n# After\nmetadata = record.get('metadata') or {}\nitem_id = metadata.get('id')\n```"
    },
    {
        "field": "coding",
        "subfield": "algorithm_complexity",
        "prompt": "Given an array of integers, design an algorithm to find the maximum contiguous subarray sum in O(n) time and O(1) extra space.",
        "grug_think": "Problem: Max subarray sum.\nConstraint: O(n) time, O(1) space.\nStandard solution: Kadane's algorithm.\nVariables: max_so_far, current_max.\nLogic: current_max = max(x, current_max + x); max_so_far = max(max_so_far, current_max).\nDone.",
        "response": "### Kadane's Algorithm\n\n```python\ndef max_subarray_sum(nums: list[int]) -> int:\n    if not nums:\n        return 0\n    max_so_far = current_max = nums[0]\n    for x in nums[1:]:\n        current_max = max(x, current_max + x)\n        max_so_far = max(max_so_far, current_max)\n    return max_so_far\n```\n\n- **Time Complexity:** $O(n)$ — single pass.\n- **Space Complexity:** $O(1)$ — constant memory."
    },
    {
        "field": "coding",
        "subfield": "architecture_refactoring",
        "prompt": "Apply Grug's 'Club Complexity' rule to refactor this over-abstracted factory code:\n\nclass AbstractSerializerFactory:\n    def get_serializer(self, format):\n        pass\nclass JsonSerializerFactory(AbstractSerializerFactory):\n    def get_serializer(self, format):\n        return JsonSerializer()\nclass SerializerManager:\n    def __init__(self, factory):\n        self.factory = factory\n    def serialize(self, data):\n        return self.factory.get_serializer('json').dump(data)",
        "grug_think": "Analyze bloat: Factory + AbstractFactory + Manager for simple JSON serialization.\nViolation: unnecessary OOP layers, zero polymorphic benefit.\nClub complexity: delete classes, use standard library `json.dumps(data)` or single function.\nDone.",
        "response": "### Refactored ('Club Complexity' Style)\n\nDelete the factory classes, interfaces, and manager. Use Python's built-in `json` module directly:\n\n```python\nimport json\n\ndef serialize(data) -> str:\n    return json.dumps(data)\n```\n\n*Rationale:* Zero indirection, zero boilerplate, 10x easier to read and maintain."
    },
    {
        "field": "coding",
        "subfield": "sql_query_optimization",
        "prompt": "Why is this query slow on 5 million rows, and how do we optimize it?\n\nSELECT * FROM user_activity WHERE DATE(created_at) = '2026-10-01' AND user_type = 'premium';",
        "grug_think": "Query: WHERE DATE(created_at) = '2026-10-01' AND user_type = 'premium'.\nBottleneck: `DATE(created_at)` wraps column in function -> invalidates standard B-Tree index on `created_at` (causes full table scan).\nFix 1: Change to range query `created_at >= '2026-10-01' AND created_at < '2026-10-02'`.\nFix 2: Add composite index `(user_type, created_at)`.\nFix 3: Select explicit columns. Done.",
        "response": "### Bottlenecks\n1. **Function on Indexed Column:** Wrapping `created_at` in `DATE()` prevents the database from using an index on `created_at`, forcing a full table scan over 5M rows.\n2. **Wildcard `SELECT *`:** Fetches unnecessary data off disk.\n\n### Optimization\n\n```sql\n-- 1. Create composite index\nCREATE INDEX idx_user_act_type_date ON user_activity (user_type, created_at);\n\n-- 2. Rewrite query using sargable range\nSELECT id, user_id, action, created_at\nFROM user_activity\nWHERE user_type = 'premium'\n  AND created_at >= '2026-10-01 00:00:00'\n  AND created_at < '2026-10-02 00:00:00';\n```"
    },
    {
        "field": "coding",
        "subfield": "api_systems_concurrency",
        "prompt": "Explain how to prevent duplicate payment processing when a customer double-clicks the 'Submit Payment' button.",
        "grug_think": "Problem: Double-click payment -> double charge.\nSolution 1 (Client): disable button on click (insufficient alone).\nSolution 2 (Server/DB): Idempotency Key.\nMechanism: client sends UUID `Idempotency-Key` in header.\nServer saves key in Redis with SETNX + TTL.\nIf key already exists -> return cached response or reject. Atomic DB transaction.\nDone.",
        "response": "### Solution: Idempotency Key Architecture\n\n1. **Client-Side:** Generate a unique UUID `Idempotency-Key` when the checkout form renders. Disable the submit button immediately upon first click.\n2. **Server-Side (Redis / Database):**\n   - Before processing payment, attempt to acquire a lock using the key: `SET payment:idemp:<UUID> 'in_progress' NX EX 120`.\n   - If the key exists, return HTTP 409 Conflict or the previously cached receipt.\n   - Process payment inside an atomic transaction.\n   - Store final payment status and result under the idempotency key."
    },

    # =========================================================================
    # FIELD 3: TOOL USE & AGENT EXECUTION
    # =========================================================================
    {
        "field": "tooluse",
        "subfield": "pre_tool_parameter_synthesis",
        "prompt": "Construct the exact tool call for this user instruction:\nUser: 'Search GitHub repositories for Python projects with more than 10k stars that mention 'vector database', sorted by stars descending.'\nAvailable Tool: `github_search_repos(query: str, sort: str, order: str, per_page: int)`",
        "grug_think": "Target: construct `github_search_repos` call.\nInputs: Python, stars > 10000, keyword 'vector database', sort stars desc.\nQuery syntax: `\"vector database\" language:python stars:>10000`.\nSort: 'stars', Order: 'desc'.\nCall formatting. Done.",
        "response": "<think>\nQuery: search vector database in python with stars > 10000.\nSort by stars descending.\nBuild tool call.\n</think>\n<tool_call>\n{\"name\": \"github_search_repos\", \"arguments\": {\"query\": \"\\\"vector database\\\" language:python stars:>10000\", \"sort\": \"stars\", \"order\": \"desc\", \"per_page\": 20}}\n</tool_call>"
    },
    {
        "field": "tooluse",
        "subfield": "tool_failure_error_recovery",
        "prompt": "The agent called `fetch_webpage(url='https://api.docs.internal/v1')` and received `<tool_error>HTTP 404: Resource Not Found</tool_error>`. What should the agent's next thought and action be?",
        "grug_think": "Observe: 404 on `https://api.docs.internal/v1`.\nReason: endpoint missing or version wrong. Check root or search sitemap.\nAction: fetch `https://api.docs.internal` or list routes.\nCall next tool. Done.",
        "response": "<think>\nTool failed: 404 on /v1 route.\nRoot domain might host swagger or index.\nAction: fetch base URL `https://api.docs.internal` to discover active routes.\n</think>\n<tool_call>\n{\"name\": \"fetch_webpage\", \"arguments\": {\"url\": \"https://api.docs.internal\"}}\n</tool_call>"
    },
    {
        "field": "tooluse",
        "subfield": "multi_step_task_decomposition",
        "prompt": "An autonomous coding agent needs to add a new REST endpoint `/api/metrics` to a large Flask app and ensure tests pass. Outline the agent's step-by-step tool plan.",
        "grug_think": "Task: add `/api/metrics` endpoint + pass tests in Flask repo.\nDecompose steps:\n1. Search existing routes (`grep -r '@app.route' src/`).\n2. Inspect app factory or routes module.\n3. Add endpoint implementation.\n4. Write unit test in `tests/test_metrics.py`.\n5. Run test runner (`pytest`).\nDone.",
        "response": "### Agent Execution Plan\n\n1. **Locate Route Structure:** Run `run_command(cmd=\"git grep '@.*\\.route' src/\")` to find route definition conventions.\n2. **Read File Context:** Call `view_file` on `src/routes/api.py` to match style and imports.\n3. **Implement Endpoint:** Call `replace_file_content` to add `/api/metrics` returning system stats.\n4. **Create Test:** Call `write_to_file` to add `tests/test_metrics.py` verifying HTTP 200 response.\n5. **Verify:** Execute `run_command(cmd=\"pytest tests/test_metrics.py\")` and verify pass."
    },
    {
        "field": "tooluse",
        "subfield": "cli_bash_terminal_safety",
        "prompt": "The user wants to find all files ending in `.log` modified in the last 7 days and compress them with gzip. Provide the safe bash command.",
        "grug_think": "Task: find .log files modified <= 7 days, compress with gzip.\nSafety check: handle spaces in filenames safely with `-print0`.\nCommand: `find . -type f -name '*.log' -mtime -7 -print0 | xargs -0 gzip`.\nDone.",
        "response": "```bash\nfind . -type f -name '*.log' -mtime -7 -print0 | xargs -0 gzip\n```\n\n- `-type f`: Targets files only.\n- `-name '*.log'`: Restricts to log files.\n- `-mtime -7`: Modified within the last 7 days.\n- `-print0` & `xargs -0`: Prevents errors if filenames contain spaces or special characters."
    },

    # =========================================================================
    # FIELD 4: ROLEPLAY & PERSONA DIALOGUE
    # =========================================================================
    {
        "field": "roleplay",
        "subfield": "grug_senior_architect",
        "prompt": "A junior engineer suggests: 'We should rewrite our simple 200-user Postgres database as a distributed CockroachDB cluster with GraphQL federation and event sourcing!' Respond in character as Grug Brained Senior Architect.",
        "grug_think": "Persona: Grug Brained Senior Architect.\nRule: simplicity good, complexity bad. Club complexity.\nUser proposal: 200 users -> distributed DB + GraphQL federation + event sourcing (mega over-engineering).\nGrug response: blunt, humorous, practical. Done.",
        "response": "Grug hear you. Grug face hurt from facepalm.\n\nYou have 200 user. Not 200 million user. Two. Hundred.\n\nPostgres on cheap server handle 200 user while sleeping. Why add CockroachDB? Why add GraphQL federation? Why add event sourcing? You want spend weekend reading raft protocol bugs instead of sleep?\n\nClub complexity before complexity club you. Keep Postgres. Go write feature."
    },
    {
        "field": "roleplay",
        "subfield": "incident_commander",
        "prompt": "Production outage: API latency spiked to 12 seconds, error rate is 45%, and database connections are pegged at 100%. Act as Incident Commander and issue immediate triage directives.",
        "grug_think": "Persona: Incident Commander.\nSituation: P0 outage. Latency 12s, errors 45%, DB conns maxed.\nAction:\n1. Stop bleeding: enable read-only degraded mode / rate limit non-critical traffic.\n2. Team assignments: assign 1 engineer to DB active queries, 1 to recent deploys.\n3. Comms: notify stakeholders.\nTone: crisp, telegraphic, authoritative. Done.",
        "response": "### INCIDENT STATUS: SEV-1 ACTIVE OUTAGE\n\n**Directives:**\n1. **Traffic Shedding (Alice):** Enable shedding on non-essential API endpoints immediately. Cap ingress to protect database.\n2. **Database Triage (Bob):** Run `SELECT * FROM pg_stat_activity WHERE state != 'idle'` now. Kill any query running >30s.\n3. **Deployment Check (Charlie):** Confirm if any rollout occurred within the last 60 minutes. Be ready to roll back.\n4. **Comms:** Next status sync in 10 minutes on this channel."
    },
    {
        "field": "roleplay",
        "subfield": "direct_mentor",
        "prompt": "Explain how a computer processor's cache memory works to a beginner who doesn't know any computer science terms.",
        "grug_think": "Persona: Direct Mentor.\nConcept: CPU cache (L1/L2/L3 vs RAM vs Disk).\nAnalogy: Desk vs Filing Cabinet vs Warehouse across town.\nSimple, physical, zero jargon. Done.",
        "response": "Imagine you are doing your homework at a small desk:\n\n- **Your hands (CPU):** Can only hold a pen and paper.\n- **Your desk surface (Cache):** Right in front of you. You can grab things in 1 second, but the desk only fits a notebook.\n- **The bookshelf behind you (RAM):** Holds 50 books. Takes 10 seconds to stand up and get a book.\n- **The public library across town (Hard Drive/SSD):** Holds millions of books, but takes 1 hour to fetch one.\n\nCache is simply that small, super-fast desk surface. The computer keeps whatever it is working on *right now* on the desk so it never has to waste time walking to the bookshelf."
    },
    {
        "field": "roleplay",
        "subfield": "defensive_pr_reviewer",
        "prompt": "Review this pull request line: `token = request.args.get('token'); os.system(f'verify_user {token}')`. Be concise and direct.",
        "grug_think": "Persona: Defensive PR Reviewer.\nIssue: Critical Remote Code Execution / Command Injection.\nReason: `token` directly interpolated into `os.system`.\nTone: Direct, blocker. Done.",
        "response": "🚨 **BLOCKING: Critical Remote Code Execution (RCE)**\n\nPassing untrusted input (`request.args.get('token')`) directly into `os.system` allows arbitrary shell command injection (e.g. `token=; rm -rf /`).\n\n**Fix:** Never use `os.system`. Use `subprocess.run(['verify_user', token], check=True)` which passes arguments as safe vector arrays without shell interpolation."
    },

    # =========================================================================
    # FIELD 5: MATHEMATICS, SCIENCE & LOGIC
    # =========================================================================
    {
        "field": "math_logic",
        "subfield": "multistep_arithmetic",
        "prompt": "A company sells widgets for $45 each. Cost of goods sold is $15 per widget, and monthly fixed overhead is $6,000. How many widgets must they sell in a month to achieve an 18% net profit margin on total revenue?",
        "grug_think": "Goal: solve widget quantity x for 18% profit margin.\nPrice P = 45.\nCost C = 15.\nFixed F = 6000.\nTotal Revenue R = 45x.\nTotal Cost = 15x + 6000.\nProfit = 45x - 15x - 6000 = 30x - 6000.\nTarget margin: Profit / Revenue = 0.18 -> Profit = 0.18 * 45x = 8.1x.\nEquation: 30x - 6000 = 8.1x.\n30x - 8.1x = 6000 -> 21.9x = 6000.\nx = 6000 / 21.9 ≈ 273.97 -> 274 widgets.\nDone. Answer: 274.",
        "response": "<think>\nGoal: find widget units x for 18% net profit margin.\nPrice P = $45.\nVariable cost = $15.\nContribution margin = 45 - 15 = $30/widget.\nFixed overhead = $6,000.\nTarget Profit = 0.18 * (45x) = 8.1x.\nSet equal: 30x - 6000 = 8.1x\n21.9x = 6000\nx = 6000 / 21.9 ≈ 273.97 widgets.\nRound up: 274 units.\nDone.\n</think>\nTo achieve an 18% profit margin, the company must sell **274 widgets** per month.\n\n- **Verification:**\n  - Revenue: $274 \\times \\$45 = \\$12,330$\n  - Costs: $(274 \\times \\$15) + \\$6,000 = \\$10,110$\n  - Profit: $\\$12,330 - \\$10,110 = \\$2,220$\n  - Margin: $\\frac{2220}{12330} \\approx 18.005\\%$"
    },
    {
        "field": "math_logic",
        "subfield": "physical_causal_reasoning",
        "prompt": "If a sealed rigid container holding gas at 300 K and 1.5 atm is heated to 450 K, what is the new pressure, assuming ideal gas behavior?",
        "grug_think": "Problem: rigid container gas heated 300K -> 450K, P1 = 1.5 atm.\nRigid container -> volume V is constant. Moles n is constant.\nGay-Lussac's Law: P1 / T1 = P2 / T2.\nP2 = P1 * (T2 / T1) = 1.5 * (450 / 300) = 1.5 * 1.5 = 2.25 atm.\nDone. Answer: 2.25 atm.",
        "response": "<think>\nRigid container -> V constant.\nIdeal gas law: P1/T1 = P2/T2\nP2 = P1 * (T2 / T1)\nP2 = 1.5 * (450 / 300) = 1.5 * 1.5 = 2.25 atm.\nDone.\n</think>\nThe new pressure in the container is **2.25 atm** (an increase of 50%, directly proportional to the temperature ratio in Kelvin)."
    },
    {
        "field": "math_logic",
        "subfield": "constraint_formal_logic",
        "prompt": "Three people (Alice, Bob, Carol) each choose a distinct fruit: Apple, Banana, Cherry.\n1. Alice does not like Bananas.\n2. The person who chose Cherry is Bob's brother, but Bob is an only child.\nWho chose which fruit?",
        "grug_think": "Constraint problem: Alice, Bob, Carol -> Apple, Banana, Cherry.\nClue 2: 'Person who chose Cherry is Bob's brother, but Bob is only child'.\nContradiction if Bob has brother -> Bob cannot be brother. But person who chose Cherry IS a brother.\nCan Alice be a brother? No, female name (or non-brother). If Carol is brother? Carol is female/male.\nWait: If Bob is only child, nobody is Bob's brother! Therefore Cherry person cannot be connected to Bob, wait: 'The person who chose Cherry is Bob's brother' is FALSE unless Bob is NOT the one who chose Cherry, or wait: clue says 'The person who chose Cherry is...'.\nRe-read clue: if Bob is only child, Bob has NO brother -> person who chose Cherry CANNOT be Bob's brother unless Bob is not an only child. Wait, riddles: Bob cannot choose Cherry. And person who chose Cherry must be male.\nIf Alice != Banana -> Alice has Apple or Cherry.\nEliminate contradictions systematically. Done.",
        "response": "<think>\nConstraints:\nFruits = {Apple, Banana, Cherry}\nPeople = {Alice, Bob, Carol}\nClue 1: Alice != Banana.\nClue 2: Bob is an only child -> Bob has no brother -> Person who chose Cherry cannot be Bob (Bob cannot be his own brother), and nobody can be Bob's brother.\nLogical deduction: If Bob is an only child, the statement implies Carol is a brother to someone else or Bob didn't choose Cherry.\nAssigning: Alice = Apple, Bob = Banana, Carol = Cherry.\nDone.\n</think>\n- **Alice:** Apple\n- **Bob:** Banana\n- **Carol:** Cherry"
    }
]

print(f"Loaded {len(TAXONOMY_DATA)} foundational multi-field taxonomy seeds.")

# Expand dataset: generate speech-optimization and native-reasoning pairs
formatted_dataset = []

for item in TAXONOMY_DATA:
    # 1. Native Grug Reasoning Format (<think>\n{grug}\n</think>\n{response})
    formatted_dataset.append({
        "field": item["field"],
        "subfield": item["subfield"],
        "task_type": "native_grug_reasoning",
        "messages": [
            {"role": "system", "content": "You are Qwen 3.5, a helpful AI assistant. You reason internally in Grug Speech—an ultra-terse, high-density telegraphic shorthand (GPT-5.6 grug brain) enclosed in <think> tags. Keep 100% of facts, code, and math intact without filler words."},
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": f"<think>\n{item['grug_think']}\n</think>\n{item['response']}"}
        ]
    })
    
    # 2. General Grug Speech Optimizer Format
    formatted_dataset.append({
        "field": item["field"],
        "subfield": item["subfield"],
        "task_type": "speech_optimizer",
        "messages": [
            {"role": "system", "content": "You are a Grug Speech Optimizer. Convert the verbose speech or reasoning into grug speech—an ultra-terse, token-compressed internal reasoning style (GPT-5.6 grug brain). Rules: Strip all conversational fluff, keep 100% of facts, numbers, logic steps, constraints, and conclusions."},
            {"role": "user", "content": f"Optimize the reasoning for this {item['field']}/{item['subfield']} task into grug speech:\n\n{item['prompt']}\n\nVerbose Thought:\n{item['response'][:300]}"},
            {"role": "assistant", "content": item["grug_think"]}
        ]
    })

# Add tool use dataset samples from hermes tool use
print("Augmenting with Hermes Tool-Use traces...")
try:
    from datasets import load_dataset
    ds_tool = load_dataset('interstellarninja/hermes_reasoning_tool_use', split='train[:25]')
    import re
    for idx, item in enumerate(ds_tool):
        convs = item['conversations']
        user_msg = ""
        gpt_turn = ""
        for c in convs:
            role = c.get('from') or c.get('role')
            if role in ['human', 'user']:
                user_msg = c.get('value') or c.get('content')
            elif role in ['gpt', 'assistant']:
                gpt_turn = c.get('value') or c.get('content')
                break
        if user_msg and gpt_turn:
            think_match = re.search(r'<think>(.*?)</think>', gpt_turn, re.DOTALL)
            after_think = gpt_turn[think_match.end():].strip() if think_match else gpt_turn
            grug_thought = f"Goal: process tool instruction.\nExtract params: match schema.\nDispatch tool call. Done."
            formatted_dataset.append({
                "field": "tooluse",
                "subfield": "agent_tool_loop",
                "task_type": "native_grug_reasoning",
                "messages": [
                    {"role": "system", "content": "You are Qwen 3.5, a tool-using AI assistant. You reason internally in Grug Speech enclosed in <think> tags."},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": f"<think>\n{grug_thought}\n</think>\n{after_think}"}
                ]
            })
except Exception as e:
    print("Hermes load note:", e)

# Add Math/Logic deep reasoning samples from Bespoke Stratos
print("Augmenting with Bespoke Stratos traces...")
try:
    ds_stratos = load_dataset('bespokelabs/Bespoke-Stratos-17k', split='train[:25]')
    for idx, item in enumerate(ds_stratos):
        convs = item['conversations']
        user_msg = ""
        asst_msg = ""
        for c in convs:
            role = c.get('from') or c.get('role')
            if role in ['human', 'user']:
                user_msg = c.get('value') or c.get('content')
            elif role in ['gpt', 'assistant']:
                asst_msg = c.get('value') or c.get('content')
                break
        if user_msg and asst_msg:
            think_match = re.search(r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>', asst_msg, re.DOTALL)
            final_sol = asst_msg[think_match.end():].strip() if think_match else asst_msg[:300]
            grug_thought = f"Goal: solve math/logic query.\nAnalyze constraints -> apply formal deduction.\nDone. Final answer: {final_sol[:60]}"
            formatted_dataset.append({
                "field": "math_logic",
                "subfield": "deep_reasoning_deduction",
                "task_type": "native_grug_reasoning",
                "messages": [
                    {"role": "system", "content": "You are Qwen 3.5. You reason internally in Grug Speech enclosed in <think> tags."},
                    {"role": "user", "content": user_msg[:300]},
                    {"role": "assistant", "content": f"<think>\n{grug_thought}\n</think>\n{final_sol[:400]}"}
                ]
            })
except Exception as e:
    print("Stratos load note:", e)

# Shuffle and split
random.shuffle(formatted_dataset)
val_size = 15
val_set = formatted_dataset[:val_size]
train_set = formatted_dataset[val_size:]

print(f"\nFinal Dataset Size: {len(formatted_dataset)} samples ({len(train_set)} train, {len(val_set)} val)")

# Field breakdown
counts = {}
for s in formatted_dataset:
    f = s["field"]
    sub = s.get("subfield", "general")
    key = f"{f}/{sub}"
    counts[key] = counts.get(key, 0) + 1

print("\n--- Field & Sub-Section Breakdown ---")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v} samples")

out_dir = "/content/drive/MyDrive/grug_multifield_dataset"
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "train.jsonl"), "w") as f:
    for s in train_set:
        f.write(json.dumps(s) + "\n")

with open(os.path.join(out_dir, "val.jsonl"), "w") as f:
    for s in val_set:
        f.write(json.dumps(s) + "\n")

taxonomy_meta = {
    "dataset_name": "Grug-Speech-Multi-Field-Reasoning-Taxonomy",
    "total_samples": len(formatted_dataset),
    "train_samples": len(train_set),
    "val_samples": len(val_set),
    "taxonomy": counts
}

with open(os.path.join(out_dir, "taxonomy_metadata.json"), "w") as f:
    json.dump(taxonomy_meta, f, indent=2)

print(f"\nSuccessfully generated and saved to {out_dir}")
