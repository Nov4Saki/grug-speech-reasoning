import json
import os
import re
import random
from datasets import load_dataset
from transformers import AutoTokenizer

random.seed(42)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')

SYSTEM_PROMPT_OPTIMIZE = (
    "You are a Grug Speech Optimizer. Convert the verbose speech or reasoning into grug speech—an ultra-terse, "
    "token-compressed internal reasoning style (GPT-5.6 grug brain). "
    "Rules:\n"
    "1. Strip all conversational fluff, filler words, polite padding, and redundant grammar.\n"
    "2. Use telegraphic shorthand, short imperative verbs, colons, and arrows (->).\n"
    "3. Keep 100% of facts, numbers, equations, logical steps, constraints, and final conclusions intact.\n"
    "4. Maximum semantic density with minimum tokens."
)

SYSTEM_PROMPT_SOLVE = (
    "You are a reasoning model that reasons in Grug Speech. When given a problem or question, think and answer in "
    "ultra-terse, token-compressed telegraphic shorthand (GPT-5.6 grug style). "
    "Retain every critical step and number, but eliminate all filler words and verbosity. State the final answer clearly."
)

# 1. Curated High-Density Agentic & Systems Reasoning Traces (GPT-5.6 Style)
CURATED_AGENT_LOGIC = [
    {
        "domain": "systems_auth",
        "question": "Implement an Express authentication middleware that verifies JWT Bearer tokens safely.",
        "verbose": (
            "We need to implement an authentication middleware for our Express server. First, we should extract the "
            "Bearer token from the Authorization header of the incoming HTTP request. If the header is missing or does "
            "not start with 'Bearer ', we must immediately return a 401 Unauthorized status with an appropriate error message. "
            "Next, if the token is present, we should verify it using jwt.verify with the environment's secret key. If the token "
            "is invalid or expired, jwt.verify will throw an error, which we must catch and respond with 403 Forbidden. Otherwise, "
            "if verification succeeds, we attach the decoded payload to req.user and call next() to pass control to the subsequent route handler."
        ),
        "grug": (
            "Task: Express auth middleware.\n"
            "Header check: get req.headers.authorization.\n"
            "If missing or no 'Bearer ' prefix -> return 401 Unauthorized.\n"
            "Parse token string.\n"
            "Verify: jwt.verify(token, SECRET).\n"
            "Catch err -> return 403 Forbidden.\n"
            "Valid -> req.user = payload, call next(). Done."
        )
    },
    {
        "domain": "devops_docker",
        "question": "How to safely remove all stopped Docker containers without affecting running containers?",
        "verbose": (
            "The user needs us to find all dangling Docker containers and clean them up without deleting active running services. "
            "We should first run `docker ps -a --filter status=exited -q` to get the list of container IDs that have already stopped. "
            "Next, we verify that this list is not empty before running the removal command, because passing an empty argument to `docker rm` "
            "will produce a syntax error. If the list contains IDs, we pipe them to `docker rm` to free disk space while leaving running containers completely untouched."
        ),
        "grug": (
            "Goal: clean dangling docker containers, keep active safe.\n"
            "Need list stopped: `docker ps -a -q -f status=exited`.\n"
            "Check list non-empty to avoid `docker rm` empty arg err.\n"
            "If IDs found -> `docker rm $(docker ps -a -q -f status=exited)`.\n"
            "Active containers untouched. Done."
        )
    },
    {
        "domain": "performance_leak",
        "question": "Diagnose and fix a steady memory leak in a Node.js microservice caused by request event handlers.",
        "verbose": (
            "We are tasked with debugging a memory leak in a Node.js microservice. Over 24 hours, the RSS memory usage climbs steadily "
            "until the process crashes with an out-of-memory error. We should first inspect the heap snapshot using Chrome DevTools or clinic.js "
            "to see which objects are retaining memory. In the heap snapshot, we notice thousands of unclosed EventEmitter listeners registered "
            "inside an HTTP request callback. Because these listeners reference the request context and are never removed with `removeListener` "
            "or `off`, the garbage collector cannot reclaim them. The fix is to use `.once()` instead of `.on()`, or manually remove the listener "
            "inside the response finish event."
        ),
        "grug": (
            "Bug: Node.js memory leak, RSS climb -> OOM crash.\n"
            "Inspect: take heap snapshot via DevTools / clinic.\n"
            "Finding: unclosed EventEmitter listeners inside request callback.\n"
            "Leak path: listeners retain req context -> GC cannot free.\n"
            "Fix: change `.on()` to `.once()`, or cleanup listener on `res.on('finish')`.\n"
            "Done."
        )
    },
    {
        "domain": "agent_architecture",
        "question": "Should we use an orchestrator with subagents or a single context loop for refactoring a huge repository?",
        "verbose": (
            "We need to choose between an autonomous subagent architecture versus a direct single-context execution loop for our long-horizon "
            "repository refactoring task. In a single-context execution loop, the context window fills up rapidly with file contents, tool outputs, "
            "and intermediate reasoning, leading to attention degradation and higher latency per turn. Conversely, invoking specialized subagents "
            "allows each subagent to run in a fresh, isolated workspace with its own dedicated context. The subagent can report back only the "
            "distilled results or diffs, preventing prompt pollution in the orchestrator. Therefore, the optimal design is an orchestrator agent "
            "delegating specific modular tasks to isolated subagents."
        ),
        "grug": (
            "Arch choice: autonomous subagents vs single context loop for large refactor.\n"
            "Single context: fast context bloat, attention degradation, high latency.\n"
            "Subagent approach: isolated workspaces, fresh context per task, return distilled diff.\n"
            "Orchestrator context stays clean.\n"
            "Decision: orchestrator + modular subagents. Need agent kind maybe open hands direct okay. Done."
        )
    },
    {
        "domain": "database_optimization",
        "question": "Optimize a slow orders table query filtering on customer_id, status, and sorting by created_at DESC.",
        "verbose": (
            "The client reports that an SQL database query `SELECT * FROM orders WHERE customer_id = 452 AND status = 'shipped' ORDER BY created_at DESC;` "
            "is running very slowly, taking over 4 seconds on a table with 10 million rows. When analyzing the query plan using EXPLAIN, we observe a sequential "
            "scan across the entire orders table because there is no composite index covering customer_id, status, and created_at. To optimize this query, we should "
            "create a composite index `CREATE INDEX idx_orders_cust_stat_created ON orders (customer_id, status, created_at DESC);`. Furthermore, instead of "
            "using `SELECT *`, we should select only the required columns to avoid unnecessary table fetches. After indexing, the query will use an Index Scan "
            "and execute in sub-millisecond time."
        ),
        "grug": (
            "Slow SQL: orders query 4s on 10M rows.\n"
            "EXPLAIN output: Seq Scan on orders, missing index.\n"
            "Fix 1: CREATE INDEX idx_orders_lookup ON orders (customer_id, status, created_at DESC).\n"
            "Fix 2: replace `SELECT *` with explicit columns.\n"
            "Result: Seq Scan -> Index Scan, execution <1ms.\n"
            "Done."
        )
    },
    {
        "domain": "graph_algorithm",
        "question": "Explain how to detect cycles in a directed graph using DFS and vertex states.",
        "verbose": (
            "To detect cycles in a directed graph with V vertices and E edges, we can use depth-first search along with a three-color state tracking system. "
            "Every vertex begins in the WHITE state (unvisited). When we visit a vertex, we mark it GRAY (currently in the recursion call stack). "
            "If during the traversal of outgoing edges we encounter a neighbor that is already marked GRAY, we have found a back-edge, which proves the existence "
            "of a cycle. Once all descendants of a vertex are processed, we mark it BLACK (fully processed). The algorithm runs in O(V + E) time and O(V) space. "
            "If no GRAY vertex is ever revisited, the graph is a Directed Acyclic Graph (DAG)."
        ),
        "grug": (
            "Task: cycle detection in directed graph (V, E).\n"
            "Method: DFS + 3 colors (White, Gray, Black).\n"
            "White = unvisited, Gray = on current stack, Black = done.\n"
            "Traverse: if edge to Gray node -> back-edge found -> cycle exists!\n"
            "Finish node -> mark Black.\n"
            "Time: O(V + E), Space: O(V).\n"
            "Done."
        )
    },
    {
        "domain": "rate_limiting",
        "question": "Design an atomic token bucket rate limiter in Redis.",
        "verbose": (
            "We want to implement a token bucket rate limiter in Redis to restrict users to 100 requests per minute. Instead of maintaining individual "
            "timestamps for each request in a sorted set which consumes high memory, we can store two keys per user: `tokens` and `last_updated`. "
            "When a request arrives, we calculate `elapsed_time = now - last_updated`. We then replenish tokens by adding `elapsed_time * fill_rate`, capped "
            "at 100 tokens. If available tokens are at least 1, we decrement tokens by 1, update `last_updated` to `now`, and allow the request. Otherwise, "
            "we deny the request with HTTP 429 Too Many Requests. To avoid race conditions, this logic must execute atomically inside a Redis Lua script."
        ),
        "grug": (
            "Goal: token bucket rate limiter, 100 req/min via Redis.\n"
            "State per user: `tokens`, `last_updated`.\n"
            "On request: elapsed = now - last_updated.\n"
            "Refill: tokens = min(100, tokens + elapsed * (100/60)).\n"
            "Check: if tokens >= 1 -> tokens -= 1, allow req. Else -> 429 Too Many Requests.\n"
            "Atomicity: run inside Redis Lua script to prevent race.\n"
            "Done."
        )
    },
    {
        "domain": "concurrency",
        "question": "What is the difference between mutexes and semaphores and when to use each?",
        "verbose": (
            "A mutex is a mutual exclusion primitive designed for locking a shared resource by a single thread at a time. The thread that acquires "
            "the lock owns it and must be the one to release it. In contrast, a counting semaphore maintains a counter representing available permits. "
            "Any thread can acquire a permit (decrementing the counter) or release a permit (incrementing the counter), allowing multiple threads up "
            "to a fixed capacity to access the resource concurrently. Semaphores are ideal for signaling and resource pools (like database connection pools), "
            "whereas mutexes are strictly for mutual exclusion of critical sections."
        ),
        "grug": (
            "Mutex vs Semaphore:\n"
            "Mutex = mutual exclusion lock, ownership enforced. Only acquiring thread can unlock. Capacity = 1.\n"
            "Semaphore = integer counter of permits. Any thread can signal/wait. Capacity = N.\n"
            "Use mutex for: protecting single critical section.\n"
            "Use semaphore for: connection pools, thread signaling.\n"
            "Done."
        )
    },
    {
        "domain": "cache_invalidation",
        "question": "How to handle cache invalidation safely under heavy concurrent reads?",
        "verbose": (
            "Under heavy concurrent read traffic, naive cache invalidation by simply deleting the key can cause a cache stampede or thundering herd problem, "
            "where hundreds of incoming queries simultaneously miss the cache and overwhelm the underlying database. To mitigate this, we can adopt the "
            "probabilistic early expiration algorithm (XFetch) or use a mutex lock around cache misses so that only one worker queries the database and "
            "populates the cache while others wait or serve slightly stale data. Another effective approach is updating the cache asynchronously in the background "
            "before expiration."
        ),
        "grug": (
            "Problem: cache invalidation stampede under high concurrent read.\n"
            "Naive delete -> 100s db queries hit at once -> DB down.\n"
            "Solution 1: single-flight / mutex lock on cache miss. 1 query DB, rest wait.\n"
            "Solution 2: probabilistic early refresh (XFetch) or stale-while-revalidate background refresh.\n"
            "Result: DB protected, zero stampede. Done."
        )
    },
    {
        "domain": "consensus",
        "question": "Summarize how Raft achieves consensus during leader election.",
        "verbose": (
            "In the Raft consensus algorithm, nodes can be in Follower, Candidate, or Leader states. If a follower node experiences an election timeout "
            "without hearing heartbeats from a leader, it increments its current term, transitions to candidate state, votes for itself, and broadcasts "
            "RequestVote RPCs to all peers. If the candidate receives votes from a strict majority of nodes (quorum), it becomes the leader and immediately "
            "begins sending AppendEntries heartbeats to suppress new elections. If split votes occur and no candidate gets a majority before a randomized timeout, "
            "a new election term begins."
        ),
        "grug": (
            "Raft leader election summary:\n"
            "Node states: Follower, Candidate, Leader.\n"
            "Trigger: heartbeat timeout -> Follower becomes Candidate.\n"
            "Action: term++, vote self, send RequestVote RPCs.\n"
            "Quorum: receive majority (>50%) votes -> Leader.\n"
            "Heartbeat: immediately send AppendEntries to peers to maintain authority.\n"
            "Split vote -> randomized timeout -> new term. Done."
        )
    }
]

# 2. Convert GSM8K examples
print("Loading GSM8K...")
ds_gsm = load_dataset('openai/gsm8k', 'main', split='train[:120]')

def convert_gsm(q, raw_ans):
    clean = re.sub(r'<<.*?>>', '', raw_ans)
    final_match = re.search(r'####\s*([^\n]+)', clean)
    final_ans = final_match.group(1).strip() if final_match else ""
    lines = [l.strip() for l in clean.split('\n') if l.strip() and not l.startswith('####')]
    
    verbose_speech = f"Question: {q}\n" + " ".join(lines) + f" Therefore, the final answer is {final_ans}."
    
    # Build grug lines
    grug_parts = [f"Goal: solve math problem."]
    for line in lines:
        eqs = re.findall(r'(\d+(?:\.\d+)?\s*[\+\-\*\/\%]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/\%]\s*\d+(?:\.\d+)?)*\s*=\s*\$?\d+(?:\.\d+)?)', line)
        if eqs:
            # find concise label
            words = line.split('=')[0].split()
            valid_w = [w for w in words if w.lower() not in ['so', 'there', 'are', 'then', 'she', 'he', 'they', 'it', 'is', 'was', 'the', 'a', 'an', 'in', 'to', 'of', 'for', 'from', 'by', 'that', 'this', 'will', 'since', 'after', 'before', 'total'] and not re.search(r'[\d\+\-\*\/\(\)\$]', w)]
            label = " ".join(valid_w[:2]) if valid_w else "Calc"
            grug_parts.append(f"{label}: {' | '.join(eqs)}.")
        else:
            simp = re.sub(r'\b(first|second|next|then|finally|in order to|we need to|we can see that|therefore|as a result|so|it follows that|notice that)\b', '', line, flags=re.IGNORECASE)
            simp = " ".join(simp.split()[:8])
            if simp:
                grug_parts.append(f"{simp}.")
                
    grug_parts.append("Done.")
    grug_parts.append(f"Answer: {final_ans}")
    grug_speech = "\n".join(grug_parts)
    return verbose_speech, grug_speech, final_ans

# 3. Convert ARC examples
print("Loading ARC-Challenge...")
ds_arc = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train[:80]')

def convert_arc(item):
    q = item['question']
    labels = item['choices']['label']
    texts = item['choices']['text']
    key = item['answerKey']
    
    choices_str = ", ".join([f"({l}) {t}" for l, t in zip(labels, texts)])
    correct_text = ""
    for l, t in zip(labels, texts):
        if l == key:
            correct_text = t
            break
            
    verbose = (
        f"The question asks: {q} The possible choices are {choices_str}. "
        f"To solve this, we analyze the underlying physical and scientific principles. "
        f"Option ({key}) '{correct_text}' is scientifically correct because it directly satisfies the causal mechanism "
        f"described by natural physical laws and biological/chemical relationships, whereas the competing choices "
        f"either violate conservation principles, describe unrelated phenomena, or have contrary causal effects. "
        f"Therefore, the correct answer is ({key}) {correct_text}."
    )
    
    grug = (
        f"Query: {q[:70]}...\n"
        f"Key facts: analyze physical/biological mechanism.\n"
        f"Alternative choices fail physical/causal constraints.\n"
        f"Correct match: {correct_text}.\n"
        f"Pick: {key} ({correct_text}). Done."
    )
    return verbose, grug, q, choices_str, key

# Build dataset samples
all_samples = []

# Add Curated Agent / Systems
for item in CURATED_AGENT_LOGIC:
    # 1. Speech Optimization Task
    all_samples.append({
        "task_type": "speech_optimization",
        "instruction": "Optimize the following verbose reasoning or speech into grug speech. Retain all key facts, numbers, logic steps, and conclusions, but eliminate all filler words and verbose syntax.",
        "input": item["verbose"],
        "output": item["grug"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_OPTIMIZE},
            {"role": "user", "content": f"Optimize this verbose reasoning into grug speech:\n\n{item['verbose']}"},
            {"role": "assistant", "content": item["grug"]}
        ]
    })
    # 2. Problem Solving Task
    all_samples.append({
        "task_type": "grug_problem_solving",
        "instruction": "Analyze and solve the following problem in grug speech, providing terse, high-density reasoning and the final answer.",
        "input": item["question"],
        "output": item["grug"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SOLVE},
            {"role": "user", "content": f"Solve in grug speech:\n\n{item['question']}"},
            {"role": "assistant", "content": item["grug"]}
        ]
    })

# Add GSM8K samples
for item in ds_gsm:
    q = item['question']
    verbose, grug, ans = convert_gsm(q, item['answer'])
    
    # Optimization
    all_samples.append({
        "task_type": "speech_optimization",
        "instruction": "Optimize the following verbose reasoning into grug speech. Keep all numbers, arithmetic operations, and the final answer intact while stripping all conversational filler.",
        "input": verbose,
        "output": grug,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_OPTIMIZE},
            {"role": "user", "content": f"Optimize the following verbose mathematical reasoning into grug speech:\n\n{verbose}"},
            {"role": "assistant", "content": grug}
        ]
    })
    # Direct solve
    all_samples.append({
        "task_type": "grug_problem_solving",
        "instruction": "Solve the mathematical problem in grug speech. Use terse telegraphic steps and state the final answer.",
        "input": q,
        "output": grug,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SOLVE},
            {"role": "user", "content": f"Solve this math problem in grug speech:\n\n{q}"},
            {"role": "assistant", "content": grug}
        ]
    })

# Add ARC samples
for item in ds_arc:
    verbose, grug, q, choices, key = convert_arc(item)
    all_samples.append({
        "task_type": "speech_optimization",
        "instruction": "Optimize the following verbose scientific explanation into grug speech. Preserve all scientific facts, causal logic, and the chosen option.",
        "input": verbose,
        "output": grug,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_OPTIMIZE},
            {"role": "user", "content": f"Optimize the following verbose scientific rationale into grug speech:\n\n{verbose}"},
            {"role": "assistant", "content": grug}
        ]
    })
    all_samples.append({
        "task_type": "grug_problem_solving",
        "instruction": "Answer the multiple-choice science question in grug speech. Use terse logic and pick the correct option.",
        "input": f"{q}\nOptions: {choices}",
        "output": grug,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SOLVE},
            {"role": "user", "content": f"Answer in grug speech:\n\n{q}\nOptions: {choices}"},
            {"role": "assistant", "content": grug}
        ]
    })

print(f"Total generated samples: {len(all_samples)}")

# Shuffle and split
random.shuffle(all_samples)
val_count = 30
val_samples = all_samples[:val_count]
train_samples = all_samples[val_count:]

print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

# Token length statistics
in_lens = []
out_lens = []
for s in all_samples:
    in_tok = len(tokenizer.encode(s['input']))
    out_tok = len(tokenizer.encode(s['output']))
    in_lens.append(in_tok)
    out_lens.append(out_tok)

avg_in = sum(in_lens) / len(in_lens)
avg_out = sum(out_lens) / len(out_lens)
compression = avg_in / avg_out if avg_out > 0 else 1.0

print(f"Token Stats: Avg Input = {avg_in:.1f} tokens, Avg Grug Output = {avg_out:.1f} tokens")
print(f"Average Compression Ratio: {compression:.2f}x ({(1 - avg_out/avg_in)*100:.1f}% token reduction)")

# Save to drive and local
out_dir = "/content/drive/MyDrive/grug_speech_dataset"
os.makedirs(out_dir, exist_ok=True)

train_path = os.path.join(out_dir, "train.jsonl")
val_path = os.path.join(out_dir, "val.jsonl")
info_path = os.path.join(out_dir, "dataset_info.json")

with open(train_path, "w") as f:
    for s in train_samples:
        f.write(json.dumps(s) + "\n")

with open(val_path, "w") as f:
    for s in val_samples:
        f.write(json.dumps(s) + "\n")

dataset_info = {
    "dataset_name": "GPT-5.6-Grug-Speech-Reasoning-Subset",
    "total_samples": len(all_samples),
    "train_samples": len(train_samples),
    "val_samples": len(val_samples),
    "sources": ["openai/gsm8k", "allenai/ai2_arc", "curated_agentic_systems"],
    "avg_input_tokens": avg_in,
    "avg_output_tokens": avg_out,
    "compression_ratio": f"{compression:.2f}x",
    "token_reduction_percent": f"{(1 - avg_out/avg_in)*100:.1f}%"
}

with open(info_path, "w") as f:
    json.dump(dataset_info, f, indent=2)

print("Dataset successfully written to:", out_dir)
print("Files:", os.listdir(out_dir))
