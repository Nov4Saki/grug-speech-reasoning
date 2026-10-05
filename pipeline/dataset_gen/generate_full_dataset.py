import json
import os
import re
import random
from datasets import load_dataset
from transformers import AutoTokenizer

random.seed(42)

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

# 1. Curated Agentic & Technical GPT-5.6 Grug Traces
AGENT_DATA = [
    {
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
        ),
        "question": "Implement an Express authentication middleware that verifies JWT Bearer tokens safely."
    },
    {
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
        ),
        "question": "How to safely remove all stopped Docker containers without affecting running containers?"
    },
    {
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
        ),
        "question": "Diagnose and fix a steady memory leak in a Node.js microservice caused by request event handlers."
    },
    {
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
        ),
        "question": "Should we use an orchestrator with subagents or a single context loop for refactoring a huge repository?"
    },
    {
        "verbose": (
            "We need to calculate the time complexity of searching an element in a balanced binary search tree versus an unbalanced degenerate "
            "binary search tree. In a balanced binary search tree, the height is logarithmic with respect to the number of nodes n, specifically O(log n). "
            "At each comparison, half of the search space is eliminated. In contrast, in the worst-case scenario where elements are inserted in strictly "
            "sorted order, a standard binary search tree degrades into a linked list with height equal to n. Therefore, search takes O(n) linear time. "
            "Thus, search is O(log n) for balanced and O(n) for unbalanced."
        ),
        "grug": (
            "Search complexity BST:\n"
            "Balanced BST: height = O(log n). Halve search space each step -> O(log n).\n"
            "Degenerate BST: worst case sorted insert -> tree become linked list -> height = O(n).\n"
            "Compare: Balanced = O(log n), Degenerate = O(n).\n"
            "Done."
        ),
        "question": "Compare search time complexity between a balanced BST and a degenerate BST."
    },
    {
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
        ),
        "question": "Optimize a slow orders table query filtering on customer_id, status, and sorting by created_at DESC."
    },
    {
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
        ),
        "question": "Explain how to detect cycles in a directed graph using DFS and vertex states."
    },
    {
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
        ),
        "question": "Design an atomic token bucket rate limiter in Redis."
    }
]

print(f"Loaded {len(AGENT_DATA)} agentic seeds.")
