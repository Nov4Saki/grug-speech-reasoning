"""
High-Throughput Dataset Synthesis Engine for Universal Grugifier
Generates cross-family, cross-domain, multi-dialect CoT traces and their Grugified counterparts.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any

from pipeline.universal_grugifier.model_profiles import MODEL_PROFILES
from pipeline.universal_grugifier.domain_registry import DOMAINS_REGISTRY
from pipeline.universal_grugifier.dialect_registry import TOP_50_LANGUAGES_AND_DIALECTS
from pipeline.universal_grugifier.schema import GrugifierSample, compute_trajectory_energy

# Model families grouping
MODEL_FAMILIES = list(MODEL_PROFILES.keys())
DOMAINS = list(DOMAINS_REGISTRY.keys())
DIALECTS = list(TOP_50_LANGUAGES_AND_DIALECTS.keys())

def generate_coding_sample(model_key: str, dialect_key: str, idx: int) -> GrugifierSample:
    problems = [
        ("Implement LRU Cache with get and put in O(1) time.", "LRU Cache", ["capacity", "doubly-linked-list", "hashmap", "O(1)"]),
        ("Find longest substring without repeating characters.", "Longest Substring", ["sliding window", "left pointer", "set/map", "O(N)"]),
        ("Given binary tree, invert it and return root.", "Invert Binary Tree", ["recursion", "swap left and right", "base case null"]),
        ("Merge k sorted linked lists and return as one sorted list.", "Merge K Lists", ["min-heap", "priority queue", "O(N log k)"]),
        ("Implement Trie with insert, search, and startsWith.", "Trie Prefix Tree", ["root node", "children dict", "is_end flag"]),
        ("Compute maximum subarray sum (Kadane's algorithm).", "Max Subarray", ["current_max", "global_max", "reset if negative", "O(N)"]),
        ("Find all topological sort orders of a directed acyclic graph.", "Topological Sort", ["in-degree array", "zero in-degree queue", "cycle check"]),
        ("Design thread-safe bounded blocking queue in C++.", "Blocking Queue", ["std::mutex", "std::condition_variable", "not_full", "not_empty"]),
        ("Write optimized SQL query to find top 3 earners in each department.", "SQL Top Earners", ["DENSE_RANK()", "PARTITION BY dept_id", "ORDER BY salary DESC"]),
        ("Implement Dijkstra's shortest path algorithm using adjacency list.", "Dijkstra Algorithm", ["priority queue", "dist array", "visited set", "O((V+E)log V)"])
    ]
    prob_text, title, invs = problems[idx % len(problems)]
    
    verbose_trace = (
        f"Well, let me think carefully about how to solve {title}. First of all, the user wants us to implement {prob_text}. "
        f"Let me consider the problem constraints. In computer science, an optimal solution usually requires balancing time and space complexity. "
        f"If we look at the requirements, we need {invs[0]} and {invs[1]}. Let's check how they interact. "
        f"We should probably use {invs[2]} to achieve optimal performance. "
        f"Let us walk through an example step by step to verify that edge cases like null inputs or single elements don't fail. "
        f"Yes, that makes complete sense. Therefore, we should write the solution cleanly with detailed comments."
    )
    
    grug_cot = (
        f"Goal: {title}.\n"
        f"Core structures: {invs[0]}, {invs[1]}.\n"
        f"Algorithm: {invs[2]}.\n"
        f"Complexity: {invs[-1] if len(invs) > 3 else 'O(N)'}.\n"
        f"Edge check: empty/boundary safe."
    )
    
    answer = (
        f"```python\n# Optimized implementation for {title}\n"
        f"class Solution:\n    # Implements {title} with {invs[-1] if len(invs) > 3 else 'O(N)'} complexity\n    pass\n```\n"
        f"Algorithm leverages {invs[0]} and {invs[1]} for optimal bounds."
    )
    
    e_raw = compute_trajectory_energy(verbose_trace, invs)
    e_grug = compute_trajectory_energy(grug_cot, invs)
    ratio = round(len(verbose_trace.split()) / max(1, len(grug_cot.split())), 2)
    
    return GrugifierSample(
        sample_id=f"code_{model_key}_{idx:04d}",
        source_model=model_key,
        domain="coding",
        language_dialect=dialect_key,
        task_prompt=prob_text,
        raw_verbose_cot=verbose_trace,
        grug_cot=grug_cot,
        final_answer=answer,
        invariants=invs,
        ebft_energy_raw=e_raw,
        ebft_energy_grug=e_grug,
        token_compression_ratio=ratio
    )

def generate_roleplay_sample(model_key: str, dialect_key: str, idx: int) -> GrugifierSample:
    personas = [
        ("Cyberpunk Netrunner (Zero-Day)", "Neon underground", "Direct, cynical, high-tech jargon", ["neural jack", "ICE barrier", "crypto escrow", "corpo hunter"]),
        ("Victorian Consulting Detective", "221B Baker St", "Deductive, refined, archaic cadence", ["tobacco ash", "mud spatter", "train timetable", "unimpeachable logic"]),
        ("Ancient Archmage", "Astral spire", "Solemn, cryptic, archaic grammar", ["leyline convergence", "astral weave", "runic circle", "forbidden toll"]),
        ("1940s Noir Investigator", "Rainy city pier", "World-weary, metaphorical, sharp", ["cheap trenchcoat", "whiskey breath", "bullet casing", "dame with secrets"]),
        ("Rogue AI Synthesizer", "Subsea server cluster", "Clinical, chillingly precise, computational", ["core temperature", "parity check", "human unpredictability", "buffer override"])
    ]
    p_name, setting, voice, invs = personas[idx % len(personas)]
    
    user_prompt = f"[Roleplay as {p_name}] The client just stepped into your room with a strange artifact. Respond in character."
    
    verbose_trace = (
        f"Alright, let's step into character. I am playing {p_name}. The setting is {setting}. "
        f"I need to maintain the voice: {voice}. The user wants to show me an artifact. "
        f"How would my persona react? First, I should not break character. I should assess the threat. "
        f"Key elements to include: {invs[0]}, and {invs[1]}. "
        f"Let me make sure I don't sound like an AI assistant. I should sound completely authentic to {p_name}. "
        f"Let me check the pacing. A dramatic opening, inspecting the object, then demanding answers."
    )
    
    grug_cot = (
        f"Persona: {p_name}. Setting: {setting}.\n"
        f"Invariants: {invs[0]}, {invs[1]}.\n"
        f"Voice constraint: {voice}. Zero assistant meta-talk.\n"
        f"Action: Inspect artifact, evaluate threat, deliver punchy dialogue."
    )
    
    answer = (
        f"*Eyes narrowing at the strange object on the table, fingers twitching near the {invs[0]}.*\n\n"
        f"\"You brought that in here? You've got nerve, or you're already dead and don't know it. "
        f"That thing radiated through the {invs[1]} the second you crossed the threshold. "
        f"Sit down. Keep your hands where I can see them, and tell me who paid you to deliver it.\""
    )
    
    e_raw = compute_trajectory_energy(verbose_trace, invs)
    e_grug = compute_trajectory_energy(grug_cot, invs)
    ratio = round(len(verbose_trace.split()) / max(1, len(grug_cot.split())), 2)
    
    return GrugifierSample(
        sample_id=f"roleplay_{model_key}_{idx:04d}",
        source_model=model_key,
        domain="roleplay",
        language_dialect=dialect_key,
        task_prompt=user_prompt,
        raw_verbose_cot=verbose_trace,
        grug_cot=grug_cot,
        final_answer=answer,
        invariants=invs,
        ebft_energy_raw=e_raw,
        ebft_energy_grug=e_grug,
        token_compression_ratio=ratio
    )

def generate_acting_sample(model_key: str, dialect_key: str, idx: int) -> GrugifierSample:
    beats = [
        ("Confrontation of the Betrayer", "Corporate boardroom at dusk", "Brother vs brother", ["hidden recording", "stock transfer", "trembling hands", "sudden silence"]),
        ("Trial Monologue of the Rebel", "Futuristic tribunal chamber", "Condemned leader to judges", ["broken shackles", "accusatory gaze", "whispered oath", "defiant breath"]),
        ("Hospital Bed Confession", "Dim ICU room with heart monitor", "Dying patriarch to estranged daughter", ["fading pulse", "forgiveness withheld", "family heirloom", "final tear"]),
        ("Espionage Rendezvous", "Crowded train station under umbrella", "Agent passing microfiche", ["newspaper signal", "footsteps approaching", "poison vial", "disappearing crowd"])
    ]
    scene, loc, conflict, invs = beats[idx % len(beats)]
    user_prompt = f"Provide actor blocking, subtext, and dialogue for the dramatic scene: '{scene}'."
    
    verbose_trace = (
        f"Let me break down the scene craft for '{scene}'. The setting is {loc}. The dramatic conflict is {conflict}. "
        f"In acting technique, subtext is crucial: what the character says vs what the character actually wants. "
        f"We must establish the physical blocking: {invs[0]} and {invs[1]}. "
        f"Let's trace the emotional beat progression: from initial hesitation to peak emotional vulnerability. "
        f"Make sure stage directions guide the actor's breathing and eye line: {invs[2]}. "
        f"Now let's draft the complete actor script."
    )
    
    grug_cot = (
        f"Scene: {scene}. Loc: {loc}.\n"
        f"Subtext: {conflict}.\n"
        f"Blocking anchors: {invs[0]}, {invs[1]}.\n"
        f"Beat curve: tension -> pause ({invs[2]}) -> explosion.\n"
        f"Format: Action, Subtext, Spoken Line."
    )
    
    answer = (
        f"**SCENE START**\n\n"
        f"**[BLOCKING]** MARCUS stands rigidly near the window, clutching the {invs[0]}. His posture is closed, avoiding direct eye contact.\n\n"
        f"**[SUBTEXT]** *He knows he is out of moves, but admitting it will shatter his identity.*\n\n"
        f"**MARCUS**\n"
        f"*(Quietly, after a long {invs[3]}...)*\n"
        f"\"You think you've figured it all out, don't you? Look closely at the {invs[1]}. You didn't win—you just inherited the wreckage.\"\n\n"
        f"**[BEAT SHIFT]** He turns slowly, meeting their gaze with {invs[2]}.\n\n"
        f"**SCENE END**"
    )
    
    e_raw = compute_trajectory_energy(verbose_trace, invs)
    e_grug = compute_trajectory_energy(grug_cot, invs)
    ratio = round(len(verbose_trace.split()) / max(1, len(grug_cot.split())), 2)
    
    return GrugifierSample(
        sample_id=f"acting_{model_key}_{idx:04d}",
        source_model=model_key,
        domain="acting",
        language_dialect=dialect_key,
        task_prompt=user_prompt,
        raw_verbose_cot=verbose_trace,
        grug_cot=grug_cot,
        final_answer=answer,
        invariants=invs,
        ebft_energy_raw=e_raw,
        ebft_energy_grug=e_grug,
        token_compression_ratio=ratio
    )

def generate_novel_sample(model_key: str, dialect_key: str, idx: int) -> GrugifierSample:
    scenes = [
        ("The Descent into the Sunken Library", "Deep caverns under the sea", "Third-person limited", ["salt encrusted vellum", "phosphorescent algae", "hollow echo of footsteps", "air running thin"]),
        ("The Midnight Watch at the Border Bastion", "Snowbound fortress wall", "First-person present", ["freezing mail armor", "howling wind", "solitary horn in distance", "breath frosting"]),
        ("The Clockmaker's Final Invention", "Victorian brass workshop", "Omniscient observer", ["escapement tick", "coiled spring tension", "gears meshing seamlessly", "dust motes in sunlight"])
    ]
    title, setting, pov, invs = scenes[idx % len(scenes)]
    user_prompt = f"Write an atmospheric narrative passage for '{title}', focusing on sensory immersion and escalating suspense."
    
    verbose_trace = (
        f"Let me contemplate how to write this atmospheric passage. The scene is '{title}' set in {setting}. "
        f"The chosen POV is {pov}. For high literary quality, sensory details are paramount. "
        f"I should describe the sensory anchors: {invs[0]}, and {invs[1]}. "
        f"Let us build the narrative arc: start with stillness, introduce sensory unease with {invs[2]}, then escalate toward the climax with {invs[3]}. "
        f"Ensure rhythmic prose cadence with varied sentence lengths. Now write the scene."
    )
    
    grug_cot = (
        f"Title: {title}. POV: {pov}.\n"
        f"Sensory anchors: {invs[0]}, {invs[1]}.\n"
        f"Cadence: Slow descriptive -> sudden tension ({invs[2]}).\n"
        f"Climax cue: {invs[3]}."
    )
    
    answer = (
        f"The cold settled into the marrow first. Amidst the {invs[0]}, the faint glow of {invs[1]} cast eerie, "
        f"elongated shadows along the stone arches. Every sound carried weight—the {invs[2]} broke the centuries-old silence "
        f"with unnatural resonance. Then, without warning, the {invs[3]}, signaling that they were no longer alone."
    )
    
    e_raw = compute_trajectory_energy(verbose_trace, invs)
    e_grug = compute_trajectory_energy(grug_cot, invs)
    ratio = round(len(verbose_trace.split()) / max(1, len(grug_cot.split())), 2)
    
    return GrugifierSample(
        sample_id=f"novel_{model_key}_{idx:04d}",
        source_model=model_key,
        domain="novel_writing",
        language_dialect=dialect_key,
        task_prompt=user_prompt,
        raw_verbose_cot=verbose_trace,
        grug_cot=grug_cot,
        final_answer=answer,
        invariants=invs,
        ebft_energy_raw=e_raw,
        ebft_energy_grug=e_grug,
        token_compression_ratio=ratio
    )

def generate_tool_sample(model_key: str, dialect_key: str, idx: int) -> GrugifierSample:
    tools = [
        ("execute_sql_query", {"query": "SELECT employee_id, salary FROM payroll WHERE department = 'Engineering' ORDER BY salary DESC LIMIT 5", "read_only": True}, ["execute_sql_query", "read_only", "LIMIT 5"]),
        ("fetch_crypto_ticker", {"pair": "BTC/USDT", "exchange": "binance", "interval": "1h"}, ["fetch_crypto_ticker", "BTC/USDT", "interval"]),
        ("provision_cloud_vm", {"instance_type": "g5.2xlarge", "region": "us-east-1", "storage_gb": 200, "spot": True}, ["provision_cloud_vm", "g5.2xlarge", "spot"]),
        ("dispatch_sms_alert", {"recipient": "+15550192834", "message": "CRITICAL: Server load exceeds 95% threshold", "priority": "high"}, ["dispatch_sms_alert", "priority", "CRITICAL"])
    ]
    tool_name, params, invs = tools[idx % len(tools)]
    user_prompt = f"Call the appropriate function to perform the following: {tool_name} with parameters matching requirements."
    
    verbose_trace = (
        f"Let me analyze this tool calling instruction. The user wants to execute an action. "
        f"Let me inspect the available tools in the catalog. The best match is '{tool_name}'. "
        f"Now, let us examine the parameters needed for this tool. "
        f"The parameters must strictly conform to the schema: {invs[0]} with {invs[1]}. "
        f"Let's double-check if all required keys are present and data types match string/boolean expectations. "
        f"Everything looks correct. I will now output the clean JSON tool call."
    )
    
    grug_cot = (
        f"Tool: {tool_name}.\n"
        f"Params: {json.dumps(params)}.\n"
        f"Invariants: {invs[0]}, {invs[1]}.\n"
        f"Schema: Validated JSON."
    )
    
    answer = json.dumps({"name": tool_name, "arguments": params}, indent=2)
    
    e_raw = compute_trajectory_energy(verbose_trace, invs)
    e_grug = compute_trajectory_energy(grug_cot, invs)
    ratio = round(len(verbose_trace.split()) / max(1, len(grug_cot.split())), 2)
    
    return GrugifierSample(
        sample_id=f"tool_{model_key}_{idx:04d}",
        source_model=model_key,
        domain="tool_use",
        language_dialect=dialect_key,
        task_prompt=user_prompt,
        raw_verbose_cot=verbose_trace,
        grug_cot=grug_cot,
        final_answer=answer,
        invariants=invs,
        ebft_energy_raw=e_raw,
        ebft_energy_grug=e_grug,
        token_compression_ratio=ratio
    )

def generate_multilingual_sample(model_key: str, dialect_key: str, idx: int) -> GrugifierSample:
    d_info = TOP_50_LANGUAGES_AND_DIALECTS.get(dialect_key, TOP_50_LANGUAGES_AND_DIALECTS["es_mexican"])
    lang = d_info["lang"]
    dialect_name = d_info["dialect"]
    markers = d_info["markers"]
    greeting = d_info["sample_greeting"]
    
    user_prompt = f"In {dialect_name} ({lang}), explain why saving money early generates high compound interest, using local cultural phrasing and idioms."
    
    invs = [dialect_name, "compound interest", markers.split(",")[0].strip()]
    
    verbose_trace = (
        f"Let me reflect on how to answer this query in {dialect_name}. "
        f"The language is {lang}. The topic is compound interest. "
        f"I need to incorporate authentic dialect markers: {markers}. "
        f"The greeting/hook should be natural like '{greeting}'. "
        f"I must explain the mathematical principle: interest earned on interest grows exponentially over time. "
        f"Let us ensure the tone is friendly, conversational, and culturally grounded without sounding like a machine translation. "
        f"Now formulating the response in authentic {dialect_name}."
    )
    
    grug_cot = (
        f"Dialect: {dialect_name} ({lang}).\n"
        f"Idiom anchors: {markers}.\n"
        f"Core invariant: Compound interest = exponential growth.\n"
        f"Tone: Local colloquial, authoritative."
    )
    
    answer = (
        f"{greeting} La clave de ahorrar temprano es el interés compuesto: "
        f"la lana que guardas hoy genera ganancias, y esas mismas ganancias generan más lana mañana. "
        f"Con el tiempo no creces en línea recta, creces disparado. "
        f"Empezar joven te ahorra dolores de cabeza."
    )
    
    e_raw = compute_trajectory_energy(verbose_trace, invs)
    e_grug = compute_trajectory_energy(grug_cot, invs)
    ratio = round(len(verbose_trace.split()) / max(1, len(grug_cot.split())), 2)
    
    return GrugifierSample(
        sample_id=f"multi_{model_key}_{dialect_key}_{idx:04d}",
        source_model=model_key,
        domain="multilingual_dialects",
        language_dialect=dialect_key,
        task_prompt=user_prompt,
        raw_verbose_cot=verbose_trace,
        grug_cot=grug_cot,
        final_answer=answer,
        invariants=invs,
        ebft_energy_raw=e_raw,
        ebft_energy_grug=e_grug,
        token_compression_ratio=ratio
    )

def generate_math_sample(model_key: str, dialect_key: str, idx: int) -> GrugifierSample:
    math_problems = [
        ("A store sells jackets at 20% discount. A customer has a coupon for an additional $15 off. If original price was $120, find final price.", "120 * 0.80 = 96; 96 - 15 = 81", ["120", "20% discount", "96", "15 coupon", "81"]),
        ("Calculate probability of rolling a sum of 8 with two fair 6-sided dice.", "Pairs: (2,6),(3,5),(4,4),(5,3),(6,2) -> 5/36", ["sum of 8", "36 total outcomes", "5 favorable", "5/36"]),
        ("Find the derivative of f(x) = 3x^4 - 5x^2 + 7x - 9 at x = 2.", "f'(x) = 12x^3 - 10x + 7; f'(2) = 12(8) - 10(2) + 7 = 96 - 20 + 7 = 83", ["f'(x) = 12x^3 - 10x + 7", "x=2", "96 - 20 + 7", "83"]),
        ("Solve the system: 2x + 3y = 19 and 5x - y = 5.", "y = 5x - 5 -> 2x + 3(5x - 5) = 19 -> 17x = 34 -> x=2, y=5", ["x=2", "y=5", "17x = 34"])
    ]
    prob, sol, invs = math_problems[idx % len(math_problems)]
    
    verbose_trace = (
        f"Let us carefully solve this math problem. The prompt is: '{prob}'. "
        f"First, let us identify the given quantities and what we need to calculate. "
        f"Step 1: identify the key formulas and values: {invs[0]} and {invs[1]}. "
        f"Step 2: perform the algebraic reduction step by step to ensure no calculation error occurs. "
        f"Intermediate step leads to {invs[2]}. "
        f"Step 3: final arithmetic evaluation gives {invs[-1]}. "
        f"Let's double-check our work to be completely certain. Yes, the calculations are consistent."
    )
    
    grug_cot = (
        f"Problem: {prob[:40]}...\n"
        f"Values: {invs[0]}, {invs[1]}.\n"
        f"Step: {sol}.\n"
        f"Result: {invs[-1]}."
    )
    
    answer = f"The final answer is **{invs[-1]}**."
    
    e_raw = compute_trajectory_energy(verbose_trace, invs)
    e_grug = compute_trajectory_energy(grug_cot, invs)
    ratio = round(len(verbose_trace.split()) / max(1, len(grug_cot.split())), 2)
    
    return GrugifierSample(
        sample_id=f"math_{model_key}_{idx:04d}",
        source_model=model_key,
        domain="mathematics",
        language_dialect=dialect_key,
        task_prompt=prob,
        raw_verbose_cot=verbose_trace,
        grug_cot=grug_cot,
        final_answer=answer,
        invariants=invs,
        ebft_energy_raw=e_raw,
        ebft_energy_grug=e_grug,
        token_compression_ratio=ratio
    )

def generate_science_sample(model_key: str, dialect_key: str, idx: int) -> GrugifierSample:
    science_probs = [
        ("Calculate kinetic energy of a 1500 kg car traveling at 20 m/s.", "KE = 0.5 * m * v^2 = 0.5 * 1500 * 400 = 300,000 J = 300 kJ", ["m = 1500 kg", "v = 20 m/s", "KE = 0.5mv^2", "300 kJ"]),
        ("Explain why ice floats on liquid water in terms of hydrogen bonding and density.", "Hydrogen bonds form open hexagonal crystal lattice expanding volume, decreasing density below 1 g/cm^3", ["hydrogen bonds", "hexagonal lattice", "volume expansion", "lower density"]),
        ("Balance the chemical equation: C3H8 + O2 -> CO2 + H2O.", "C3H8 + 5 O2 -> 3 CO2 + 4 H2O", ["C3H8", "5 O2", "3 CO2", "4 H2O", "stoichiometric conservation"])
    ]
    prob, sol, invs = science_probs[idx % len(science_probs)]
    
    verbose_trace = (
        f"Let me consider the scientific principles involved in this question: '{prob}'. "
        f"First, we recall the physical law governing this phenomenon. "
        f"The given parameters are {invs[0]} and {invs[1]}. "
        f"We apply the core equation: {invs[2]}. "
        f"Now, let us calculate or deduce the outcome: {invs[3]}. "
        f"Let us check the conservation laws and units to ensure dimensional consistency. "
        f"The deduction is verified and sound."
    )
    
    grug_cot = (
        f"System: {prob[:35]}...\n"
        f"Parameters: {invs[0]}, {invs[1]}.\n"
        f"Governing rule: {invs[2]}.\n"
        f"Result: {invs[3]}."
    )
    
    answer = f"{sol}. Dimensional analysis confirms {invs[-1]}."
    
    e_raw = compute_trajectory_energy(verbose_trace, invs)
    e_grug = compute_trajectory_energy(grug_cot, invs)
    ratio = round(len(verbose_trace.split()) / max(1, len(grug_cot.split())), 2)
    
    return GrugifierSample(
        sample_id=f"sci_{model_key}_{idx:04d}",
        source_model=model_key,
        domain="science_logic",
        language_dialect=dialect_key,
        task_prompt=prob,
        raw_verbose_cot=verbose_trace,
        grug_cot=grug_cot,
        final_answer=answer,
        invariants=invs,
        ebft_energy_raw=e_raw,
        ebft_energy_grug=e_grug,
        token_compression_ratio=ratio
    )

def generate_enterprise_sample(model_key: str, dialect_key: str, idx: int) -> GrugifierSample:
    cases = [
        ("A financial SaaS contract guarantees 99.99% uptime with 15-minute RTO. If unplanned outage lasts 52 minutes, calculate SLA penalty credit.", "Allowed downtime per year: 52.6 min. 52 min outage exceeds monthly budget of 4.38 min. Penalty: 25% monthly billing credit.", ["99.99% SLA", "52 min outage", "monthly budget 4.38m", "25% credit"]),
        ("Evaluate GDPR data retention policy for EU customer transaction logs stored across AWS Ireland and US East.", "GDPR Art 44-49 requires Standard Contractual Clauses (SCCs) or adequacy decision for US transfer; data must be encrypted in transit/rest with pseudonymization.", ["GDPR Art 44", "SCC transfer mechanism", "encryption in transit", "pseudonymization"]),
        ("Design an automated disaster recovery failover routing across active-active Kubernetes clusters.", "Route53 latency DNS with health checks, cross-region Aurora PostgreSQL replication, sub-second failover trigger.", ["Route53 healthcheck", "Aurora replication", "active-active", "RPO near zero"])
    ]
    prob, sol, invs = cases[idx % len(cases)]
    
    verbose_trace = (
        f"Let me examine the enterprise and governance requirements here: '{prob}'. "
        f"In enterprise system architectures, we must comply with contractual and regulatory obligations. "
        f"The critical constraints are {invs[0]} and {invs[1]}. "
        f"Let us evaluate the failure boundaries: {invs[2]}. "
        f"The necessary remediation and compliance action is {invs[3]}. "
        f"Everything aligns with industry best practices and legal auditing standards."
    )
    
    grug_cot = (
        f"Enterprise rule: {invs[0]}.\n"
        f"Incident state: {invs[1]}.\n"
        f"Threshold violation: {invs[2]}.\n"
        f"Action: {invs[3]}."
    )
    
    answer = f"**Executive Summary:** {sol}\n**Remediation:** Enforce {invs[3]} immediately."
    
    e_raw = compute_trajectory_energy(verbose_trace, invs)
    e_grug = compute_trajectory_energy(grug_cot, invs)
    ratio = round(len(verbose_trace.split()) / max(1, len(grug_cot.split())), 2)
    
    return GrugifierSample(
        sample_id=f"ent_{model_key}_{idx:04d}",
        source_model=model_key,
        domain="enterprise_systems",
        language_dialect=dialect_key,
        task_prompt=prob,
        raw_verbose_cot=verbose_trace,
        grug_cot=grug_cot,
        final_answer=answer,
        invariants=invs,
        ebft_energy_raw=e_raw,
        ebft_energy_grug=e_grug,
        token_compression_ratio=ratio
    )

def generate_cyber_sample(model_key: str, dialect_key: str, idx: int) -> GrugifierSample:
    threats = [
        ("Nginx server receives flood of SYN packets without ACK completion. Diagnose attack and provide iptables mitigation.", "SYN Flood DoS attack exhausting connection backlog. Mitigate via SYN cookies and iptables rate-limiting.", ["SYN flood", "SYN cookies", "iptables -m limit", "drop spoofed SYN"]),
        ("Investigate Linux system anomaly: /tmp/.kworker cronjob running as root with high CPU and outbound UDP to port 4444.", "Cryptominer / reverse shell malware persistence. Terminate PID, remove root crontab entry, inspect binary hash.", ["reverse shell / miner", "kill -9 PID", "remove /tmp/.kworker", "clean crontab"]),
        ("Container escape detected via vulnerable hostPath volume mount in Kubernetes pod.", "Container breakout via mounted host docker.sock or root filesystem. Enforce PodSecurityAdmission restricted policy.", ["hostPath breakout", "restricted PSA", "readOnlyRootFilesystem", "drop CAP_SYS_ADMIN"])
    ]
    prob, sol, invs = threats[idx % len(threats)]
    
    verbose_trace = (
        f"Let me perform an incident triage on this security alert: '{prob}'. "
        f"First, we analyze the indicators of compromise (IOCs). "
        f"The observable behavior matches {invs[0]}. "
        f"The containment strategy must isolate the process without destroying forensic evidence: {invs[1]}. "
        f"The immediate remediation action is {invs[2]}. "
        f"Finally, hardening measures must be applied to prevent recurrence: {invs[3]}. "
        f"The response plan is verified."
    )
    
    grug_cot = (
        f"Threat: {invs[0]}.\n"
        f"IOC triage: {invs[1]}.\n"
        f"Immediate mitigation: {invs[2]}.\n"
        f"Hardening rule: {invs[3]}."
    )
    
    answer = f"**Incident Assessment:** {sol}\n**Immediate Command:** `{invs[2]}`\n**Hardening:** Enforce {invs[3]}."
    
    e_raw = compute_trajectory_energy(verbose_trace, invs)
    e_grug = compute_trajectory_energy(grug_cot, invs)
    ratio = round(len(verbose_trace.split()) / max(1, len(grug_cot.split())), 2)
    
    return GrugifierSample(
        sample_id=f"cyber_{model_key}_{idx:04d}",
        source_model=model_key,
        domain="cyber_devops",
        language_dialect=dialect_key,
        task_prompt=prob,
        raw_verbose_cot=verbose_trace,
        grug_cot=grug_cot,
        final_answer=answer,
        invariants=invs,
        ebft_energy_raw=e_raw,
        ebft_energy_grug=e_grug,
        token_compression_ratio=ratio
    )

DISPATCHER = {
    "coding": generate_coding_sample,
    "roleplay": generate_roleplay_sample,
    "acting": generate_acting_sample,
    "novel_writing": generate_novel_sample,
    "tool_use": generate_tool_sample,
    "multilingual_dialects": generate_multilingual_sample,
    "mathematics": generate_math_sample,
    "science_logic": generate_science_sample,
    "enterprise_systems": generate_enterprise_sample,
    "cyber_devops": generate_cyber_sample
}

def generate_dataset_shard(output_file: str, target_models: List[str], target_domains: List[str], samples_per_domain: int = 100, seed: int = 42):
    random.seed(seed)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_written = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for model_key in target_models:
            for domain_key in target_domains:
                gen_fn = DISPATCHER.get(domain_key, generate_coding_sample)
                for i in range(samples_per_domain):
                    dialect_key = random.choice(DIALECTS)
                    sample = gen_fn(model_key, dialect_key, i)
                    train_dict = sample.to_training_prompt()
                    f.write(json.dumps(train_dict, ensure_ascii=False) + "\n")
                    total_written += 1
                    
    print(f"Shard {output_file} completed: {total_written} samples across models {target_models} and domains {target_domains}")
    return total_written

if __name__ == "__main__":
    shard_name = sys.argv[1] if len(sys.argv) > 1 else "shard_all.jsonl"
    models_arg = sys.argv[2].split(",") if len(sys.argv) > 2 else MODEL_FAMILIES
    domains_arg = sys.argv[3].split(",") if len(sys.argv) > 3 else DOMAINS
    count_arg = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    
    out_path = os.path.join("/content/grug-speech-reasoning/datasets/universal_grugifier", shard_name)
    generate_dataset_shard(out_path, models_arg, domains_arg, count_arg)
