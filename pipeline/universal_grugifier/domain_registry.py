"""
Domain Registry and Invariant Rules for 10+ Diverse Reasoning Fields.
"""

from typing import Dict, List

DOMAINS_REGISTRY: Dict[str, Dict[str, str]] = {
    "coding": {
        "name": "Coding & Software Engineering",
        "description": "Algorithmic logic, data structures, concurrency, debugging, SQL, Rust, C++, Python",
        "invariant_focus": "Time/space complexity, edge cases, loop invariants, state mutations, deterministic return types",
        "grug_template": "Input: [types/state]. Complexity target: [O-notation]. Invariant: [loop/branch]. Done."
    },
    "roleplay": {
        "name": "Roleplay & Character Persona",
        "description": "Interactive storytelling, distinct voice, fictional world constraints, emotional consistency",
        "invariant_focus": "Persona tone, canonical knowledge boundaries, motivations, relationship to interlocutor",
        "grug_template": "Persona: [character]. Goal: [immediate motivation]. Voice rules: [dialect/quirks]. Invariant: [taboos/limits]. Done."
    },
    "acting": {
        "name": "Acting & Dramatic Screenplay",
        "description": "Dramatic subtext, blocking, beat transitions, emotional stakes, pacing and pause direction",
        "invariant_focus": "Dramatic objective, subtext vs spoken text, physical blocking, internal conflict shift",
        "grug_template": "Scene: [location]. Objective: [actor goal]. Subtext: [hidden motive]. Beat: [shift point]. Done."
    },
    "novel_writing": {
        "name": "Novel Writing & Creative Narrative",
        "description": "Scene atmosphere, sensory immersion, plot mechanics, theme foreshadowing, climax progression",
        "invariant_focus": "Point of view (POV), sensory anchor, tension escalation, reveal mechanics",
        "grug_template": "POV: [viewpoint]. Sensory anchor: [detail]. Arc: [tension curve]. Climax trigger: [reveal]. Done."
    },
    "tool_use": {
        "name": "Tool Use & Function Dispatch",
        "description": "JSON API schemas, argument validation, external execution sequencing, error handling",
        "invariant_focus": "Strict parameter types, required keys, schema boundaries, return payload parsing",
        "grug_template": "Target tool: [tool_name]. Params: [key:val]. Validation: [types/bounds]. Exec plan: [call]. Done."
    },
    "multilingual_dialects": {
        "name": "Multilingual Dialects & Global Idioms",
        "description": "Top 50 global languages, regional vernaculars, cultural idioms, code-switching",
        "invariant_focus": "Regional vocabulary, grammatical mood, cultural honorifics, phonological cues",
        "grug_template": "Target dialect: [name/region]. Grammar rules: [morphology]. Idiom map: [slang/phrases]. Tone: [cultural nuance]. Done."
    },
    "mathematics": {
        "name": "Mathematics & Quantitative Deduction",
        "description": "Algebra, number theory, combinatorics, geometry, calculus, probability",
        "invariant_focus": "Conservation of equality, unit consistency, domain boundaries, non-zero divisors",
        "grug_template": "Given: [vars]. Theorem/Formula: [rule]. Reduction: [step1 -> step2]. Boundary check: [limits]. Done."
    },
    "science_logic": {
        "name": "Science, Physics & Empirical Deduction",
        "description": "Newtonian mechanics, thermodynamics, organic chemistry synthesis, molecular biology",
        "invariant_focus": "Physical laws (energy conservation, stoichiometry, causality, thermodynamic limits)",
        "grug_template": "System: [boundary/state]. Governing law: [equation]. Force/Reaction: [vector/flux]. State transition: [final]. Done."
    },
    "enterprise_systems": {
        "name": "Enterprise Systems & Governance",
        "description": "Multi-constraint SLA schedules, GDPR/HIPAA compliance, supply chain routing, audit logs",
        "invariant_focus": "Policy compliance constraints, deadline guarantees, privacy boundaries, redundancy",
        "grug_template": "SLA: [threshold]. Compliance rule: [regulation]. Resource capacity: [limits]. Failover: [backup]. Done."
    },
    "cyber_devops": {
        "name": "Cybersecurity & DevOps Engineering",
        "description": "Packet inspection, privilege escalation containment, Kubernetes orchestration, CI/CD pipelines",
        "invariant_focus": "Least privilege, boundary firewall rules, idempotent manifests, rollback triggers",
        "grug_template": "Threat/Task: [vector/job]. Boundary: [net/role]. Mitigation/Spec: [steps]. Verification: [healthcheck]. Done."
    }
}
