"""
Universal Grugifier Data Schemas and Invariant Verifiers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import re

@dataclass
class GrugifierSample:
    sample_id: str
    source_model: str
    domain: str
    language_dialect: str
    task_prompt: str
    raw_verbose_cot: str
    grug_cot: str
    final_answer: str
    invariants: List[str] = field(default_factory=list)
    ebft_energy_raw: float = 0.0
    ebft_energy_grug: float = 0.0
    token_compression_ratio: float = 1.0

    def to_training_prompt(self) -> Dict[str, str]:
        """
        Format as instruction tuning pair for the Universal Grugifier model.
        System + User prompt -> Grug Reasoning + Output Answer.
        """
        user_msg = (
            f"[Source Model Family]: {self.source_model}\n"
            f"[Domain]: {self.domain}\n"
            f"[Language / Dialect]: {self.language_dialect}\n"
            f"[Task Prompt]:\n{self.task_prompt}\n\n"
            f"[Raw Verbose Reasoning]:\n{self.raw_verbose_cot}"
        )
        
        assistant_msg = (
            f"<think>\n{self.grug_cot}\nDone.\n</think>\n"
            f"{self.final_answer}"
        )
        
        return {
            "system": (
                "You are the Universal Grugifier Engine. "
                "Transform verbose reasoning traces from any source model into dense, "
                "invariant-anchored Grug Speech (<think>...Done.</think>) optimized for "
                "Energy-Based Fine-Tuning (EBFT). Preserve all domain, persona, and causal invariants."
            ),
            "user": user_msg,
            "assistant": assistant_msg,
            "sample_id": self.sample_id,
            "domain": self.domain,
            "source_model": self.source_model,
            "language_dialect": self.language_dialect,
            "compression_ratio": self.token_compression_ratio
        }

def compute_trajectory_energy(text_cot: str, invariants: List[str]) -> float:
    """
    Computes EBFT trajectory energy:
    E(x,y) = alpha * Length + beta * Entropy_penalty + gamma * Violation - delta * Invariant_bonus
    """
    tokens = text_cot.split()
    length_penalty = len(tokens) * 0.15
    
    # Repetition / Discursiveness penalty
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    redundancy_penalty = (1.0 - unique_ratio) * 15.0
    
    # Tag violation penalty
    violation_penalty = 0.0
    if not (text_cot.strip().endswith("Done.") or "Done." in text_cot):
        violation_penalty += 20.0
    
    # Invariant preservation reward
    invariant_hits = sum(1 for inv in invariants if inv.lower() in text_cot.lower())
    invariant_ratio = invariant_hits / max(1, len(invariants)) if invariants else 1.0
    invariant_reward = invariant_ratio * 25.0
    
    energy = length_penalty + redundancy_penalty + violation_penalty - invariant_reward
    return round(float(energy), 2)
