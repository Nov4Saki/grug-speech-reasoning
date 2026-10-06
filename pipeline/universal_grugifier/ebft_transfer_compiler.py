"""
EBFT Transfer Compiler:
Compiles verbose traces from any source model into Grug Speech and exports
training pairs formatted in the target model's native chat template with EBFT energy scores.
"""

import json
import re
from typing import Dict, List, Any, Optional

from pipeline.universal_grugifier.model_profiles import MODEL_PROFILES, format_target_training_sample
from pipeline.universal_grugifier.schema import compute_trajectory_energy

class EBFTTransferCompiler:
    def __init__(self, target_model_key: str = "qwen-3.5-2b"):
        self.target_model_key = target_model_key
        self.profile = MODEL_PROFILES.get(target_model_key, MODEL_PROFILES["qwen-3.5-2b"])

    def grugify_trace_rule_based(self, prompt: str, verbose_trace: str, domain: str) -> Dict[str, Any]:
        """
        Synthesizes / distills raw verbose trace into canonical Grug Speech.
        Prunes conversational filler, identifies core equations/variables,
        and anchors invariants.
        """
        # Step 1: Extract invariants (numbers, quoted terms, key nouns)
        numbers = re.findall(r"[-+]?\d*\.?\d+", prompt + " " + verbose_trace)
        unique_numbers = list(dict.fromkeys(numbers))
        
        # Step 2: Strip typical polite/conversational filler phrases
        filler_patterns = [
            r"Well, let me think.*?\.",
            r"First of all, the user wants.*?\.",
            r"Let me contemplate.*?\.",
            r"Let us walk through.*?\.",
            r"In order to solve this.*?\.",
            r"I would be happy to.*?\.",
            r"Let us carefully check.*?\.",
            r"Now let us examine.*?\."
        ]
        
        cleaned_text = verbose_trace
        for pat in filler_patterns:
            cleaned_text = re.sub(pat, "", cleaned_text, flags=re.IGNORECASE)
            
        sentences = [s.strip() for s in cleaned_text.split(".") if len(s.strip()) > 5]
        
        # Step 3: Format dense Grug primitive
        core_steps = sentences[:3] if len(sentences) >= 3 else sentences
        grug_body_lines = []
        grug_body_lines.append(f"Domain: {domain}.")
        if unique_numbers:
            grug_body_lines.append(f"Invariants: {', '.join(unique_numbers[:4])}.")
        for st in core_steps:
            # Compress sentence
            words = st.split()
            short_st = " ".join(words[:10])
            grug_body_lines.append(f"Step: {short_st}.")
        
        grug_body = "\n".join(grug_body_lines)
        grug_cot = f"{grug_body}\nDone."
        
        # Step 4: EBFT metrics
        e_raw = compute_trajectory_energy(verbose_trace, unique_numbers)
        e_grug = compute_trajectory_energy(grug_cot, unique_numbers)
        compression = round(len(verbose_trace.split()) / max(1, len(grug_cot.split())), 2)
        
        return {
            "grug_cot": grug_cot,
            "raw_energy": e_raw,
            "grug_energy": e_grug,
            "energy_reduction_pct": round(max(0, (e_raw - e_grug) / max(1e-5, abs(e_raw))) * 100, 1),
            "compression_ratio": compression,
            "invariants_tracked": unique_numbers[:5]
        }

    def compile_ebft_dataset_record(
        self,
        task_prompt: str,
        raw_verbose_cot: str,
        final_answer: str,
        domain: str = "general",
        target_model_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Takes raw prompt + verbose trace, runs Grugifier compilation,
        and formats into the target model's native template for EBFT fine-tuning.
        """
        m_key = target_model_key or self.target_model_key
        res = self.grugify_trace_rule_based(task_prompt, raw_verbose_cot, domain)
        grug_cot = res["grug_cot"]
        
        # Strip Done. from body if present since format_target_training_sample adds it
        clean_body = grug_cot.replace("\nDone.", "").replace("Done.", "").strip()
        
        formatted_training_text = format_target_training_sample(
            profile_key=m_key,
            system_prompt=(
                "You are Grug Reasoner. Think in caveman logic inside <think>...</think> before answering. "
                "Keep thinking dense, invariant-focused, and eliminate conversational filler. End thinking with Done."
            ),
            user_prompt=task_prompt,
            grug_cot=clean_body,
            answer=final_answer
        )
        
        return {
            "target_model": m_key,
            "domain": domain,
            "formatted_training_text": formatted_training_text,
            "grug_cot": grug_cot,
            "compression_ratio": res["compression_ratio"],
            "raw_energy": res["raw_energy"],
            "grug_energy": res["grug_energy"],
            "energy_reduction_pct": res["energy_reduction_pct"],
            "invariants": res["invariants_tracked"]
        }
