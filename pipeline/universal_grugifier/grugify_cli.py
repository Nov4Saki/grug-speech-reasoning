"""
Universal Grugifier CLI:
The cognitive compiler for compressing reasoning traces into EBFT-ready Grug Speech.
"""

import sys
import argparse
import json
from pipeline.universal_grugifier.ebft_transfer_compiler import EBFTTransferCompiler
from pipeline.universal_grugifier.model_profiles import MODEL_PROFILES

def main():
    parser = argparse.ArgumentParser(description="Universal Grugifier CLI: Invariant-Anchored Reasoning Distiller")
    parser.add_argument("--target-model", type=str, default="gemma-2-9b", choices=list(MODEL_PROFILES.keys()), help="Target model architecture for EBFT transfer")
    parser.add_argument("--domain", type=str, default="coding", help="Problem domain (coding, roleplay, acting, tool_use, multilingual_dialects, etc.)")
    parser.add_argument("--prompt", type=str, required=True, help="User task prompt")
    parser.add_argument("--verbose-trace", type=str, required=True, help="Raw verbose reasoning trace from any source model")
    parser.add_argument("--answer", type=str, default=None, help="Optional ground truth answer")
    parser.add_argument("--output-json", action="store_true", help="Output full JSON object with EBFT metrics")
    
    args = parser.parse_args()
    
    compiler = EBFTTransferCompiler(target_model_key=args.target_model)
    answer = args.answer or "Direct answer derived from invariant constraints."
    
    record = compiler.compile_ebft_dataset_record(
        task_prompt=args.prompt,
        raw_verbose_cot=args.verbose_trace,
        final_answer=answer,
        domain=args.domain,
        target_model_key=args.target_model
    )
    
    if args.output_json:
        print(json.dumps(record, indent=2))
    else:
        print("=" * 80)
        print(f"UNIVERSAL GRUGIFIER COMPILER -> Target Model: [{args.target_model.upper()}]")
        print(f"Domain: {args.domain} | Token Compression: {record['compression_ratio']}x | Energy Reduction: {record['energy_reduction_pct']}%")
        print("=" * 80)
        print("\n[COMPILED GRUG REASONING TRACE]:")
        print(f"<think>\n{record['grug_cot']}\n</think>")
        print("\n[TARGET MODEL EBFT TRAINING FORMAT]:")
        print(record["formatted_training_text"])
        print("=" * 80)

if __name__ == "__main__":
    main()
