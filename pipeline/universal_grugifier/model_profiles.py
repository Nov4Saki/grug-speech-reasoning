"""
Model Profiles and Prompt Formats for Cross-Family Grugification
Supports:
- Gemma Family: gemma-2-2b, gemma-2-9b, gemma-4-e2b
- Qwen Family: qwen-3.5-2b, qwen-3.5-4b, qwen-3.8-27b
- Nanbeige: nanbeige-4.2-3b
- MiniCPM: minicpm-5-2b
- LFM Family: lfm-3b, lfm-7b, lfm-40b
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelProfile:
    family: str
    model_id: str
    display_name: str
    architecture: str
    context_window: int
    system_tag: str
    user_tag: str
    assistant_tag: str
    end_tag: str
    cot_start_marker: str
    cot_end_marker: str
    chat_template_type: str
    typical_verbosity: str  # "high", "extreme", "moderate"
    default_stop_tokens: List[str]

MODEL_PROFILES: Dict[str, ModelProfile] = {
    # 1. Gemma Family (3 models)
    "gemma-2-2b": ModelProfile(
        family="gemma",
        model_id="google/gemma-2-2b-it",
        display_name="Gemma-2-2B-IT",
        architecture="gemma2",
        context_window=8192,
        system_tag="<start_of_turn>user\n",
        user_tag="<start_of_turn>user\n",
        assistant_tag="<start_of_turn>model\n",
        end_tag="<end_of_turn>",
        cot_start_marker="<think>\n",
        cot_end_marker="\nDone.\n</think>\n",
        chat_template_type="gemma",
        typical_verbosity="high",
        default_stop_tokens=["<end_of_turn>"]
    ),
    "gemma-2-9b": ModelProfile(
        family="gemma",
        model_id="google/gemma-2-9b-it",
        display_name="Gemma-2-9B-IT",
        architecture="gemma2",
        context_window=8192,
        system_tag="<start_of_turn>user\n",
        user_tag="<start_of_turn>user\n",
        assistant_tag="<start_of_turn>model\n",
        end_tag="<end_of_turn>",
        cot_start_marker="<think>\n",
        cot_end_marker="\nDone.\n</think>\n",
        chat_template_type="gemma",
        typical_verbosity="extreme",
        default_stop_tokens=["<end_of_turn>"]
    ),
    "gemma-4-e2b": ModelProfile(
        family="gemma",
        model_id="google/gemma-4-e2b",
        display_name="Gemma-4-E2B",
        architecture="gemma4",
        context_window=16384,
        system_tag="<start_of_turn>user\n",
        user_tag="<start_of_turn>user\n",
        assistant_tag="<start_of_turn>model\n",
        end_tag="<end_of_turn>",
        cot_start_marker="<think>\n",
        cot_end_marker="\nDone.\n</think>\n",
        chat_template_type="gemma",
        typical_verbosity="high",
        default_stop_tokens=["<end_of_turn>"]
    ),

    # 2. Qwen Family (3 models)
    "qwen-3.5-2b": ModelProfile(
        family="qwen",
        model_id="Qwen/Qwen3.5-2B-Instruct",
        display_name="Qwen-3.5-2B-Instruct",
        architecture="qwen2",
        context_window=32768,
        system_tag="<|im_start|>system\n",
        user_tag="<|im_start|>user\n",
        assistant_tag="<|im_start|>assistant\n",
        end_tag="<|im_end|>",
        cot_start_marker="<think>\n",
        cot_end_marker="\nDone.\n</think>\n",
        chat_template_type="chatml",
        typical_verbosity="high",
        default_stop_tokens=["<|im_end|>"]
    ),
    "qwen-3.5-4b": ModelProfile(
        family="qwen",
        model_id="Qwen/Qwen3.5-4B-Instruct",
        display_name="Qwen-3.5-4B-Instruct",
        architecture="qwen2",
        context_window=32768,
        system_tag="<|im_start|>system\n",
        user_tag="<|im_start|>user\n",
        assistant_tag="<|im_start|>assistant\n",
        end_tag="<|im_end|>",
        cot_start_marker="<think>\n",
        cot_end_marker="\nDone.\n</think>\n",
        chat_template_type="chatml",
        typical_verbosity="high",
        default_stop_tokens=["<|im_end|>"]
    ),
    "qwen-3.8-27b": ModelProfile(
        family="qwen",
        model_id="Qwen/Qwen3.8-27B-Instruct",
        display_name="Qwen-3.8-27B-Instruct",
        architecture="qwen2",
        context_window=65536,
        system_tag="<|im_start|>system\n",
        user_tag="<|im_start|>user\n",
        assistant_tag="<|im_start|>assistant\n",
        end_tag="<|im_end|>",
        cot_start_marker="<think>\n",
        cot_end_marker="\nDone.\n</think>\n",
        chat_template_type="chatml",
        typical_verbosity="extreme",
        default_stop_tokens=["<|im_end|>"]
    ),

    # 3. Nanbeige Family (1 model)
    "nanbeige-4.2-3b": ModelProfile(
        family="nanbeige",
        model_id="Nanbeige/Nanbeige4.2-3B",
        display_name="Nanbeige-4.2-3B",
        architecture="nanbeige",
        context_window=8192,
        system_tag="<|system|>\n",
        user_tag="<|user|>\n",
        assistant_tag="<|assistant|>\n",
        end_tag="<|end|>",
        cot_start_marker="<think>\n",
        cot_end_marker="\nDone.\n</think>\n",
        chat_template_type="nanbeige",
        typical_verbosity="high",
        default_stop_tokens=["<|end|>"]
    ),

    # 4. MiniCPM Family (1 model)
    "minicpm-5-2b": ModelProfile(
        family="minicpm",
        model_id="openbmb/MiniCPM5-2B",
        display_name="MiniCPM5-2B",
        architecture="minicpm",
        context_window=8192,
        system_tag="<用户>",
        user_tag="<用户>",
        assistant_tag="<AI>",
        end_tag="</s>",
        cot_start_marker="<think>\n",
        cot_end_marker="\nDone.\n</think>\n",
        chat_template_type="minicpm",
        typical_verbosity="moderate",
        default_stop_tokens=["</s>", "<用户>"]
    ),

    # 5. LFM (Liquid Foundation Models) Family (3 models)
    "lfm-3b": ModelProfile(
        family="lfm",
        model_id="LiquidAI/LFM-3B-Instruct",
        display_name="LFM-3B-Instruct",
        architecture="liquid_conv",
        context_window=32768,
        system_tag="<|prompt|>\n",
        user_tag="<|prompt|>\n",
        assistant_tag="<|reply|>\n",
        end_tag="<|endoftext|>",
        cot_start_marker="<think>\n",
        cot_end_marker="\nDone.\n</think>\n",
        chat_template_type="lfm",
        typical_verbosity="high",
        default_stop_tokens=["<|endoftext|>"]
    ),
    "lfm-7b": ModelProfile(
        family="lfm",
        model_id="LiquidAI/LFM-7B-Instruct",
        display_name="LFM-7B-Instruct",
        architecture="liquid_conv",
        context_window=32768,
        system_tag="<|prompt|>\n",
        user_tag="<|prompt|>\n",
        assistant_tag="<|reply|>\n",
        end_tag="<|endoftext|>",
        cot_start_marker="<think>\n",
        cot_end_marker="\nDone.\n</think>\n",
        chat_template_type="lfm",
        typical_verbosity="extreme",
        default_stop_tokens=["<|endoftext|>"]
    ),
    "lfm-40b": ModelProfile(
        family="lfm",
        model_id="LiquidAI/LFM-40B-Instruct",
        display_name="LFM-40B-Instruct",
        architecture="liquid_conv",
        context_window=65536,
        system_tag="<|prompt|>\n",
        user_tag="<|prompt|>\n",
        assistant_tag="<|reply|>\n",
        end_tag="<|endoftext|>",
        cot_start_marker="<think>\n",
        cot_end_marker="\nDone.\n</think>\n",
        chat_template_type="lfm",
        typical_verbosity="extreme",
        default_stop_tokens=["<|endoftext|>"]
    ),
}

def format_target_training_sample(profile_key: str, system_prompt: str, user_prompt: str, grug_cot: str, answer: str) -> str:
    """Format sample into target model's native chat template."""
    p = MODEL_PROFILES.get(profile_key, MODEL_PROFILES["qwen-3.5-2b"])
    
    if p.chat_template_type == "chatml":
        return (
            f"{p.system_tag}{system_prompt}{p.end_tag}\n"
            f"{p.user_tag}{user_prompt}{p.end_tag}\n"
            f"{p.assistant_tag}{p.cot_start_marker}{grug_cot}{p.cot_end_marker}{answer}{p.end_tag}"
        )
    elif p.chat_template_type == "gemma":
        return (
            f"{p.user_tag}{system_prompt}\n\n{user_prompt}{p.end_tag}\n"
            f"{p.assistant_tag}{p.cot_start_marker}{grug_cot}{p.cot_end_marker}{answer}{p.end_tag}"
        )
    elif p.chat_template_type == "nanbeige":
        return (
            f"{p.system_tag}{system_prompt}{p.end_tag}\n"
            f"{p.user_tag}{user_prompt}{p.end_tag}\n"
            f"{p.assistant_tag}{p.cot_start_marker}{grug_cot}{p.cot_end_marker}{answer}{p.end_tag}"
        )
    elif p.chat_template_type == "minicpm":
        return (
            f"{p.user_tag}{system_prompt}\n{user_prompt}"
            f"{p.assistant_tag}{p.cot_start_marker}{grug_cot}{p.cot_end_marker}{answer}{p.end_tag}"
        )
    elif p.chat_template_type == "lfm":
        return (
            f"{p.user_tag}[System: {system_prompt}]\n{user_prompt}\n"
            f"{p.assistant_tag}{p.cot_start_marker}{grug_cot}{p.cot_end_marker}{answer}{p.end_tag}"
        )
    else:
        return f"{user_prompt}\n{p.cot_start_marker}{grug_cot}{p.cot_end_marker}{answer}"
