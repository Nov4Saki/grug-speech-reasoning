import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

OUTPUT_DIR = "/content/grug-speech-reasoning/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = "/content/grug-speech-reasoning/evaluation/large_scale_300_benchmark_results.json"

with open(RESULTS_FILE) as f:
    results = json.load(f)

# ---------------------------------------------------------------------------
# 1. TOOL CALLING ACCURACY COMPARISON (NORMAL VS GRUGIFIED FROM SAME MODEL)
# ---------------------------------------------------------------------------
def plot_tool_calling_comparison():
    models = ['DeepSeek-R1-7B', 'SmolLM2-1.7B', 'Qwen3.5-4B', 'Qwen2.5-3B']
    normal_tools = [
        results['deepseek_r1_normal']['tool_pass_rate_pct'],
        results['smollm_1.7b_normal']['tool_pass_rate_pct'],
        results['qwen35_4b_normal']['tool_pass_rate_pct'],
        results['qwen25_3b_normal']['tool_pass_rate_pct']
    ]
    grug_tools = [
        results['deepseek_r1_grugified']['tool_pass_rate_pct'],
        results['smollm_1.7b_grugified']['tool_pass_rate_pct'],
        results['qwen35_4b_grugified']['tool_pass_rate_pct'],
        results['qwen25_3b_grugified']['tool_pass_rate_pct']
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, normal_tools, width, label='Normal Model (Verbose / Unconstrained)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, grug_tools, width, label='Grugified Model (Grug Speech Scaffold)', color='#2ecc71')

    ax.set_ylabel('Tool Calling Pass Rate (%) - 100 Tasks', fontsize=12, fontweight='bold')
    ax.set_title('Glaive AI Tool Calling: Normal vs Grugified from the Same Model', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor='#f8f9fa', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Annotations
    for bar in bars1:
        y = bar.get_height()
        ax.annotate(f'{y:.1f}%', xy=(bar.get_x() + bar.get_width()/2, y),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        y = bar.get_height()
        ax.annotate(f'{y:.1f}%', xy=(bar.get_x() + bar.get_width()/2, y),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "tool_calling_surge.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# 2. OVERALL ACCURACY ACROSS ALL 300 TASKS (NORMAL VS GRUGIFIED)
# ---------------------------------------------------------------------------
def plot_overall_comparison():
    models = ['DeepSeek-R1-7B', 'SmolLM2-1.7B', 'Qwen3.5-4B', 'Qwen2.5-3B']
    normal_overall = [
        results['deepseek_r1_normal']['overall_accuracy_pct'],
        results['smollm_1.7b_normal']['overall_accuracy_pct'],
        results['qwen35_4b_normal']['overall_accuracy_pct'],
        results['qwen25_3b_normal']['overall_accuracy_pct']
    ]
    grug_overall = [
        results['deepseek_r1_grugified']['overall_accuracy_pct'],
        results['smollm_1.7b_grugified']['overall_accuracy_pct'],
        results['qwen35_4b_grugified']['overall_accuracy_pct'],
        results['qwen25_3b_grugified']['overall_accuracy_pct']
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, normal_overall, width, label='Normal Model (Vanilla / Baseline)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, grug_overall, width, label='Grugified Model (Grug Speech Scaffold)', color='#3498db')

    ax.set_ylabel('Overall Accuracy (%) - 300 Tasks', fontsize=12, fontweight='bold')
    ax.set_title('Large-Scale 300-Benchmark Accuracy: Normal vs Grugified from Same Model', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 75)
    ax.legend(frameon=True, facecolor='#f8f9fa', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars1:
        y = bar.get_height()
        ax.annotate(f'{y:.1f}%', xy=(bar.get_x() + bar.get_width()/2, y),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        y = bar.get_height()
        ax.annotate(f'{y:.1f}%', xy=(bar.get_x() + bar.get_width()/2, y),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# 3. GSM8K MATH ACCURACY & LATENCY SURGE
# ---------------------------------------------------------------------------
def plot_math_comparison():
    models = ['DeepSeek-R1-7B', 'Qwen3.5-4B', 'SmolLM2-1.7B', 'Qwen2.5-3B']
    normal_math = [
        results['deepseek_r1_normal']['math_pass_rate_pct'],
        results['qwen35_4b_normal']['math_pass_rate_pct'],
        results['smollm_1.7b_normal']['math_pass_rate_pct'],
        results['qwen25_3b_normal']['math_pass_rate_pct']
    ]
    grug_math = [
        results['deepseek_r1_grugified']['math_pass_rate_pct'],
        results['qwen35_4b_grugified']['math_pass_rate_pct'],
        results['smollm_1.7b_grugified']['math_pass_rate_pct'],
        results['qwen25_3b_grugified']['math_pass_rate_pct']
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, normal_math, width, label='Normal Model', color='#e74c3c')
    bars2 = ax.bar(x + width/2, grug_math, width, label='Grugified Model', color='#f39c12')

    ax.set_ylabel('GSM8K Math Pass Rate (%) - 100 Tasks', fontsize=12, fontweight='bold')
    ax.set_title('GSM8K Quantitative Reasoning: Normal vs Grugified from Same Model', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 55)
    ax.legend(frameon=True, facecolor='#f8f9fa', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars1:
        y = bar.get_height()
        ax.annotate(f'{y:.1f}%', xy=(bar.get_x() + bar.get_width()/2, y),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        y = bar.get_height()
        ax.annotate(f'{y:.1f}%', xy=(bar.get_x() + bar.get_width()/2, y),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "math_accuracy_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# 4. TRAINING LOSS CONVERGENCE (QWEN 3.5 4B)
# ---------------------------------------------------------------------------
def plot_training_loss():
    steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    train_loss = [1.491, 0.3826, 0.0364, 0.0068, 0.0035, 0.00168, 0.00030, 0.00021, 0.00017, 0.00014, 0.00012, 0.000118, 0.000108, 0.000112, 0.000095]
    val_steps = [50, 100, 150]
    val_loss = [0.001736, 0.0001329, 0.0001136]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(steps, train_loss, 'o-', color='#3498db', linewidth=2.5, label='Qwen 3.5 4B Training Loss')
    ax.scatter(val_steps, val_loss, color='#e74c3c', s=120, zorder=5, label='1,100 Sample Val Loss')

    ax.set_yscale('log')
    ax.set_xlabel('Optimization Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Entropy Loss (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('Universal Grugifier Optimization: Logarithmic Loss Convergence', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='#f8f9fa', fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "training_loss_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_tool_calling_comparison()
    plot_overall_comparison()
    plot_math_comparison()
    plot_training_loss()
    print("All comparison graphs generated successfully!")
