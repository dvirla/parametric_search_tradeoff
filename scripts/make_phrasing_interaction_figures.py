#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import wilcoxon

ROOT = "/home/dvirla/projects/parametric_search_tradeoff"
OUT = os.path.join(ROOT, "results", "cue_briefing")
os.makedirs(OUT, exist_ok=True)

MODEL_ORDER = ["gemini-3.1-pro-preview", "gemini-3.5-flash", "qwen3.5_122b", "qwen3.5_35b", "qwen3.5_4b", "gemma4_31b", "gemma4_e4b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b"]
MODEL_LABEL = {
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "qwen3.5_122b": "Qwen3.5 122B", "qwen3.5_35b": "Qwen3.5 35B", "qwen3.5_4b": "Qwen3.5 4B",
    "gemma4_31b": "Gemma4 31B", "gemma4_e4b": "Gemma4 E4B", 
    "gpt-oss_120b": "GPT-OSS 120B", "gpt-oss_20b": "GPT-OSS 20B",
    "nemotron-3-nano_30b": "Nemotron3 30B"
}
PHRASING_COLOR = {"verbose": "#1b9e77", "terse": "#d95f02", "epi_strong": "#7570b3", "orig": "#1b9e77"}

def base_cue(cond):
    c = cond
    for p in ("verbose_", "terse_", "orig_", "epi_strong_"):
        if c.startswith(p):
            c = c[len(p):]
    return c

ALL_CUES = ["plain", "natural", "elaborate", "polite", "query", "direct"]
ALL_CUE_LABEL = {c: c.upper() for c in ALL_CUES}

def load_tokens_fig4(path):
    d = pd.read_csv(path)
    d["cue"] = d["condition"].map(base_cue)
    d["phrasing"] = d["dataset"]
    return d[["model", "phrasing", "cue", "example_id", "search_calls"]]

TOK4 = {
    "FRAMES": load_tokens_fig4(os.path.join(ROOT, "results/frames_token_analysis/joined_tokens.csv")),
    "MedQA": load_tokens_fig4(os.path.join(ROOT, "results/medqa_token_analysis/joined_tokens.csv"))
}

sns.set_theme(style="whitegrid")

MODEL_GROUP_1 = ['gemini-3.1-pro-preview', 'gemini-3.5-flash', 'qwen3.5_122b', 'gemma4_31b', 'gpt-oss_120b']
MODEL_GROUP_2 = [m for m in MODEL_ORDER if m not in MODEL_GROUP_1]
MODEL_GROUPS = [('primary', MODEL_GROUP_1), ('secondary', MODEL_GROUP_2)]

for group_name, group_models in MODEL_GROUPS:
    N = len(group_models)
    
    # Interaction
    fig, axes = plt.subplots(2, N, figsize=(4.2 * N, 8), constrained_layout=True)
    for r, ds in enumerate(["FRAMES", "MedQA"]):
        df = TOK4[ds]
        df = df[df["cue"].isin(ALL_CUES)]
        for cidx, m in enumerate(group_models):
            # If N=1, axes[r][cidx] won't work, but here N>1
            ax = axes[r][cidx]
            m_df = df[df["model"] == m]
            
            if m_df.empty or len(m_df["phrasing"].unique()) < 2:
                ax.set_axis_off()
                if r == 0: ax.set_title(MODEL_LABEL[m], fontsize=12, fontweight="bold")
                continue
                
            sns.pointplot(data=m_df, x="cue", y="search_calls", hue="phrasing", ax=ax, 
                          order=ALL_CUES, palette=PHRASING_COLOR, dodge=True, markers=["o", "s", "D"],
                          err_kws={'linewidth': 1.5}, capsize=0.1)
            
            ax.set_xticklabels([ALL_CUE_LABEL[c][:4] for c in ALL_CUES], rotation=30)
            ax.set_xlabel("")
            if cidx == 0:
                ax.set_ylabel(f"{ds}\nMean Search Calls")
            else:
                ax.set_ylabel("")
                
            if r == 0:
                ax.set_title(MODEL_LABEL[m], fontsize=12, fontweight="bold")
                
            if r == 0 and cidx == 0:
                ax.legend(title="Phrasing", fontsize=9)
            else:
                if ax.get_legend(): ax.get_legend().remove()

    fig.suptitle("Interaction between Cue and Phrasing (ANOVA Visualized)\nParallel lines = no interaction. Crossing/diverging lines = phrasing is coupled with specific cues.", fontsize=15, fontweight="bold")
    fig.savefig(os.path.join(OUT, f"fig4_phrasing_interaction_{group_name}.png"))
    plt.close(fig)

    # Difference
    fig, axes = plt.subplots(2, N, figsize=(4.2 * N, 8), constrained_layout=True)
    for r, (ds, ph1, ph2) in enumerate([("FRAMES", "verbose", "terse"), ("MedQA", "orig", "terse")]):
        df = TOK4[ds]
        df = df[df["cue"].isin(ALL_CUES)]
        for cidx, m in enumerate(group_models):
            ax = axes[r][cidx]
            m_df = df[df["model"] == m]
            
            if m_df.empty or ph1 not in m_df["phrasing"].values or ph2 not in m_df["phrasing"].values:
                ax.set_axis_off()
                if r == 0: ax.set_title(MODEL_LABEL[m], fontsize=12, fontweight="bold")
                continue
                
            piv = m_df.pivot_table(index=["example_id", "cue"], columns="phrasing", values="search_calls").reset_index()
            piv = piv.dropna(subset=[ph1, ph2])
            piv["diff"] = piv[ph1] - piv[ph2]
            
            if piv.empty:
                ax.set_axis_off()
                if r == 0: ax.set_title(MODEL_LABEL[m], fontsize=12, fontweight="bold")
                continue

            sns.pointplot(data=piv, x="cue", y="diff", ax=ax, order=ALL_CUES, 
                          color="#d95f02", markers="o", err_kws={'linewidth': 1.5}, capsize=0.1)
            
            ax.axhline(0, color="black", ls="--", lw=1)
            ax.set_xticklabels([ALL_CUE_LABEL[c][:4] for c in ALL_CUES], rotation=30)
            ax.set_xlabel("")
            if cidx == 0:
                ax.set_ylabel(f"{ds}\nΔ Search Calls\n({ph1.capitalize()} − {ph2.capitalize()})")
            else:
                ax.set_ylabel("")
                
            if r == 0:
                ax.set_title(MODEL_LABEL[m], fontsize=12, fontweight="bold")

    fig.suptitle("Repeated-Measures Interaction Plot: Paired Differences\nError bars show the 95% CI of the WITHIN-SUBJECT difference.", fontsize=15, fontweight="bold")
    fig.savefig(os.path.join(OUT, f"fig4_phrasing_paired_interaction_{group_name}.png"))
    plt.close(fig)

print("Saved phrasing interaction plots.")
