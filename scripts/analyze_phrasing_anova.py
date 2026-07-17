import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = "/home/dvirla/projects/parametric_search_tradeoff"
MODEL_ORDER = ["gemini-3.1-pro-preview", "qwen3.5_122b", "qwen3.5_35b", "qwen3.5_4b", "gemma4_31b", "gemma4_e4b", "nemotron-3-nano_30b"]
CUES = ["plain", "natural", "elaborate", "polite", "query", "direct"]

def base_cue(cond):
    c = cond
    for p in ("verbose_", "terse_", "orig_", "epi_strong_"):
        if c.startswith(p):
            c = c[len(p):]
    return c

def load_tokens(path):
    d = pd.read_csv(path)
    d["cue"] = d["condition"].map(base_cue)
    d["phrasing"] = d["dataset"]
    return d[["model", "phrasing", "cue", "example_id", "search_calls"]]

TOK = {
    "FRAMES": load_tokens(os.path.join(ROOT, "results/frames_token_analysis/joined_tokens.csv")),
    "MedQA": load_tokens(os.path.join(ROOT, "results/medqa_token_analysis/joined_tokens.csv"))
}

for ds in ["FRAMES", "MedQA"]:
    print(f"\n=======================================================")
    print(f" DATASET: {ds}")
    print(f"=======================================================\n")
    
    df = TOK[ds]
    # Filter to only the cues we care about
    df = df[df["cue"].isin(CUES)].copy()
    
    for m in MODEL_ORDER:
        m_df = df[df["model"] == m].copy()
        if m_df.empty:
            continue
            
        print(f"--- Model: {m} ---")
        
        # We must drop NaNs
        m_df = m_df.dropna(subset=["search_calls"])
        
        # If there is no variance in phrasing, skip
        if len(m_df["phrasing"].unique()) < 2:
            print("  Not enough phrasings to compare.\n")
            continue
            
        # Fit Linear Mixed-Effects Model
        # Formula: search_calls ~ phrasing + cue + phrasing:cue
        # Random effect: example_id (repeated measures)
        try:
            model = smf.mixedlm("search_calls ~ C(phrasing) * C(cue)", m_df, groups=m_df["example_id"])
            result = model.fit(method='cg') # cg is sometimes more stable for these simple categorical models
            
            # Print the ANOVA-style summary (Wald tests for the terms)
            # We can get the wald test for the main effects and interactions
            
            # We will use Wald test on the parameters
            # Instead of manually extracting, we can just print the p-values of the terms
            # or use an ANOVA table if available for mixedlm. 
            # statsmodels doesn't have an easy ANOVA wrapper for mixedlm, so we'll look at the summary table
            
            # Let's extract the p-values for phrasing and interactions
            # We look for terms containing 'phrasing'
            pvals = result.pvalues
            
            phrasing_main_sig = any(p < 0.05 for name, p in pvals.items() if 'C(phrasing)' in name and ':' not in name)
            interaction_sig = any(p < 0.05 for name, p in pvals.items() if ':' in name)
            
            print(f"  Main effect of Phrasing significant? {'YES' if phrasing_main_sig else 'NO'}")
            print(f"  Phrasing x Cue Interaction significant? {'YES' if interaction_sig else 'NO'}")
            
            # To be more descriptive, let's print the actual p-values for the interaction terms
            if interaction_sig:
                print("  Significant Interactions (Phrasing coupled with specific Cue):")
                for name, p in pvals.items():
                    if ':' in name and p < 0.05:
                        print(f"    - {name}: p = {p:.4f}")
            else:
                print("  The phrasing effect is consistent across cues (no significant interaction).")
            print()
            
        except Exception as e:
            print(f"  Could not fit MixedLM: {e}\n")
