import pandas as pd, json

tuned = pd.read_csv("results/musique-natural/interplay_analysis_tuned_val100/interplay_summary.csv")
tuned_ids = set(tuned["example_id"].unique())
print(f"Tuned model covers {len(tuned_ids)} example_ids")

base = pd.read_csv("results/musique-natural/interplay_analysis/interplay_summary.csv")
base_filtered = base[(base["model"] == "nemotron-3-nano_30b") & (base["example_id"].isin(tuned_ids))]
print(f"Base model filtered to {base_filtered['example_id'].nunique()} matching example_ids")

base_filtered.to_csv("results/musique-natural/interplay_analysis/interplay_summary_nemotron_base_100q.csv", index=False)