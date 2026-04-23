import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def plot_correlation_deltas():
    base_dir = "logs/facts_one_hop"
    models = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    data = []

    for model in models:
        vanilla_path = os.path.join(base_dir, model, "analysis_vanilla", "correlation_summary.csv")
        instruct_path = os.path.join(base_dir, model, "analysis_with_sys_instruct", "correlation_summary.csv")

        if os.path.exists(vanilla_path) and os.path.exists(instruct_path):
            try:
                df_vanilla = pd.read_csv(vanilla_path)
                df_instruct = pd.read_csv(instruct_path)

                # We assume the files have the same structure and we want the last 3 rows.
                # We'll rely on the 'Relationship' and 'Method' columns to match them to be safe.
                
                # Taking the last 3 rows from vanilla as the reference keys
                target_rows = df_vanilla.tail(3)

                for _, row in target_rows.iterrows():
                    relationship = row['Relationship']
                    method = row['Method']
                    
                    # Find matching row in instruct
                    match_instruct = df_instruct[
                        (df_instruct['Relationship'] == relationship) & 
                        (df_instruct['Method'] == method)
                    ]
                    
                    if not match_instruct.empty:
                        corr_vanilla = row['Correlation']
                        corr_instruct = match_instruct.iloc[0]['Correlation']
                        delta = corr_instruct - corr_vanilla
                        
                        # Create a readable label for the relationship
                        label = f"{relationship} ({method})"
                        # Simplify label for plot
                        if "No-Search Confidence vs. Searches" in relationship and "Spearman" in method:
                            short_label = "No-Search Conf vs. Searches"
                        elif "Pre-Search Confidence vs. Searches" in relationship:
                            short_label = "Pre-Search Conf vs. Searches"
                        elif "No-Search Confidence vs. Pre-Search Confidence" in relationship:
                            short_label = "No-Search Conf vs. Pre-Search Conf"
                        else:
                            short_label = label

                        data.append({
                            "Model": model,
                            "Relationship": short_label,
                            "Delta": delta
                        })
            except Exception as e:
                print(f"Error processing model {model}: {e}")
        else:
            print(f"Skipping {model}: missing analysis files.")

    if not data:
        print("No data found to plot.")
        return

    df_plot = pd.DataFrame(data)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    # Create grouped bar chart
    chart = sns.barplot(
        data=df_plot,
        x="Model",
        y="Delta",
        hue="Relationship",
        palette="viridis"
    )
    
    plt.title("Change in Correlation (System Instruct - Vanilla)", fontsize=16)
    plt.ylabel("Delta Correlation", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.axhline(0, color='black', linewidth=1)
    plt.legend(title="Relationship", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = os.path.join(base_dir, "correlation_delta_plot.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_correlation_deltas()
