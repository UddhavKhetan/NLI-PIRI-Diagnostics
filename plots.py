import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison(roberta_csv="results/snli_roberta_results.csv", deberta_csv="results/snli_deberta_results.csv", filename="model_comparison.png"):
    print("Reading data for both models...")
    try:
        df_rob = pd.read_csv(roberta_csv)
        df_deb = pd.read_csv(deberta_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure both models have been evaluated.")
        return

    # Calculate accuracies
    def get_acc(df):
        total = len(df)
        return [
            (df['pred_full'] == df['gold_label']).sum() / total * 100,
            (df['pred_hyp_only'] == df['gold_label']).sum() / total * 100,
            (df['pred_prem_only'] == df['gold_label']).sum() / total * 100
        ]

    acc_rob = get_acc(df_rob)
    acc_deb = get_acc(df_deb)
    
    # Setup plot
    conditions = ['Full Input', 'Hypothesis-Only', 'Premise-Only']
    x = np.arange(len(conditions))
    width = 0.35  

    print(f"Generating grouped bar chart: {filename}...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Draw bars
    rects1 = ax.bar(x - width/2, acc_rob, width, label='RoBERTa', color='#4C72B0')
    rects2 = ax.bar(x + width/2, acc_deb, width, label='DeBERTa', color='#DD8452')

    # Formatting
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Drop Across Models', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 105)
    ax.legend()

    # Add text labels on bars
    ax.bar_label(rects1, fmt='%.1f%%', padding=3, fontsize=10)
    ax.bar_label(rects2, fmt='%.1f%%', padding=3, fontsize=10)

    # Random baseline
    ax.axhline(y=33.3, color='gray', linestyle='--', alpha=0.7)
    ax.text(2.45, 35, 'Random Baseline (33%)', color='gray', fontsize=9, ha='right', va='bottom')

    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Plot saved successfully to {filename}")

def plot_cross_dataset_comparison(filename="cross_dataset_comparison.png"):
    """Generates a 1x2 subplot comparing both datasets and both models."""
    datasets = ["snli", "mnli"]
    models = ["roberta", "deberta"]
    conditions = ['Full', 'Hyp-Only', 'Prem-Only']
    x = np.arange(len(conditions))
    width = 0.35  

    print(f"Generating cross-dataset plot: {filename}...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for i, ds in enumerate(datasets):
        ax = axes[i]
        
        # Helper to safely load accuracy arrays
        def get_acc(mod):
            try:
                df = pd.read_csv(f"results/{ds}_{mod}_results.csv")
                total = len(df)
                return [
                    (df['pred_full'] == df['gold_label']).sum() / total * 100,
                    (df['pred_hyp_only'] == df['gold_label']).sum() / total * 100,
                    (df['pred_prem_only'] == df['gold_label']).sum() / total * 100
                ]
            except FileNotFoundError:
                return [0, 0, 0]

        acc_rob = get_acc("roberta")
        acc_deb = get_acc("deberta")
        
        rects1 = ax.bar(x - width/2, acc_rob, width, label='RoBERTa', color='#4C72B0')
        rects2 = ax.bar(x + width/2, acc_deb, width, label='DeBERTa', color='#DD8452')

        ax.set_title(f'{ds.upper()} Dataset', fontsize=14, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylim(0, 105)
        
        # Random baseline
        ax.axhline(y=33.3, color='gray', linestyle='--', alpha=0.7)
        
        # Formatting
        if i == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.legend(loc='upper right')
        
        ax.bar_label(rects1, fmt='%.1f', padding=3, fontsize=9)
        ax.bar_label(rects2, fmt='%.1f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved successfully to {filename}")

def plot_vulnerability_summary(filename="vulnerability_summary.png"):
    """Creates a dual-plot comparing Hypothesis Bias (PIRI) vs Syntactic Bias (HANS)."""
    print(f"Generating artifact vulnerability plot: {filename}...")
    
    models = ["roberta", "deberta"]
    piri_snli, piri_mnli, hans_acc = [], [], []
    
    # Safely extract all metrics
    for mod in models:
        try:
            df = pd.read_csv(f"results/snli_{mod}_results.csv")
            f, h = df['pred_full'] == df['gold_label'], df['pred_hyp_only'] == df['gold_label']
            piri_snli.append((f.mean() - h.mean()) / f.mean() if f.mean() > 0 else 0)
        except: piri_snli.append(0)
            
        try:
            df = pd.read_csv(f"results/mnli_{mod}_results.csv")
            f, h = df['pred_full'] == df['gold_label'], df['pred_hyp_only'] == df['gold_label']
            piri_mnli.append((f.mean() - h.mean()) / f.mean() if f.mean() > 0 else 0)
        except: piri_mnli.append(0)
            
        try:
            df = pd.read_csv(f"results/hans_{mod}_results.csv")
            correct = (df['pred_full'] == df['gold_label']) | ((df['gold_label'] == 2) & (df['pred_full'] == 1))
            hans_acc.append(correct.mean() * 100)
        except: hans_acc.append(0)

    # Build the 1x2 plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(models))
    width = 0.35
    
    # Left: PIRI (Hypothesis Bias)
    rects1 = ax1.bar(x - width/2, piri_snli, width, label='SNLI', color='#4C72B0')
    rects2 = ax1.bar(x + width/2, piri_mnli, width, label='MNLI', color='#55A868')
    ax1.set_ylabel('Partial Input Reliance Index (PIRI_hyp)', fontsize=11)
    ax1.set_title('Hypothesis Bias\n(Higher PIRI = More reliant on premise)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['RoBERTa', 'DeBERTa'], fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.bar_label(rects1, fmt='%.3f', padding=3)
    ax1.bar_label(rects2, fmt='%.3f', padding=3)

    # Right: HANS (Syntactic Robustness)
    rects3 = ax2.bar(x, hans_acc, width=0.5, color='#DD8452')
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Syntactic Robustness (HANS)\n(Higher = Better)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['RoBERTa', 'DeBERTa'], fontsize=11)
    ax2.set_ylim(0, 105)
    
    # HANS essentially operates as a binary task (entailment vs non-entailment)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7) 
    ax2.text(1.4, 52, 'Random Baseline (50%)', color='gray', fontsize=9, ha='right')
    ax2.bar_label(rects3, fmt='%.1f%%', padding=3)

    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved successfully to {filename}")

def plot_piri_vs_hans_scatter(filename="piri_vs_hans_scatter.png"):
    print(f"Generating scatter plot: {filename}...")
    
    # Added the experimental model
    models = ['RoBERTa', 'DeBERTa', 'DistilRoBERTa', 'DistilBERT', 'PIRI-Reg DistilRoBERTa']
    mnli_piri = [0.577, 0.547, 0.567, 0.559, 0.508]
    hans_acc = [71.0, 79.1, 53.2, 54.3, 51.1]
    
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#9370DB']
    markers = ['o', 'o', 'o', 'o', '*'] # Star for the experimental model
    sizes = [150, 150, 150, 150, 300]   # Make the star larger
    
    plt.figure(figsize=(9, 6))
    
    for i in range(len(models)):
        plt.scatter(mnli_piri[i], hans_acc[i], color=colors[i], marker=markers[i], s=sizes[i], label=models[i], edgecolors='black', zorder=5)
        # Offset text slightly
        plt.text(mnli_piri[i] + 0.002, hans_acc[i] + 0.5, models[i], fontsize=10, va='bottom')

    plt.title('Hypothesis Bias vs. Syntactic Robustness', fontsize=14, pad=15)
    plt.xlabel('MNLI Partial Input Reliance Index (PIRI_hyp)', fontsize=12)
    plt.ylabel('HANS Full-Input Accuracy (%)', fontsize=12)
    
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.text(0.585, 50.5, 'HANS Random Baseline', color='gray', fontsize=9)
    
    # Expand X-axis slightly to fit the new data point
    plt.xlim(0.50, 0.59) 
    plt.ylim(45, 85)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Plot saved successfully to {filename}")

if __name__ == "__main__":
    plot_model_comparison()
    plot_cross_dataset_comparison()
    plot_vulnerability_summary()
    plot_piri_vs_hans_scatter()