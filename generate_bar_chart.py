import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def binomial_confidence_interval(accuracy_percentage, n_samples=1000, z=1.96):
    """Calculates the 95% CI for a binomial proportion (accuracy)."""
    p = accuracy_percentage / 100.0
    # Standard error formula for proportions
    se = np.sqrt((p * (1 - p)) / n_samples)
    # Return as percentage
    return z * se * 100.0

def generate_plot():
    # Target the multi-seed CSV
    csv_path = "results/snli_distilroberta_empty_results.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}.")
        return

    df = pd.read_csv(csv_path)
    
    # Calculate overall mean accuracy across all evaluated samples
    m_full = (df['pred_full'] == df['gold_label']).mean() * 100
    m_hyp = (df['pred_hyp_only'] == df['gold_label']).mean() * 100
    m_prem = (df['pred_prem_only'] == df['gold_label']).mean() * 100
    
    n_total = len(df) # Total samples evaluated
    
    # Get Binomial 95% CIs
    ci_full = binomial_confidence_interval(m_full, n_total)
    ci_hyp = binomial_confidence_interval(m_hyp, n_total)
    ci_prem = binomial_confidence_interval(m_prem, n_total)
    
    # Plot Configuration
    conditions = ['Full Input', 'Hypothesis-Only', 'Premise-Only']
    means = [m_full, m_hyp, m_prem]
    cis = [ci_full, ci_hyp, ci_prem]
    colors = ['#4c72b0', '#dd8452', '#55a868']

    plt.figure(figsize=(9, 6))
    
    # Create bars with error caps
    bars = plt.bar(conditions, means, yerr=cis, capsize=8, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Accuracy Degradation Across Input Conditions (95% Binomial CI)', fontsize=14, fontweight='bold')
    
    # Zoom the Y-axis slightly so error bars aren't squashed at the top
    plt.ylim(0, 105) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add numerical labels
    for bar, ci in zip(bars, cis):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + ci + 1.5, f"{yval:.1f}% ±{ci:.1f}%", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    os.makedirs("images", exist_ok=True)
    out_path = "images/condition_dropoff_error_bars.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Success! Detailed bar chart with binomial error bars saved to {out_path}")

if __name__ == "__main__":
    generate_plot()
