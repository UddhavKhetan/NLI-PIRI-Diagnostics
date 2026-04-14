import re
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

def load_results(filepath="results/snli_roberta_results.csv"):
    """Loads the prediction CSV into a pandas DataFrame."""
    return pd.read_csv(filepath)

def print_overall_metrics(df):
    """Recomputes standard accuracies and Reliance Indices."""
    total = len(df)
    acc_full = (df['pred_full'] == df['gold_label']).sum() / total
    acc_hyp = (df['pred_hyp_only'] == df['gold_label']).sum() / total
    acc_prem = (df['pred_prem_only'] == df['gold_label']).sum() / total
    
    piri_hyp = (acc_full - acc_hyp) / acc_full if acc_full > 0 else 0.0
    piri_prem = (acc_full - acc_prem) / acc_full if acc_full > 0 else 0.0

    print("--- Overall Metrics (Sanity Check) ---")
    print(f"Full Input Accuracy:      {acc_full * 100:.1f}%")
    print(f"Hypothesis-Only Accuracy: {acc_hyp * 100:.1f}%")
    print(f"Premise-Only Accuracy:    {acc_prem * 100:.1f}%")
    print(f"PIRI_hyp:                 {piri_hyp:.3f}")
    print(f"PIRI_prem:                {piri_prem:.3f}\n")

def print_per_class_metrics(df):
    """Computes accuracy broken down by the gold label."""
    label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
    
    print("--- Per-Class Accuracies ---")
    print(f"{'Class':<15} | {'Full':<6} | {'Hyp-only':<8} | {'Prem-only':<9}")
    print("-" * 45)
    
    for label_val, label_name in label_map.items():
        subset = df[df['gold_label'] == label_val]
        if len(subset) == 0: continue
        
        acc_full = (subset['pred_full'] == subset['gold_label']).sum() / len(subset)
        acc_hyp = (subset['pred_hyp_only'] == subset['gold_label']).sum() / len(subset)
        acc_prem = (subset['pred_prem_only'] == subset['gold_label']).sum() / len(subset)
        
        print(f"{label_name:<15} | {acc_full*100:>5.1f}% | {acc_hyp*100:>7.1f}% | {acc_prem*100:>8.1f}%")
    print()

def print_confusion_matrices(df):
    """Prints text-based confusion matrices for all three conditions."""
    labels = [0, 1, 2]
    label_names = ["Ent", "Neu", "Con"]
    
    conditions = [
        ("Full Input", 'pred_full'),
        ("Hypothesis-Only", 'pred_hyp_only'),
        ("Premise-Only", 'pred_prem_only')
    ]
    
    for name, col in conditions:
        cm = confusion_matrix(df['gold_label'], df[col], labels=labels)
        print(f"--- Confusion Matrix: {name} ---")
        print(f"          Predicted")
        print(f"         {label_names[0]:>4} {label_names[1]:>4} {label_names[2]:>4}")
        for i, row_label in enumerate(label_names):
            print(f"True {row_label:<3} {cm[i][0]:>4} {cm[i][1]:>4} {cm[i][2]:>4}")
        print()

def add_negation_feature(df):
    """Adds a boolean 'has_negation' column based on lexical triggers in the hypothesis."""
    # \b ensures we match whole words (e.g., 'not' but not 'nothing' twice), 
    # and we handle "n't" separately since it attaches to words.
    pattern = r"\b(?:no|not|never|nobody|nothing|none)\b|n't"
    df['has_negation'] = df['hypothesis'].str.lower().str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
    return df

def print_negation_analysis(df):
    """Prints overall and per-class accuracies sliced by the presence of negation."""
    subsets = [
        ("Negation", df[df['has_negation'] == True]),
        ("No-negation", df[df['has_negation'] == False])
    ]
    
    print("--- Negation vs Non-Negation Accuracies ---")
    print(f"{'Subset':<13} | {'Condition':<11} | {'Accuracy':<8}")
    print("-" * 38)
    
    for name, subset in subsets:
        if len(subset) == 0: continue
        
        acc_full = (subset['pred_full'] == subset['gold_label']).sum() / len(subset)
        acc_hyp = (subset['pred_hyp_only'] == subset['gold_label']).sum() / len(subset)
        acc_prem = (subset['pred_prem_only'] == subset['gold_label']).sum() / len(subset)
        
        print(f"{name:<13} | {'Full':<11} | {acc_full*100:>5.1f}%")
        print(f"{name:<13} | {'Hyp-only':<11} | {acc_hyp*100:>5.1f}%")
        print(f"{name:<13} | {'Prem-only':<11} | {acc_prem*100:>5.1f}%")
        print("-" * 38)

    print("\n--- Negation Slice: Per-Class Accuracies ---")
    label_map = {0: "Ent", 1: "Neu", 2: "Con"}
    print(f"{'Subset':<13} | {'Class':<5} | {'Full':<6} | {'Hyp-only':<8} | {'Prem-only':<9}")
    print("-" * 51)
    
    for name, subset in subsets:
        for label_val, label_name in label_map.items():
            class_subset = subset[subset['gold_label'] == label_val]
            if len(class_subset) == 0: continue
            
            acc_full = (class_subset['pred_full'] == class_subset['gold_label']).sum() / len(class_subset)
            acc_hyp = (class_subset['pred_hyp_only'] == class_subset['gold_label']).sum() / len(class_subset)
            acc_prem = (class_subset['pred_prem_only'] == class_subset['gold_label']).sum() / len(class_subset)
            
            print(f"{name:<13} | {label_name:<5} | {acc_full*100:>5.1f}% | {acc_hyp*100:>7.1f}% | {acc_prem*100:>8.1f}%")
        print("-" * 51)
    print()

def print_cross_dataset_summary():
    """Reads all model/dataset combinations and prints a master summary table."""
    print("--- Cross-Dataset & Cross-Model Summary ---")
    print(f"{'Dataset':<8} | {'Model':<8} | {'Full Acc':<10} | {'Hyp-only':<10} | {'Prem-only':<10} | {'PIRI_hyp':<8}")
    print("-" * 65)
    
    datasets = ["snli", "mnli"]
    models = ["roberta", "deberta", "distilroberta", "distilbert", "piri_distilroberta"]
    
    for ds in datasets:
        for mod in models:
            filepath = f"results/{ds}_{mod}_results.csv"
            try:
                df = pd.read_csv(filepath)
                total = len(df)
                acc_full = (df['pred_full'] == df['gold_label']).sum() / total
                acc_hyp = (df['pred_hyp_only'] == df['gold_label']).sum() / total
                acc_prem = (df['pred_prem_only'] == df['gold_label']).sum() / total
                piri_hyp = (acc_full - acc_hyp) / acc_full if acc_full > 0 else 0.0
                
                print(f"{ds.upper():<8} | {mod.capitalize():<8} | {acc_full*100:>8.1f}%   | {acc_hyp*100:>8.1f}%   | {acc_prem*100:>8.1f}%   | {piri_hyp:.3f}")
            except FileNotFoundError:
                print(f"{ds.upper():<8} | {mod.capitalize():<8} | --- FILE NOT FOUND ---")
    print()

def print_hans_summary():
    print("--- HANS Full-Input Accuracy ---")
    print(f"{'Model':<10} | {'Overall':<8} | {'Lexical':<8} | {'Subsequence':<12} | {'Constituent':<11}")
    print("-" * 57)
    
    models = ["roberta", "deberta"]
    for mod in models:
        filepath = f"results/hans_{mod}_results.csv"
        try:
            df = pd.read_csv(filepath)
            
            # Apply the fair accuracy logic for HANS
            correct_mask = (df['pred_full'] == df['gold_label']) | ((df['gold_label'] == 2) & (df['pred_full'] == 1))
            
            acc_overall = correct_mask.mean() * 100
            acc_lex = correct_mask[df['heuristic'] == 'lexical_overlap'].mean() * 100
            acc_sub = correct_mask[df['heuristic'] == 'subsequence'].mean() * 100
            acc_con = correct_mask[df['heuristic'] == 'constituent'].mean() * 100
            
            print(f"{mod.capitalize():<10} | {acc_overall:>5.1f}%   | {acc_lex:>6.1f}%   | {acc_sub:>10.1f}%  | {acc_con:>9.1f}%")
        except FileNotFoundError:
            print(f"{mod.capitalize():<10} | --- FILE NOT FOUND ---")
    print()

def print_piri_vs_hans_summary():
    """Combines Partial Input Reliance (PIRI) with HANS robustness in one table."""
    print("--- Artifact Vulnerability Summary ---")
    print(f"{'Model':<10} | {'SNLI PIRI_hyp':<15} | {'MNLI PIRI_hyp':<15} | {'HANS Accuracy':<15}")
    print("-" * 62)
    
    models = ["roberta", "deberta", "distilroberta", "distilbert", "piri_distilroberta"]
    for mod in models:
        # 1. Calculate SNLI PIRI
        try:
            df_snli = pd.read_csv(f"results/snli_{mod}_results.csv")
            acc_full_s = (df_snli['pred_full'] == df_snli['gold_label']).sum() / len(df_snli)
            acc_hyp_s = (df_snli['pred_hyp_only'] == df_snli['gold_label']).sum() / len(df_snli)
            piri_snli = (acc_full_s - acc_hyp_s) / acc_full_s if acc_full_s > 0 else 0.0
        except FileNotFoundError:
            piri_snli = 0.0

        # 2. Calculate MNLI PIRI
        try:
            df_mnli = pd.read_csv(f"results/mnli_{mod}_results.csv")
            acc_full_m = (df_mnli['pred_full'] == df_mnli['gold_label']).sum() / len(df_mnli)
            acc_hyp_m = (df_mnli['pred_hyp_only'] == df_mnli['gold_label']).sum() / len(df_mnli)
            piri_mnli = (acc_full_m - acc_hyp_m) / acc_full_m if acc_full_m > 0 else 0.0
        except FileNotFoundError:
            piri_mnli = 0.0
            
        # 3. Calculate HANS Overall Accuracy
        try:
            df_hans = pd.read_csv(f"results/hans_{mod}_results.csv")
            correct_mask = (df_hans['pred_full'] == df_hans['gold_label']) | ((df_hans['gold_label'] == 2) & (df_hans['pred_full'] == 1))
            hans_acc = correct_mask.mean() * 100
        except FileNotFoundError:
            hans_acc = 0.0
            
        print(f"{mod.capitalize():<10} | {piri_snli:<15.3f} | {piri_mnli:<15.3f} | {hans_acc:>8.1f}%")
    print()

def print_grand_master_summary():
    """Compiles the final master table across all 5 datasets."""
    print("\n" + "="*85)
    print(" "*25 + "THE GRAND MASTER SUMMARY")
    print("="*85)
    print(f"{'Model':<10} | {'SNLI (PIRI)':<16} | {'MNLI (PIRI)':<16} | {'HANS Acc':<10} | {'ANLI Acc':<10} | {'SICK Acc':<10}")
    print("-" * 85)
    
    models = ["roberta", "deberta"]
    for mod in models:
        # Helper to safely get Full Acc and PIRI
        def get_metrics(ds):
            try:
                df = pd.read_csv(f"results/{ds}_{mod}_results.csv")
                acc_full = (df['pred_full'] == df['gold_label']).sum() / len(df)
                acc_hyp = (df['pred_hyp_only'] == df['gold_label']).sum() / len(df)
                piri = (acc_full - acc_hyp) / acc_full if acc_full > 0 else 0.0
                return acc_full * 100, piri
            except FileNotFoundError:
                return 0.0, 0.0

        # Helper for HANS (custom logic)
        def get_hans():
            try:
                df = pd.read_csv(f"results/hans_{mod}_results.csv")
                correct = (df['pred_full'] == df['gold_label']) | ((df['gold_label'] == 2) & (df['pred_full'] == 1))
                return correct.mean() * 100
            except FileNotFoundError:
                return 0.0

        snli_acc, snli_piri = get_metrics("snli")
        mnli_acc, mnli_piri = get_metrics("mnli")
        anli_acc, _ = get_metrics("anli")
        sick_acc, _ = get_metrics("sick")
        hans_acc = get_hans()

        # Format strings for the PIRI columns
        snli_str = f"{snli_acc:.1f}% ({snli_piri:.3f})"
        mnli_str = f"{mnli_acc:.1f}% ({mnli_piri:.3f})"
        
        print(f"{mod.capitalize():<10} | {snli_str:<16} | {mnli_str:<16} | {hans_acc:>5.1f}%    | {anli_acc:>5.1f}%    | {sick_acc:>5.1f}%")
    print("="*85 + "\n")

def generate_summary_csv(output_path="results/summary.csv"):
    """Aggregates all individual run CSVs into a single machine-readable summary."""
    print(f"\nGenerating system summary -> {output_path}")
    from config import MODELS, DATASETS # Import your new registry
    
    summary_data = []
    
    for mod in MODELS.keys():
        for ds in DATASETS.keys():
            filepath = f"results/{ds}_{mod}_results.csv"
            if not os.path.exists(filepath):
                continue
                
            df = pd.read_csv(filepath)
            total = len(df)
            
            # Standard metrics
            acc_full = (df['pred_full'] == df['gold_label']).sum() / total if total > 0 else 0
            acc_hyp = (df['pred_hyp_only'] == df['gold_label']).sum() / total if total > 0 else 0
            acc_prem = (df['pred_prem_only'] == df['gold_label']).sum() / total if total > 0 else 0
            
            # HANS specific fair-evaluation
            if ds == "hans":
                correct = (df['pred_full'] == df['gold_label']) | ((df['gold_label'] == 2) & (df['pred_full'] == 1))
                acc_full = correct.sum() / total if total > 0 else 0
            
            piri_hyp = (acc_full - acc_hyp) / acc_full if acc_full > 0 else 0.0
            
            summary_data.append({
                "Model": mod,
                "Dataset": ds,
                "Full_Acc": acc_full,
                "Hyp_Acc": acc_hyp,
                "Prem_Acc": acc_prem,
                "PIRI_hyp": piri_hyp
            })
            
    pd.DataFrame(summary_data).to_csv(output_path, index=False)

if __name__ == "__main__":
    df = load_results()
    
    # Existing analysis
    print_overall_metrics(df)
    print_per_class_metrics(df)
    print_confusion_matrices(df)
    
    # New negation slice analysis
    df = add_negation_feature(df)
    print_negation_analysis(df)

    print_cross_dataset_summary()

    print_hans_summary()

    print_piri_vs_hans_summary()

    print_grand_master_summary()