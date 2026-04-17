import argparse
import os
import pandas as pd
from data import get_dataset
from models import get_model
from analyze import generate_summary_csv
from config import MODELS, DATASETS

def get_ablated_text(strategy, tokenizer):
    """Returns the replacement string based on the ablation strategy."""
    if strategy == "empty":
        return ""
    elif strategy == "neutral":
        return "The entity is present."
    elif strategy == "mask":
        mask_tok = tokenizer.mask_token if tokenizer.mask_token else "[MASK]"
        return " ".join([mask_tok] * 5) # Approximate average sentence length
    elif strategy == "random":
        return "apple guitar window abstract train" # Arbitrary non-informative tokens
    return ""

def evaluate_combination(model_key, dataset_key, sample_size=1000, seeds=[42, 43, 44], ablation_strategy="empty"):
    print(f"\nEvaluating: {model_key.upper()} on {dataset_key.upper()} with ablation '{ablation_strategy}'...")
    
    model = get_model(model_key)
    ablated_string = get_ablated_text(ablation_strategy, model.tokenizer)
    
    all_results = []
    
    for seed in seeds:
        print(f"  Running seed {seed}...")
        dataset = get_dataset(dataset_key, sample_size=sample_size, seed=seed)
        
        for example in dataset:
            premise = example['premise']
            hypothesis = example['hypothesis']
            label = example['label']
            heuristic = example.get('heuristic', 'standard')
            
            # Full Input
            pred_full, probs_full = model.predict(premise, hypothesis)
            
            # Hypothesis-Only (Ablate Premise)
            pred_hyp, probs_hyp = model.predict(ablated_string, hypothesis)
            
            # Premise-Only (Ablate Hypothesis)
            pred_prem, probs_prem = model.predict(premise, ablated_string)
            
            all_results.append({
                'seed': seed,
                'ablation_strategy': ablation_strategy,
                'premise': premise,
                'hypothesis': hypothesis,
                'gold_label': label,
                'heuristic': heuristic,
                'pred_full': pred_full,
                'prob_full_ent': probs_full[0], 'prob_full_neu': probs_full[1], 'prob_full_con': probs_full[2],
                'pred_hyp_only': pred_hyp,
                'prob_hyp_ent': probs_hyp[0], 'prob_hyp_neu': probs_hyp[1], 'prob_hyp_con': probs_hyp[2],
                'pred_prem_only': pred_prem,
                'prob_prem_ent': probs_prem[0], 'prob_prem_neu': probs_prem[1], 'prob_prem_con': probs_prem[2],
            })

    # Save aggregated multi-seed log
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/{dataset_key}_{model_key}_{ablation_strategy}_results.csv"
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"Saved multi-seed logs to: {csv_path}")


if __name__ == "__main__":
    import argparse
    # Ensure MODELS and DATASETS are imported from config
    # Ensure generate_summary_csv is imported from analyze
    
    parser = argparse.ArgumentParser(description="NLI Partial Input Diagnostics Pipeline")
    parser.add_argument("--models", nargs='+', required=True, help="List of model keys from config")
    parser.add_argument("--datasets", nargs='+', required=True, help="List of dataset keys from config")
    parser.add_argument("--sample_size", type=int, default=None, help="Override default sample size")
    # --- CHANGED: --seed is now --seeds ---
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 43, 44], help="List of random seeds for multiple runs")
    # --- NEW: Ablation strategy ---
    parser.add_argument("--ablation_strategy", type=str, default="empty", choices=["empty", "mask", "neutral", "random"], help="Strategy for masking premise/hypothesis")
    
    args = parser.parse_args()

    # --- VALIDATION BLOCK ---
    from config import MODELS, DATASETS # Make sure these exist in your imports
    for m in args.models:
        if m not in MODELS:
            parser.error(f"Unknown model: '{m}'. Available: {', '.join(MODELS.keys())}")
            
    for d in args.datasets:
        if d not in DATASETS:
            parser.error(f"Unknown dataset: '{d}'. Available: {', '.join(DATASETS.keys())}")
    # -----------------------------

    # 1. Run all requested combinations
    for d in args.datasets:
        for m in args.models:
            evaluate_combination(
                model_key=m, 
                dataset_key=d, 
                sample_size=args.sample_size, 
                seeds=args.seeds, 
                ablation_strategy=args.ablation_strategy
            )

    # 2. Update the master summary for the dashboard
    from analyze import generate_summary_csv
    generate_summary_csv()
    print("\nDiagnostics complete!")