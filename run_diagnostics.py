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
        
        # Unpack the entire dataset into lists
        premises = [ex['premise'] for ex in dataset]
        hypotheses = [ex['hypothesis'] for ex in dataset]
        labels = [ex['label'] for ex in dataset]
        heuristics = [ex.get('heuristic', 'standard') for ex in dataset]
        
        # Create full arrays of ablated text
        ablated_list = [ablated_string] * len(premises)
        
        # --- BATCH INFERENCE ---
        # 1. Full Input
        preds_full, probs_full = model.predict_batch(premises, hypotheses)
        # 2. Hypothesis-Only
        preds_hyp, probs_hyp = model.predict_batch(ablated_list, hypotheses)
        # 3. Premise-Only
        preds_prem, probs_prem = model.predict_batch(premises, ablated_list)
        
        # Reconstruct the results log
        for i in range(len(premises)):
            all_results.append({
                'seed': seed,
                'ablation_strategy': ablation_strategy,
                'premise': premises[i],
                'hypothesis': hypotheses[i],
                'gold_label': labels[i],
                'heuristic': heuristics[i],
                'pred_full': preds_full[i],
                'prob_full_ent': probs_full[i][0], 'prob_full_neu': probs_full[i][1], 'prob_full_con': probs_full[i][2],
                'pred_hyp_only': preds_hyp[i],
                'prob_hyp_ent': probs_hyp[i][0], 'prob_hyp_neu': probs_hyp[i][1], 'prob_hyp_con': probs_hyp[i][2],
                'pred_prem_only': preds_prem[i],
                'prob_prem_ent': probs_prem[i][0], 'prob_prem_neu': probs_prem[i][1], 'prob_prem_con': probs_prem[i][2],
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