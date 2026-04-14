import argparse
import os
import pandas as pd
from data import get_dataset
from models import get_model
from analyze import generate_summary_csv
from config import MODELS, DATASETS

def evaluate_combination(model_key, dataset_key, sample_size=None, seed=42):
    print(f"\nEvaluating: {model_key.upper()} on {dataset_key.upper()}...")
    dataset = get_dataset(dataset_key, sample_size=sample_size, seed=seed)
    model = get_model(model_key)

    results = []
    for example in dataset:
        premise = example['premise']
        hypothesis = example['hypothesis']
        label = example['label']
        heuristic = example.get('heuristic', 'standard')

        # Inference
        pred_full = model.predict(premise, hypothesis)
        
        pred_hyp = model.predict("", hypothesis)
        pred_prem = model.predict(premise, "")

        results.append({
            'premise': premise,
            'hypothesis': hypothesis,
            'gold_label': label,
            'heuristic': heuristic,
            'pred_full': pred_full,
            'pred_hyp_only': pred_hyp,
            'pred_prem_only': pred_prem
        })

    # Save individual log
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/{dataset_key}_{model_key}_results.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Saved logs to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLI Partial Input Diagnostics Pipeline")
    parser.add_argument("--models", nargs='+', required=True, help="List of model keys from config")
    parser.add_argument("--datasets", nargs='+', required=True, help="List of dataset keys from config")
    parser.add_argument("--sample_size", type=int, default=None, help="Override default sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset sampling")
    
    args = parser.parse_args()

    # --- NEW: VALIDATION BLOCK ---
    for m in args.models:
        if m not in MODELS:
            parser.error(f"Unknown model: '{m}'. Available models: {', '.join(MODELS.keys())}")
            
    for d in args.datasets:
        if d not in DATASETS:
            parser.error(f"Unknown dataset: '{d}'. Available datasets: {', '.join(DATASETS.keys())}")
    # -----------------------------

    # 1. Run all requested combinations
    for d in args.datasets:
        for m in args.models:
            evaluate_combination(m, d, args.sample_size, args.seed)

    # 2. Update the master summary for the dashboard
    generate_summary_csv()
    print("\nDiagnostics complete!")