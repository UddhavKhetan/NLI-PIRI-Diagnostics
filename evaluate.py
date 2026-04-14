import os
import pandas as pd
from data import load_snli_data, load_mnli_matched, load_hans, load_xnli, load_anli, load_sick
from models import NLIModel

def evaluate_pipeline():
    sample_size = 1000 
    TARGET_DATASET = "sick"  # Set to SICK for final evaluation, can be 'snli', 'mnli', 'hans', 'xnli', or 'anli' as well
    XNLI_LANG = "en"         # Change this to 'sw', 'ar', 'fr', etc. later
    
    if TARGET_DATASET == "snli":
        dataset = load_snli_data(sample_size=sample_size)
    elif TARGET_DATASET == "mnli":
        dataset = load_mnli_matched(sample_size=sample_size)
    elif TARGET_DATASET == "hans":
        dataset = load_hans(sample_size=sample_size)
    elif TARGET_DATASET == "xnli":
        dataset = load_xnli(language=XNLI_LANG, sample_size=sample_size)
    elif TARGET_DATASET == "anli":
        dataset = load_anli(round_num=3, sample_size=sample_size)
    elif TARGET_DATASET == "sick":
        dataset = load_sick(sample_size=sample_size)
    else:
        raise ValueError("Invalid TARGET_DATASET.")
        
    models_to_test = {
        "roberta": "cross-encoder/nli-roberta-base",
        "deberta": "cross-encoder/nli-deberta-v3-base"
    }
    
    os.makedirs("results", exist_ok=True)
    
    for short_name, hf_name in models_to_test.items():
        print(f"\n{'='*50}")
        print(f"Evaluating Model: {short_name.upper()} on {TARGET_DATASET.upper()}")
        print(f"{'='*50}")
        
        nli_system = NLIModel(model_name=hf_name)
        
        correct_full, correct_hyp, correct_prem = 0, 0, 0
        results_data = []
        
        for i, example in enumerate(dataset):
            premise = example['premise']
            hypothesis = example['hypothesis']
            target_label = example['label'] 
            # Safely grab the heuristic (SNLI/MNLI don't have this)
            heuristic = example.get('heuristic', 'standard')
            
            # 1. Full Input
            pred_full = nli_system.predict(premise, hypothesis)
            
            # Fair HANS evaluation: accept Neutral (1) or Contradiction (2) as correct for non-entailment
            if TARGET_DATASET == "hans" and target_label == 2:
                is_correct = (pred_full in [1, 2])
            else:
                is_correct = (pred_full == target_label)
                
            if is_correct: correct_full += 1
                
            # 2 & 3. Conditional Partial Inputs
            if TARGET_DATASET != "hans":
                pred_hyp = nli_system.predict("", hypothesis)
                if pred_hyp == target_label: correct_hyp += 1

                pred_prem = nli_system.predict(premise, "")
                if pred_prem == target_label: correct_prem += 1
            else:
                pred_hyp, pred_prem = -1, -1
                
            results_data.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'gold_label': target_label,
                'heuristic': heuristic,  # NEW COLUMN
                'pred_full': pred_full,
                'pred_hyp_only': pred_hyp,
                'pred_prem_only': pred_prem
            })
                
            if (i + 1) % 100 == 0:
                print(f"[{short_name.upper()}] Processed {i + 1}/{sample_size}...")
                
        # Calculate & Print Metrics
        acc_full = correct_full / sample_size
        acc_hyp = correct_hyp / sample_size
        acc_prem = correct_prem / sample_size
        
        piri_hyp = (acc_full - acc_hyp) / acc_full if acc_full > 0 else 0.0
        piri_prem = (acc_full - acc_prem) / acc_full if acc_full > 0 else 0.0
        
        print(f"\n--- {short_name.upper()} Final Results ---")
        print(f"Full Input Accuracy:      {acc_full * 100:.1f}%")
        print(f"Hypothesis-Only Accuracy: {acc_hyp * 100:.1f}%")
        print(f"Premise-Only Accuracy:    {acc_prem * 100:.1f}%")
        print(f"PIRI_hyp:                 {piri_hyp:.3f}")
        
        # Update the CSV save path to include the language if XNLI is used
        if TARGET_DATASET == "xnli":
            csv_path = f"results/xnli_{XNLI_LANG}_{short_name}_results.csv"
        else:
            csv_path = f"results/{TARGET_DATASET}_{short_name}_results.csv"
            
        df = pd.DataFrame(results_data)
        df.to_csv(csv_path, index=False)
        print(f"Saved {short_name} results to {csv_path}")

if __name__ == "__main__":
    evaluate_pipeline()