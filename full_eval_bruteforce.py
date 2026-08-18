# full_eval_bruteforce.py
import os
import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Adjusted imports for flat directory structure
from metrics import compute_accuracy, compute_piri, compute_chance_corrected_piri, compute_macro_f1
from error_analysis import build_confusion_matrix, compute_per_label_accuracy, save_confusion_matrix_to_csv

# Dummy configurations for standalone execution context
MODELS = ["roberta-base", "microsoft/deberta-base", "facebook/bart-large-mnli"]
DATASETS = ["snli", "mnli", "hans", "sick"]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
OUTPUT_DIR = "results/bruteforce_eval"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dummy_load_dataset(dataset_name):
    """Placeholder for actual dataset loading logic."""
    return [{"premise": "A man is playing a guitar.", "hypothesis": "A man is making music.", "label": 0}]

def dummy_apply_ablation(premise, hypothesis, condition):
    """Placeholder for ablation logic."""
    if condition == "full":
        return premise, hypothesis
    elif condition == "premise-only":
        return premise, ""
    elif condition == "hypothesis-only":
        return "", hypothesis

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in MODELS:
        for dataset_name in DATASETS:
            for seed in SEEDS:
                print(f"Running evaluation: Model={model_name}, Dataset={dataset_name}, Seed={seed}")
                
                # Unoptimized: Re-seed and re-load model/tokenizer per run
                set_seed(seed)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model.to(device)
                model.eval()

                # Unoptimized: Re-load dataset per run
                dataset = dummy_load_dataset(dataset_name)
                num_classes = 3 if dataset_name != "hans" else 2
                label_names = [f"Class_{i}" for i in range(num_classes)]

                conditions = ["full", "premise-only", "hypothesis-only"]
                results_dict = {"seed": seed, "model": model_name, "dataset": dataset_name}
                predictions_dict = {"true_labels": []}
                
                for condition in conditions:
                    predictions_dict[condition] = []

                # Inference loop
                for example in dataset:
                    predictions_dict["true_labels"].append(example["label"])
                    
                    for condition in conditions:
                        ab_prem, ab_hyp = dummy_apply_ablation(example["premise"], example["hypothesis"], condition)
                        
                        inputs = tokenizer(ab_prem, ab_hyp, return_tensors="pt", padding=True, truncation=True).to(device)
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                            pred_label = torch.argmax(outputs.logits, dim=1).item()
                            predictions_dict[condition].append(pred_label)

                # Metric computation
                y_true = predictions_dict["true_labels"]
                acc_full = compute_accuracy(y_true, predictions_dict["full"])
                acc_prem = compute_accuracy(y_true, predictions_dict["premise-only"])
                acc_hyp = compute_accuracy(y_true, predictions_dict["hypothesis-only"])

                results_dict["accuracy_full"] = acc_full
                results_dict["accuracy_premise"] = acc_prem
                results_dict["accuracy_hypothesis"] = acc_hyp
                
                results_dict["piri_hypothesis"] = compute_piri(acc_full, acc_hyp)
                results_dict["piri_premise"] = compute_piri(acc_full, acc_prem)
                
                results_dict["cc_piri_hypothesis"] = compute_chance_corrected_piri(acc_full, acc_hyp, num_classes)
                results_dict["cc_piri_premise"] = compute_chance_corrected_piri(acc_full, acc_prem, num_classes)
                
                results_dict["macro_f1_full"] = compute_macro_f1(y_true, predictions_dict["full"], num_classes)

                # Error Analysis
                for condition in conditions:
                    cm = build_confusion_matrix(y_true, predictions_dict[condition], num_classes)
                    per_label_acc = compute_per_label_accuracy(cm, num_classes)
                    
                    cm_filename = os.path.join(OUTPUT_DIR, f"{model_name.replace('/', '-')}_{dataset_name}_seed{seed}_{condition}_cm.csv")
                    save_confusion_matrix_to_csv(cm, cm_filename, label_names)
                    
                    results_dict[f"per_label_acc_{condition}"] = per_label_acc

                # Save JSON results
                result_filename = os.path.join(OUTPUT_DIR, f"{model_name.replace('/', '-')}_{dataset_name}_seed{seed}_results.json")
                with open(result_filename, "w") as f:
                    json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    main()