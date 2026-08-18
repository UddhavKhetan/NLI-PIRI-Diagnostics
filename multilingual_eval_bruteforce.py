# multilingual_eval_bruteforce.py
import os
import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from metrics import compute_accuracy, compute_piri, compute_chance_corrected_piri
from error_analysis import build_confusion_matrix, save_confusion_matrix_to_csv

LANGUAGES = ["en", "fr", "es", "de", "zh"]
MODELS = ["roberta-base", "microsoft/deberta-base"]
SEEDS = [42, 43, 44]
OUTPUT_DIR = "results/xnli_multilingual"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def dummy_load_xnli(language):
    return [{"premise": "Example premise", "hypothesis": "Example hypothesis", "label": 0}]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    label_names = ["Entailment", "Neutral", "Contradiction"]

    for model_name in MODELS:
        for lang in LANGUAGES:
            for seed in SEEDS:
                print(f"XNLI Eval: Model={model_name}, Lang={lang}, Seed={seed}")
                
                set_seed(seed)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
                model.eval()

                dataset = dummy_load_xnli(lang)
                y_true = []
                y_pred_full = []
                y_pred_hyp = []

                for example in dataset:
                    y_true.append(example["label"])
                    
                    # Full Condition
                    inputs_full = tokenizer(example["premise"], example["hypothesis"], return_tensors="pt").to(device)
                    # Hypothesis-only Condition
                    inputs_hyp = tokenizer("", example["hypothesis"], return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        out_full = model(**inputs_full)
                        out_hyp = model(**inputs_hyp)
                        
                        y_pred_full.append(torch.argmax(out_full.logits, dim=1).item())
                        y_pred_hyp.append(torch.argmax(out_hyp.logits, dim=1).item())

                acc_full = compute_accuracy(y_true, y_pred_full)
                acc_hyp = compute_accuracy(y_true, y_pred_hyp)
                
                results = {
                    "model": model_name,
                    "language": lang,
                    "seed": seed,
                    "acc_full": acc_full,
                    "acc_hyp": acc_hyp,
                    "piri_hyp": compute_piri(acc_full, acc_hyp),
                    "cc_piri_hyp": compute_chance_corrected_piri(acc_full, acc_hyp, num_classes)
                }

                # Save metrics
                res_file = os.path.join(OUTPUT_DIR, f"{model_name.replace('/', '-')}_xnli_{lang}_seed{seed}.json")
                with open(res_file, "w") as f:
                    json.dump(results, f, indent=4)
                    
                # Save confusion matrices
                cm_full = build_confusion_matrix(y_true, y_pred_full, num_classes)
                save_confusion_matrix_to_csv(cm_full, res_file.replace(".json", "_full_cm.csv"), label_names)

if __name__ == "__main__":
    main()