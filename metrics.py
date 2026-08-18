import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_piri(acc_full: float, acc_partial: float) -> float:
    if acc_full == 0:
        return 0.0
    return 1.0 - (acc_partial / acc_full)

def compute_chance_corrected_piri(acc_full: float, acc_partial: float, num_classes: int) -> float:
    chance = 1.0 / num_classes
    if acc_full <= chance:
        return 0.0
    return 1.0 - ((acc_partial - chance) / (acc_full - chance))

def compute_all_metrics(labels: list, preds_full: list, preds_prem: list, preds_hyp: list, num_classes: int = 3) -> dict:
    acc_full = accuracy_score(labels, preds_full)
    acc_prem = accuracy_score(labels, preds_prem)
    acc_hyp = accuracy_score(labels, preds_hyp)
    
    macro_f1 = f1_score(labels, preds_full, average='macro')
    
    return {
        "accuracy_full": acc_full,
        "accuracy_premise": acc_prem,
        "accuracy_hypothesis": acc_hyp,
        "macro_f1_full": macro_f1,
        "piri_premise": compute_piri(acc_full, acc_prem),
        "piri_hypothesis": compute_piri(acc_full, acc_hyp),
        "cc_piri_premise": compute_chance_corrected_piri(acc_full, acc_prem, num_classes),
        "cc_piri_hypothesis": compute_chance_corrected_piri(acc_full, acc_hyp, num_classes)
    }