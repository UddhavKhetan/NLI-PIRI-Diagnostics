# src/metrics.py

def compute_accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

def compute_piri(acc_full, acc_partial):
    """
    Original PIRI computation.
    """
    if acc_full == 0.0:
        return 0.0
    return 1.0 - (acc_partial / acc_full)

def compute_chance_corrected_piri(acc_full, acc_partial, num_classes):
    """
    Chance-corrected PIRI computation.
    Accounts for random guessing baseline.
    """
    chance = 1.0 / num_classes
    
    # If the model performs worse than or equal to chance, PIRI is undefined or 0.
    if acc_full <= chance:
        return 0.0
        
    numerator = acc_partial - chance
    denominator = acc_full - chance
    
    return 1.0 - (numerator / denominator)

def compute_macro_f1(y_true, y_pred, num_classes):
    """
    Unoptimized, explicit Macro F1 calculation.
    """
    f1_scores = []
    
    for cls in range(num_classes):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for i in range(len(y_true)):
            if y_true[i] == cls and y_pred[i] == cls:
                true_positives += 1
            if y_true[i] != cls and y_pred[i] == cls:
                false_positives += 1
            if y_true[i] == cls and y_pred[i] != cls:
                false_negatives += 1
                
        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)
            
        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)
            
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        f1_scores.append(f1)
        
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1