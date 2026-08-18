import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def paired_bootstrap_test(preds1: np.ndarray, preds2: np.ndarray, labels: np.ndarray, num_samples=10000, seed=42):
    """Computes p-value and 95% CI for the difference in accuracy between two models/conditions."""
    np.random.seed(seed)
    n = len(labels)
    diffs = np.zeros(num_samples)
    
    orig_acc1 = np.mean(preds1 == labels)
    orig_acc2 = np.mean(preds2 == labels)
    orig_diff = orig_acc1 - orig_acc2
    
    for i in range(num_samples):
        idx = np.random.choice(n, n, replace=True)
        acc1 = np.mean(preds1[idx] == labels[idx])
        acc2 = np.mean(preds2[idx] == labels[idx])
        diffs[i] = acc1 - acc2
        
    # Two-tailed p-value
    p_val = np.mean(diffs <= 0) if orig_diff > 0 else np.mean(diffs >= 0)
    p_val = min(p_val * 2.0, 1.0)
    
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return p_val, (ci_lower, ci_upper)

def mcnemar_test(preds1: np.ndarray, preds2: np.ndarray, labels: np.ndarray):
    """Computes McNemar's test for paired nominal data."""
    corr1 = (preds1 == labels)
    corr2 = (preds2 == labels)
    
    n00 = np.sum(~corr1 & ~corr2)
    n01 = np.sum(~corr1 & corr2)
    n10 = np.sum(corr1 & ~corr2)
    n11 = np.sum(corr1 & corr2)
    
    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue

def permutation_test(preds1: np.ndarray, preds2: np.ndarray, labels: np.ndarray, num_perms=10000, seed=42):
    """Computes exact p-value using a random permutation test over accuracy differences."""
    np.random.seed(seed)
    n = len(labels)
    
    corr1 = (preds1 == labels)
    corr2 = (preds2 == labels)
    orig_diff = abs(np.mean(corr1) - np.mean(corr2))
    
    count = 0
    for _ in range(num_perms):
        swap = np.random.randint(0, 2, n).astype(bool)
        temp1 = np.where(swap, corr2, corr1)
        temp2 = np.where(swap, corr1, corr2)
        diff = abs(np.mean(temp1) - np.mean(temp2))
        if diff >= orig_diff:
            count += 1
            
    return count / num_perms