import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

def wilson_score_interval(count, nobs, alpha=0.05):
    """Compute Wilson score interval for a binomial proportion."""
    if nobs == 0:
        return 0.0, (0.0, 0.0)
    p = count / nobs
    z = 1.959963984540054  # 95% normal quantile
    denom = 1.0 + z**2 / nobs
    center = (p + (z**2 / (2 * nobs))) / denom
    half_width = (z * np.sqrt((p * (1 - p) / nobs) + (z**2 / (4 * nobs**2)))) / denom
    ci_low = max(0.0, float(center - half_width))
    ci_high = min(1.0, float(center + half_width))
    return float(p), (ci_low, ci_high)

def compute_ece(probs, y_true, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) with n_bins.
    probs: (N, K) class probabilities
    y_true: (N,) true integer class indices (0..K-1)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / len(confidences)) * bin_error
            bin_details.append({
                "bin_idx": i,
                "range": (bin_lower, bin_upper),
                "count": int(bin_count),
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "gap": bin_error
            })
        else:
            bin_details.append({
                "bin_idx": i,
                "range": (bin_lower, bin_upper),
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "gap": 0.0
            })
            
    return float(ece), bin_details

def fit_temperature_scaling(logits_val, y_val_idx):
    """
    Fit temperature parameter T > 0 on validation set logits to minimize NLL.
    logits: (N, K)
    y_val_idx: (N,) int labels
    """
    def nll_obj(T):
        if T <= 1e-4:
            return 1e9
        scaled_logits = logits_val / T
        # log_softmax
        log_probs = scaled_logits - logsumexp(scaled_logits, axis=1, keepdims=True)
        # NLL = - mean log prob of true class
        nll = -np.mean(log_probs[np.arange(len(y_val_idx)), y_val_idx])
        return nll

    res = minimize_scalar(nll_obj, bounds=(0.01, 10.0), method='bounded')
    T_opt = float(res.x)
    return T_opt

def main():
    cache_path = "cache/features_resnet18_dev.npz"
    if not os.path.exists(cache_path):
        cache_path = "cache/features_resnet18.npz"
    print(f"Loading {cache_path}...")
    data = np.load(cache_path, allow_pickle=True)
    X = data["features"]
    cats = data["categories"]
    splits = data["splits"]
    
    classes = sorted(np.unique(cats).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_indices = np.array([class_to_idx[c] for c in cats])
    
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    
    X_train, y_train_idx, cat_train = X[train_mask], y_indices[train_mask], cats[train_mask]
    X_val, y_val_idx, cat_val = X[val_mask], y_indices[val_mask], cats[val_mask]
    X_test, y_test_idx, cat_test = X[test_mask], y_indices[test_mask], cats[test_mask]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train LogisticRegression
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train, cat_train)
    
    # Get decision function (logits) on val and test
    logits_val = clf.decision_function(X_val)
    logits_test = clf.decision_function(X_test)
    
    # Uncalibrated probabilities (T=1.0)
    probs_val_uncal = softmax(logits_val, axis=1)
    probs_test_uncal = softmax(logits_test, axis=1)
    
    ece_test_uncal, _ = compute_ece(probs_test_uncal, y_test_idx)
    print(f"Test ECE before calibration (T=1.0): {ece_test_uncal:.6f}")
    
    # Fit temperature on VAL
    T_opt = fit_temperature_scaling(logits_val, y_val_idx)
    print(f"Optimized Temperature T: {T_opt:.4f}")
    
    # Calibrated probabilities on TEST
    probs_test_cal = softmax(logits_test / T_opt, axis=1)
    ece_test_cal, bin_details_cal = compute_ece(probs_test_cal, y_test_idx)
    print(f"Test ECE after calibration (T={T_opt:.4f}): {ece_test_cal:.6f}")
    
    # Test metrics 5-class
    y_test_pred = np.array(classes)[np.argmax(probs_test_cal, axis=1)]
    acc_5class = accuracy_score(cat_test, y_test_pred)
    f1_macro = f1_score(cat_test, y_test_pred, average="macro")
    print(f"5-Class Accuracy: {acc_5class*100:.2f}%, Macro F1: {f1_macro:.4f}")
    
    # Binary good vs defective
    # Normal is class "normal"
    norm_idx = classes.index("normal")
    p_defective_test = 1.0 - probs_test_cal[:, norm_idx]
    y_test_bin = (cat_test != "normal").astype(int)
    
    # Wilson interval for FRR and FAR at standard 0.5 threshold
    y_pred_bin = (p_defective_test >= 0.5).astype(int)
    cm = confusion_matrix(y_test_bin, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()
    frr, frr_ci = wilson_score_interval(fp, tn + fp)
    far, far_ci = wilson_score_interval(fn, tp + fn)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"FRR: {frr*100:.2f}% (Count: {fp}/{tn+fp}, 95% CI: [{frr_ci[0]*100:.2f}%, {frr_ci[1]*100:.2f}%])")
    print(f"FAR: {far*100:.2f}% (Count: {fn}/{tp+fn}, 95% CI: [{far_ci[0]*100:.2f}%, {far_ci[1]*100:.2f}%])")
    
    # Cost-based reject threshold on VAL
    logits_val_cal = logits_val / T_opt
    probs_val_cal = softmax(logits_val_cal, axis=1)
    p_def_val = 1.0 - probs_val_cal[:, norm_idx]
    y_val_bin = (cat_val != "normal").astype(int)
    
    cost_fr = 41.72
    cost_fa = 5 * cost_fr
    pi = 0.10 # assumed prevalence
    
    # Sweep thresholds on VAL
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh = 0.5
    min_cost_val = float('inf')
    
    val_norm_total = np.sum(y_val_bin == 0)
    val_def_total = np.sum(y_val_bin == 1)
    
    for t in thresholds:
        pred_def = (p_def_val >= t).astype(int)
        fp_v = np.sum((y_val_bin == 0) & (pred_def == 1))
        fn_v = np.sum((y_val_bin == 1) & (pred_def == 0))
        frr_v = fp_v / max(1, val_norm_total)
        far_v = fn_v / max(1, val_def_total)
        
        # Expected cost per 1000 units
        exp_cost = 1000.0 * (cost_fr * (1.0 - pi) * frr_v + cost_fa * pi * far_v)
        if exp_cost < min_cost_val:
            min_cost_val = exp_cost
            best_thresh = t
            
    print(f"Optimal VAL Threshold: {best_thresh:.4f} with Expected Cost on VAL: ${min_cost_val:.2f}/1000 units")
    
    # Evaluate optimal threshold on TEST
    pred_def_test_opt = (p_defective_test >= best_thresh).astype(int)
    test_norm_total = np.sum(y_test_bin == 0)
    test_def_total = np.sum(y_test_bin == 1)
    fp_t = np.sum((y_test_bin == 0) & (pred_def_test_opt == 1))
    fn_t = np.sum((y_test_bin == 1) & (pred_def_test_opt == 0))
    frr_opt, frr_opt_ci = wilson_score_interval(fp_t, test_norm_total)
    far_opt, far_opt_ci = wilson_score_interval(fn_t, test_def_total)
    exp_cost_test = 1000.0 * (cost_fr * (1.0 - pi) * frr_opt + cost_fa * pi * far_opt)
    print(f"TEST with Optimal Threshold ({best_thresh:.4f}):")
    print(f"  FRR: {frr_opt*100:.2f}% ({fp_t}/{test_norm_total}, 95% CI: [{frr_opt_ci[0]*100:.2f}%, {frr_opt_ci[1]*100:.2f}%])")
    print(f"  FAR: {far_opt*100:.2f}% ({fn_t}/{test_def_total}, 95% CI: [{far_opt_ci[0]*100:.2f}%, {far_opt_ci[1]*100:.2f}%])")
    print(f"  Expected Cost on TEST: ${exp_cost_test:.2f} per 1,000 units")

if __name__ == "__main__":
    main()
