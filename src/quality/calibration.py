"""
src/quality/calibration.py
FactoryPulse Statistical Calibration, Uncertainty, and Cost-Based Decision Support.

Conforms to Checkpoint 3 STAGE B ground rules:
1. Multi-class temperature scaling fitted on validation split logits by minimizing NLL with scipy.
2. 10-bin Expected Calibration Error (ECE) before and after calibration evaluated on test split.
3. Wilson score interval (95%) for binomial error rates (FRR and FAR).
4. Cost-based reject threshold optimization on validation split with prevalence reweighting.
5. Coverage-vs-accuracy evaluation for selective classification.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax


def wilson_score_interval(count: int, nobs: int, alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    """
    Compute the Wilson score interval for a binomial proportion.
    Robust for small sample sizes and proportions near 0 or 1.
    
    Args:
        count: Number of successes (e.g., false rejects or false accepts)
        nobs: Total number of trials
        alpha: Significance level (default 0.05 for 95% confidence interval)
        
    Returns:
        Tuple of (point_estimate, (ci_lower, ci_upper))
    """
    if nobs <= 0:
        return 0.0, (0.0, 0.0)
    
    p = float(count / nobs)
    # Standard normal critical value for 95% CI is 1.959963984540054
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-4 else 1.96
    
    denominator = 1.0 + (z ** 2) / nobs
    center = (p + ((z ** 2) / (2.0 * nobs))) / denominator
    half_width = (z * np.sqrt((p * (1.0 - p) / nobs) + ((z ** 2) / (4.0 * (nobs ** 2))))) / denominator
    
    if count == 0:
        ci_lower = 0.0
    else:
        ci_lower = max(0.0, float(center - half_width))
        
    if count == nobs:
        ci_upper = 1.0
    else:
        ci_upper = min(1.0, float(center + half_width))
        
    return p, (ci_lower, ci_upper)


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute Expected Calibration Error (ECE) across n_bins equally spaced confidence bins.
    
    Args:
        probs: (N, K) class probability array
        y_true: (N,) true integer class labels (0 to K-1)
        n_bins: Number of confidence bins (default 10)
        
    Returns:
        Tuple of (ece, bin_details_list)
    """
    if len(probs) == 0 or len(y_true) == 0:
        return 0.0, []
        
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []
    total_samples = len(confidences)
    
    for i in range(n_bins):
        bin_lower = float(bin_boundaries[i])
        bin_upper = float(bin_boundaries[i + 1])
        
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_gap = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * bin_gap
            bin_details.append({
                "bin_idx": i,
                "bin_lower": round(bin_lower, 2),
                "bin_upper": round(bin_upper, 2),
                "bin_center": round((bin_lower + bin_upper) / 2.0, 2),
                "count": bin_count,
                "fraction": round(bin_count / total_samples, 4),
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "gap": round(bin_gap, 4)
            })
        else:
            bin_details.append({
                "bin_idx": i,
                "bin_lower": round(bin_lower, 2),
                "bin_upper": round(bin_upper, 2),
                "bin_center": round((bin_lower + bin_upper) / 2.0, 2),
                "count": 0,
                "fraction": 0.0,
                "accuracy": 0.0,
                "confidence": round((bin_lower + bin_upper) / 2.0, 2),
                "gap": 0.0
            })
            
    return float(ece), bin_details


def fit_temperature_scaling(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Fit scalar temperature parameter T > 0 on validation split logits by minimizing Negative Log-Likelihood (NLL).
    
    Args:
        logits_val: (N_val, K) raw logit scores from classifier decision_function
        y_val: (N_val,) true integer class labels (0 to K-1)
        
    Returns:
        Optimal temperature parameter T_opt
    """
    if len(logits_val) == 0:
        return 1.0
        
    n_samples = len(y_val)
    row_indices = np.arange(n_samples)
    
    def nll_objective(T: float) -> float:
        if T <= 1e-4:
            return 1e9
        scaled = logits_val / T
        # Numerically stable log_softmax
        log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
        # NLL = - 1/N sum_i log(p_{i, y_i})
        nll = -float(np.mean(log_probs[row_indices, y_val]))
        return nll

    # Bounded scalar minimization for temperature in range [0.01, 10.0]
    res = minimize_scalar(nll_objective, bounds=(0.01, 10.0), method="bounded")
    T_opt = float(res.x)
    return max(0.01, T_opt)


def calibrate_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling and softmax normalization to raw logits.
    
    Args:
        logits: (N, K) logit array
        temperature: Scalar temperature parameter T > 0
        
    Returns:
        (N, K) calibrated probability array
    """
    T = max(0.01, float(temperature))
    scaled = logits / T
    return softmax(scaled, axis=1)


def optimize_cost_threshold(
    p_def_val: np.ndarray,
    y_val_binary: np.ndarray,
    cost_fr: float,
    cost_fa: float,
    prevalence: float = 0.10,
    n_grid: int = 200
) -> Dict[str, Any]:
    """
    Select the decision threshold tau on the validation split that minimizes expected cost per 1,000 units,
    reweighting class-wise error rates by assumed real-world prevalence.
    
    Args:
        p_def_val: (N_val,) predicted probability of defect on validation split
        y_val_binary: (N_val,) true binary label (0 = Good/Normal, 1 = Defective)
        cost_fr: Cost of a false reject (e.g. scrap/rework unit cost)
        cost_fa: Cost of an escaped defect (false accept)
        prevalence: Assumed real-world defect prevalence pi (e.g. 0.10 = 10%)
        n_grid: Number of candidate thresholds to sweep
        
    Returns:
        Dict with optimal threshold and expected operational costs
    """
    pi = float(np.clip(prevalence, 0.001, 0.999))
    val_norm_total = int(np.sum(y_val_binary == 0))
    val_def_total = int(np.sum(y_val_binary == 1))
    
    thresholds = np.linspace(0.01, 0.99, n_grid)
    best_thresh = 0.50
    min_cost_val = float("inf")
    best_frr_val = 0.0
    best_far_val = 0.0
    
    for t in thresholds:
        pred_def = (p_def_val >= t).astype(int)
        fp_v = int(np.sum((y_val_binary == 0) & (pred_def == 1)))
        fn_v = int(np.sum((y_val_binary == 1) & (pred_def == 0)))
        
        frr_v = fp_v / max(1, val_norm_total)
        far_v = fn_v / max(1, val_def_total)
        
        # Expected cost per 1,000 units under prevalence pi
        # Expected false rejects = 1000 * (1 - pi) * FRR
        # Expected false accepts = 1000 * pi * FAR
        exp_cost = 1000.0 * (cost_fr * (1.0 - pi) * frr_v + cost_fa * pi * far_v)
        
        if exp_cost < min_cost_val:
            min_cost_val = exp_cost
            best_thresh = float(t)
            best_frr_val = frr_v
            best_far_val = far_v

    return {
        "optimal_threshold": round(best_thresh, 4),
        "val_expected_cost_per_1000": round(min_cost_val, 2),
        "val_frr": round(best_frr_val, 4),
        "val_far": round(best_far_val, 4),
        "assumed_prevalence": pi,
        "cost_false_reject": cost_fr,
        "cost_false_accept": cost_fa
    }


def compute_coverage_accuracy_curve(
    probs: np.ndarray,
    y_true_binary: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute coverage fraction vs accuracy on confident predictions across confidence thresholds.
    
    Args:
        probs: (N, K) class probability array
        y_true_binary: (N,) binary true labels (0 = Normal, 1 = Defective)
        thresholds: Optional array of threshold values
        
    Returns:
        Dict with thresholds, coverage fractions, and confident accuracies
    """
    if thresholds is None:
        thresholds = np.linspace(0.50, 0.99, 50)
        
    confidences = np.max(probs, axis=1)
    predictions_binary = (np.argmax(probs, axis=1) != 2).astype(int) if probs.shape[1] == 5 else (confidences >= 0.5).astype(int)
    # Generic binary check: if index of normal is known, or binary prediction
    # If 5 classes ['crack', 'hole', 'normal', 'rust', 'scratch'], normal is index 2
    
    total = len(y_true_binary)
    coverages = []
    accuracies = []
    
    for t in thresholds:
        confident_mask = confidences >= t
        n_conf = int(np.sum(confident_mask))
        cov = float(n_conf / max(1, total))
        if n_conf > 0:
            acc = float(np.mean(predictions_binary[confident_mask] == y_true_binary[confident_mask]))
        else:
            acc = 1.0
        coverages.append(round(cov, 4))
        accuracies.append(round(acc, 4))
        
    return {
        "thresholds": [round(float(t), 3) for t in thresholds],
        "coverage": coverages,
        "accuracy": accuracies
    }
