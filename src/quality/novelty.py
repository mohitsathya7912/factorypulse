"""
src/quality/novelty.py
FactoryPulse Out-of-Distribution Novelty Detection and Held-Out Class Evaluation.

Conforms to Checkpoint 3 STAGE D ground rules:
1. Novelty score = minimum, over the known classes, of the mean cosine distance
   to the k=5 nearest TRAIN features of that class.
2. Threshold = 95th percentile of the novelty scores of known-class VAL images.
3. Separation of Concerns: The PatchCore anomaly score is for good-vs-defective
   and spatial localization; it is NOT used to decide "new defect type".
4. Held-out-class test: For two defect classes ('scratch', then 'rust'):
   - Retrain 4-class classifier without that class.
   - Set 95th-percentile threshold on remaining VAL classes.
   - Report:
     (a) fraction of held-out test images flagged NOVEL or UNCERTAIN.
     (b) false-novel rate on known-class test images (with 95% Wilson CI).
   - Report both honestly.
5. Live Inspection integration: Flags NOVEL / UNKNOWN badge when novelty score > threshold.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

from src.quality.calibration import (
    wilson_score_interval,
    fit_temperature_scaling,
    calibrate_logits
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


class NoveltyDetector:
    """
    Out-of-Distribution Novelty Detector for unseen defect types using
    k-NN cosine distance in ResNet18 feature space.
    
    Formula:
        NoveltyScore(z) = min_{c in C_known} (1/k sum_{j=1}^k d_cos(z, x_{c,(j)}))
    where d_cos(z, x) = 1 - (z . x) / (||z|| * ||x||).
    """

    def __init__(self, k: int = 5):
        self.k = int(k)
        self.known_classes: List[str] = []
        self.class_train_features: Dict[str, np.ndarray] = {}
        self.threshold_95th: float = 0.50
        self.val_scores_mean: float = 0.0
        self.val_scores_max: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, classes: Optional[List[str]] = None) -> "NoveltyDetector":
        """
        Store L2-normalized training features partitioned by known class.
        
        Args:
            X_train: (N_train, D) feature array
            y_train: (N_train,) class label array
            classes: Optional list of class names to consider
        """
        if classes is None:
            classes = sorted(np.unique(y_train).tolist())
        self.known_classes = list(classes)
        self.class_train_features = {}

        for c in self.known_classes:
            mask = y_train == c
            X_c = X_train[mask]
            if len(X_c) == 0:
                continue
            # L2 normalize
            norms = np.linalg.norm(X_c, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self.class_train_features[c] = X_c / norms

        self.is_fitted = True
        return self

    def score(self, X_query: np.ndarray) -> np.ndarray:
        """
        Compute the novelty score for each query feature vector.
        
        Args:
            X_query: (N, D) query feature array
            
        Returns:
            (N,) novelty score array (minimum mean cosine distance to k nearest train features)
        """
        if not self.is_fitted or not self.class_train_features:
            raise RuntimeError("NoveltyDetector must be fitted before scoring.")

        if len(X_query) == 0:
            return np.array([], dtype=np.float32)

        # L2-normalize queries
        norms = np.linalg.norm(X_query, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        Z = X_query / norms

        N = len(Z)
        mean_dists_per_class = []

        for c in self.known_classes:
            X_c_norm = self.class_train_features.get(c)
            if X_c_norm is None or len(X_c_norm) == 0:
                continue

            # Cosine similarity matrix (N, N_c)
            # Dot product between normalized vectors
            sim = np.dot(Z, X_c_norm.T)
            # Cosine distance: d_cos = 1 - sim, bounded in [0, 2]
            dist = 1.0 - sim

            # Find k smallest distances per query
            k_eff = min(self.k, dist.shape[1])
            if k_eff == dist.shape[1]:
                top_k = dist
            else:
                top_k = np.partition(dist, k_eff - 1, axis=1)[:, :k_eff]

            mean_k = np.mean(top_k, axis=1)
            mean_dists_per_class.append(mean_k)

        if not mean_dists_per_class:
            return np.zeros(N, dtype=np.float32)

        # Minimum across all known classes
        stacked = np.stack(mean_dists_per_class, axis=0) # shape (n_classes, N)
        novelty_scores = np.min(stacked, axis=0) # shape (N,)
        return novelty_scores.astype(np.float32)

    def calibrate_threshold(self, X_val: np.ndarray, percentile: float = 95.0) -> float:
        """
        Compute the 95th percentile threshold on known-class validation images.
        """
        scores = self.score(X_val)
        if len(scores) == 0:
            self.threshold_95th = 0.50
        else:
            self.threshold_95th = float(np.percentile(scores, percentile))
            self.val_scores_mean = float(np.mean(scores))
            self.val_scores_max = float(np.max(scores))
        return self.threshold_95th

    def predict_is_novel(self, X_query: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score queries and flag as NOVEL if novelty score exceeds threshold.
        
        Returns:
            Tuple of (scores, is_novel_bool_array)
        """
        thresh = self.threshold_95th if threshold is None else float(threshold)
        scores = self.score(X_query)
        is_novel = scores > thresh
        return scores, is_novel


# =============================================================================
# Held-Out Class Stress Tests Pipeline
# =============================================================================

def run_held_out_class_experiment(
    held_out_class: str,
    orig_X: np.ndarray,
    orig_cats: np.ndarray,
    orig_splits: np.ndarray,
    uncertainty_threshold: float = 0.70
) -> Dict[str, Any]:
    """
    Simulate discovery of an unseen defect category by holding out one class.
    
    Workflow:
    1. Retrain 4-class LogisticRegression without the held-out class on TRAIN split.
    2. Fit NoveltyDetector on the 4 known classes on TRAIN split.
    3. Fit temperature scaling on the 4 known classes on VAL split.
    4. Compute 95th-percentile novelty threshold on the 4 known classes on VAL split.
    5. Evaluate held-out TEST images:
       - Fraction flagged NOVEL
       - Fraction flagged UNCERTAIN
       - Fraction flagged NOVEL or UNCERTAIN
    6. Evaluate known TEST images:
       - False-novel rate with 95% Wilson confidence interval
    """
    train_mask = orig_splits == "train"
    val_mask = orig_splits == "val"
    test_mask = orig_splits == "test"

    # Known classes are all except the held-out class
    all_classes = sorted(np.unique(orig_cats).tolist())
    known_classes = [c for c in all_classes if c != held_out_class]

    # 1. Filter training set to known classes only
    train_known_mask = train_mask & (orig_cats != held_out_class)
    X_train_known = orig_X[train_known_mask]
    y_train_known = orig_cats[train_known_mask]

    # Filter validation set to known classes only
    val_known_mask = val_mask & (orig_cats != held_out_class)
    X_val_known = orig_X[val_known_mask]
    y_val_known = orig_cats[val_known_mask]

    # Train 4-class classifier
    clf_4class = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_4class.fit(X_train_known, y_train_known)
    classes_4 = sorted(clf_4class.classes_.tolist())

    # Calibrate on 4-class VAL logits
    y_val_known_idx = np.array([classes_4.index(c) for c in y_val_known])
    logits_val = clf_4class.decision_function(X_val_known)
    temp_4class = fit_temperature_scaling(logits_val, y_val_known_idx)

    # 2. Fit NoveltyDetector on 4-class TRAIN features
    detector = NoveltyDetector(k=5)
    detector.fit(X_train_known, y_train_known, classes=classes_4)

    # 3. Set 95th-percentile threshold on 4-class VAL features
    threshold_95th = detector.calibrate_threshold(X_val_known, percentile=95.0)

    # 4. Evaluate Held-Out TEST images
    test_heldout_mask = test_mask & (orig_cats == held_out_class)
    X_test_heldout = orig_X[test_heldout_mask]
    n_heldout = len(X_test_heldout)

    scores_heldout = detector.score(X_test_heldout)
    logits_heldout = clf_4class.decision_function(X_test_heldout)
    probs_heldout = calibrate_logits(logits_heldout, temp_4class)
    max_prob_heldout = np.max(probs_heldout, axis=1)

    flagged_novel_heldout = scores_heldout > threshold_95th
    flagged_unc_heldout = max_prob_heldout < uncertainty_threshold
    flagged_combined_heldout = flagged_novel_heldout | flagged_unc_heldout

    n_novel_heldout = int(np.sum(flagged_novel_heldout))
    n_unc_heldout = int(np.sum(flagged_unc_heldout))
    n_comb_heldout = int(np.sum(flagged_combined_heldout))

    frac_novel = float(n_novel_heldout / max(1, n_heldout))
    frac_unc = float(n_unc_heldout / max(1, n_heldout))
    frac_comb = float(n_comb_heldout / max(1, n_heldout))

    # 5. Evaluate Known TEST images (False-Novel Rate)
    test_known_mask = test_mask & (orig_cats != held_out_class)
    X_test_known = orig_X[test_known_mask]
    y_test_known = orig_cats[test_known_mask]
    n_known_test = len(X_test_known)

    scores_known_test = detector.score(X_test_known)
    flagged_novel_known = scores_known_test > threshold_95th
    n_false_novel = int(np.sum(flagged_novel_known))
    false_novel_rate, fnr_ci = wilson_score_interval(n_false_novel, n_known_test)

    # Per-class false novel breakdown on known classes
    known_breakdown = {}
    for c in known_classes:
        mask_c = y_test_known == c
        c_count = int(np.sum(mask_c))
        c_fn = int(np.sum(flagged_novel_known[mask_c]))
        rate_c, ci_c = wilson_score_interval(c_fn, c_count)
        known_breakdown[c] = {
            "total": c_count,
            "false_novel_count": c_fn,
            "false_novel_rate": round(rate_c, 4),
            "false_novel_rate_pct": round(rate_c * 100.0, 2),
            "ci_95": [round(ci_c[0], 4), round(ci_c[1], 4)]
        }

    return {
        "held_out_class": held_out_class,
        "known_classes": known_classes,
        "train_samples_known": len(X_train_known),
        "val_samples_known": len(X_val_known),
        "test_heldout_count": n_heldout,
        "test_known_count": n_known_test,
        "threshold_95th": round(threshold_95th, 4),
        "temperature_4class": round(temp_4class, 4),
        "uncertainty_threshold": uncertainty_threshold,
        # Part (a): Fraction of held-out test images flagged NOVEL or UNCERTAIN
        "fraction_flagged_novel_or_uncertain": round(frac_comb, 4),
        "fraction_flagged_novel_or_uncertain_pct": round(frac_comb * 100.0, 2),
        "fraction_flagged_novel_only": round(frac_novel, 4),
        "fraction_flagged_novel_only_pct": round(frac_novel * 100.0, 2),
        "fraction_flagged_uncertain_only": round(frac_unc, 4),
        "fraction_flagged_uncertain_only_pct": round(frac_unc * 100.0, 2),
        "count_novel_or_uncertain": n_comb_heldout,
        "count_novel_only": n_novel_heldout,
        "count_uncertain_only": n_unc_heldout,
        "mean_novelty_score_heldout": round(float(np.mean(scores_heldout)), 4),
        # Part (b): False-novel rate on known-class test images
        "false_novel_count": n_false_novel,
        "false_novel_rate": round(false_novel_rate, 4),
        "false_novel_rate_pct": round(false_novel_rate * 100.0, 2),
        "false_novel_ci": [round(fnr_ci[0], 4), round(fnr_ci[1], 4)],
        "known_class_breakdown": known_breakdown
    }


def run_all_novelty_experiments(force_recompute: bool = False) -> Dict[str, Any]:
    """
    Run complete STAGE D Novelty Detection experiments:
    1. Held-out class test for 'scratch'
    2. Held-out class test for 'rust'
    3. Production 5-class NoveltyDetector fit on all training classes and calibrated on VAL.
    4. Caches results to cache/novelty_held_out_eval.json and cache/novelty_detector.joblib.
    """
    eval_cache_path = os.path.join(CACHE_DIR, "novelty_held_out_eval.json")
    detector_cache_path = os.path.join(CACHE_DIR, "novelty_detector.joblib")

    if os.path.exists(eval_cache_path) and os.path.exists(detector_cache_path) and not force_recompute:
        try:
            with open(eval_cache_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    print("Loading ResNet18 feature cache for Novelty Detection experiments...")
    feat_data = np.load(os.path.join(CACHE_DIR, "features_resnet18.npz"), allow_pickle=True)
    orig_X = feat_data["features"]
    orig_cats = feat_data["categories"]
    orig_splits = feat_data["splits"]

    # 1. Held-out class test for 'scratch'
    print("Running held-out class experiment: 'scratch'...")
    t0 = time.time()
    exp_scratch = run_held_out_class_experiment(
        held_out_class="scratch",
        orig_X=orig_X,
        orig_cats=orig_cats,
        orig_splits=orig_splits,
        uncertainty_threshold=0.70
    )
    print(f"  Held-out 'scratch': flagged NOVEL or UNCERTAIN = {exp_scratch['fraction_flagged_novel_or_uncertain_pct']}%, False-Novel Rate = {exp_scratch['false_novel_rate_pct']}% ({time.time() - t0:.1f} s)")

    # 2. Held-out class test for 'rust'
    print("Running held-out class experiment: 'rust'...")
    t0 = time.time()
    exp_rust = run_held_out_class_experiment(
        held_out_class="rust",
        orig_X=orig_X,
        orig_cats=orig_cats,
        orig_splits=orig_splits,
        uncertainty_threshold=0.70
    )
    print(f"  Held-out 'rust': flagged NOVEL or UNCERTAIN = {exp_rust['fraction_flagged_novel_or_uncertain_pct']}%, False-Novel Rate = {exp_rust['false_novel_rate_pct']}% ({time.time() - t0:.1f} s)")

    # 3. Production 5-class NoveltyDetector (used in Live Inspection)
    print("Fitting production 5-class NoveltyDetector...")
    train_mask = orig_splits == "train"
    val_mask = orig_splits == "val"
    prod_detector = NoveltyDetector(k=5)
    prod_detector.fit(orig_X[train_mask], orig_cats[train_mask])
    threshold_5class = prod_detector.calibrate_threshold(orig_X[val_mask], percentile=95.0)

    joblib.dump({
        "detector": prod_detector,
        "k": 5,
        "threshold_95th": threshold_5class,
        "known_classes": prod_detector.known_classes,
        "val_scores_mean": prod_detector.val_scores_mean,
        "val_scores_max": prod_detector.val_scores_max
    }, detector_cache_path)
    print(f"Production 5-class NoveltyDetector saved: Threshold tau_novel = {threshold_5class:.4f}")

    results = {
        "scratch_experiment": exp_scratch,
        "rust_experiment": exp_rust,
        "production_detector": {
            "k": 5,
            "known_classes": prod_detector.known_classes,
            "threshold_95th": round(threshold_5class, 4),
            "val_scores_mean": round(prod_detector.val_scores_mean, 4),
            "val_scores_max": round(prod_detector.val_scores_max, 4)
        }
    }

    with open(eval_cache_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def get_production_novelty_detector() -> NoveltyDetector:
    """
    Load or initialize the production 5-class NoveltyDetector from cache.
    """
    detector_cache_path = os.path.join(CACHE_DIR, "novelty_detector.joblib")
    if os.path.exists(detector_cache_path):
        try:
            saved = joblib.load(detector_cache_path)
            return saved["detector"]
        except Exception:
            pass

    # Build on the fly if needed
    run_all_novelty_experiments(force_recompute=False)
    saved = joblib.load(detector_cache_path)
    return saved["detector"]
