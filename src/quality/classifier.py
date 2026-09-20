"""
src/quality/classifier.py
FactoryPulse Two-Stage Quality AI Classifier.

Pipeline:
1. Feature Extraction:
   - Check if data/organizer contains images. If present and torchvision is available,
     extract features with a frozen pretrained CNN (ResNet18 / MobileNet on CPU).
   - If not (Demo mode): train RandomForest on tabular inspection/process features,
     clearly labelled DEMO.
2. Group Splitting:
   - Stratified group split by batch_id to prevent data leakage across batches.
3. Two-Stage Classifier:
   - Stage 1: Binary Good (0) vs Defective (1) with class_weight='balanced'.
   - Stage 2: Multi-class defect category classification on defective units.
4. Evaluation:
   - Strictly computed on held-out test set: Accuracy, Precision (macro + binary),
     Recall (macro + binary), F1 (macro + binary), Confusion Matrix,
     False-Reject Rate (FRR: FP/(TN+FP)), False-Accept Rate (FAR: FN/(TP+FN)).
5. Selective Classification (UNCERTAIN Flag):
   - Predictions with max class probability below configurable threshold are flagged
     as "UNCERTAIN" and routed for human inspector triage.
6. Persistence:
   - Save/load with joblib to avoid unnecessary retraining.
"""

import os
import glob
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from config.schema import is_defective
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from src.quality.calibration import (
    wilson_score_interval,
    compute_ece,
    fit_temperature_scaling,
    calibrate_logits,
    optimize_cost_threshold,
    compute_coverage_accuracy_curve
)

def has_organizer_images(organizer_dir: str = "./data/organizer") -> bool:
    """
    Check if official organizer directory contains image files or class subfolders.
    Returns True immediately upon finding the first valid image or class folder.
    """
    if not os.path.exists(organizer_dir):
        return False
    class_names = {"normal", "crack", "hole", "rust", "scratch"}
    for sub in [organizer_dir, os.path.join(organizer_dir, "train"), os.path.join(organizer_dir, "train", "train")]:
        if os.path.exists(sub):
            try:
                for entry in os.scandir(sub):
                    if entry.is_dir() and entry.name.lower() in class_names:
                        return True
            except Exception:
                pass
    for root, _, files in os.walk(organizer_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                return True

def generate_plain_language_why(
    status: str,
    confidence: float,
    threshold: float,
    top1_cls: str,
    top1_prob: float,
    top2_cls: str,
    top2_prob: float,
    anomaly_score: float,
    anomaly_thresh: float,
    novelty_score: float,
    novelty_thresh: float
) -> str:
    """
    Construct a deterministic plain-language 'why' rationale built strictly from
    computed mathematical outputs (calibrated confidence, top-2 probabilities,
    anomaly score vs threshold, novelty score vs threshold).

    Strict ground rule adherence: NO LLM. Built ONLY from computed numbers.
    """
    if status == "NOVEL" or (novelty_thresh > 0 and novelty_score > novelty_thresh):
        return (
            f"NOVEL / UNKNOWN: Novelty score {novelty_score:.3f} exceeds threshold {novelty_thresh:.3f} "
            f"(k=5 nearest-neighbor train distance). Feature embedding is out-of-distribution across all 5 known classes. "
            f"Triage recommended for unseen defect characterization."
        )
    elif status == "UNCERTAIN" or confidence < threshold:
        return (
            f"UNCERTAIN: Calibrated confidence {confidence:.2f} ({top1_cls}) is below threshold {threshold:.2f}. "
            f"Top-2 competing classes: {top1_cls} ({top1_prob*100:.1f}%) and {top2_cls} ({top2_prob*100:.1f}%). "
            f"Prediction is ambiguous; routed to human inspector triage queue."
        )
    elif status == "DEFECTIVE":
        anom_clause = (
            f"patch-anomaly score {anomaly_score:.2f} exceeds normal limit {anomaly_thresh:.2f}"
            if anomaly_score > anomaly_thresh
            else f"patch-anomaly score {anomaly_score:.2f} is within envelope ({anomaly_thresh:.2f})"
        )
        return (
            f"DEFECTIVE ({top1_cls.upper()}): Calibrated confidence {confidence:.2f} meets or exceeds threshold {threshold:.2f} "
            f"(top-2 classes: {top1_cls} {top1_prob*100:.1f}%, {top2_cls} {top2_prob*100:.1f}%); "
            f"{anom_clause}."
        )
    else:  # GOOD
        anom_clause = (
            f"patch-anomaly score {anomaly_score:.2f} is within normal threshold {anomaly_thresh:.2f}"
            if anomaly_score <= anomaly_thresh
            else f"minor localized anomaly {anomaly_score:.2f} vs limit {anomaly_thresh:.2f}"
        )
        return (
            f"GOOD (CONFORMING): Calibrated confidence {confidence:.2f} meets or exceeds threshold {threshold:.2f} "
            f"(top-2 classes: normal {top1_prob*100:.1f}%, {top2_cls} {top2_prob*100:.1f}%); "
            f"{anom_clause}."
        )


class QualityClassifier:
    """
    Two-stage manufacturing defect classifier and uncertainty triage system.
    Supports image feature extraction (when images exist) and tabular telemetry features.
    """

    def __init__(self, model_dir: str = "./models", organizer_dir: str = "./data/organizer"):
        # Robust path resolution to repo root if current working directory differs
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if not os.path.isabs(model_dir) and not os.path.exists(model_dir):
            candidate = os.path.join(root_dir, model_dir.lstrip("./").lstrip(".\\"))
            if os.path.exists(candidate) or os.path.exists(os.path.dirname(candidate)):
                model_dir = candidate
        if not os.path.isabs(organizer_dir) and not os.path.exists(organizer_dir):
            candidate = os.path.join(root_dir, organizer_dir.lstrip("./").lstrip(".\\"))
            if os.path.exists(candidate) or os.path.exists(os.path.dirname(candidate)):
                organizer_dir = candidate

        self.model_dir = model_dir
        self.organizer_dir = organizer_dir
        self.model_path = os.path.join(model_dir, "quality_model.joblib")
        
        self.binary_clf: Optional[RandomForestClassifier] = None
        self.family_clf: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.is_image_based: bool = False
        self.data_mode: str = "Demo"
        self.test_metrics: Dict[str, Any] = {}
        self.test_indices: List[int] = []
        self.test_batches: List[str] = []
        self.uncertainty_threshold: float = 0.70
        self.stage2_classes: List[str] = []
        self.clf_5class: Optional[Any] = None
        self.classes_5class: List[str] = ["crack", "hole", "normal", "rust", "scratch"]

    def get_organizer_resnet18_features(
        self,
        use_full_dataset: bool = False,
        dev_mode: Optional[bool] = None,
        progress_bar=None,
        status_text=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features with frozen ResNet18 (CPU) and cache them to disk under cache/.
        Supports:
        - Dev mode (400 images per class, 2,000 total) -> cache/features_resnet18_dev.npz
        - Full dataset (all 12,000 images) -> cache/features_resnet18.npz
        
        Returns:
            Tuple of (X, y_binary, defect_categories, image_paths, splits)
        """
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cache_dir = os.path.join(root_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        if dev_mode is None:
            dev_mode = not use_full_dataset

        full_cache_path = os.path.join(cache_dir, "features_resnet18.npz")
        dev_cache_path = os.path.join(cache_dir, "features_resnet18_dev.npz")

        # 1. If dev_mode is requested
        if dev_mode:
            if os.path.exists(dev_cache_path):
                data = np.load(dev_cache_path, allow_pickle=True)
                X = data["features"]
                cats = data["categories"]
                paths = data["paths"]
                splits = data["splits"]
                labels = data["labels"] if "labels" in data else np.array(["good" if c == "normal" else "defective" for c in cats])
                y_binary = (labels == "defective").astype(int)
                return X, y_binary, cats, paths, splits

            # If full cache exists, slice dev mode from it
            if os.path.exists(full_cache_path):
                data = np.load(full_cache_path, allow_pickle=True)
                full_X = data["features"]
                full_cats = data["categories"]
                full_paths = data["paths"]
                full_splits = data["splits"]
                
                # Take 240 train, 80 val, 80 test per category
                dev_indices = []
                for c in ["crack", "hole", "normal", "rust", "scratch"]:
                    mask_c = full_cats == c
                    idx_train = np.where(mask_c & (full_splits == "train"))[0][:240]
                    idx_val = np.where(mask_c & (full_splits == "val"))[0][:80]
                    idx_test = np.where(mask_c & (full_splits == "test"))[0][:80]
                    dev_indices.extend(idx_train)
                    dev_indices.extend(idx_val)
                    dev_indices.extend(idx_test)

                dev_indices = np.array(dev_indices)
                X = full_X[dev_indices]
                cats = full_cats[dev_indices]
                paths = full_paths[dev_indices]
                splits = full_splits[dev_indices]
                labels = np.array(["good" if c == "normal" else "defective" for c in cats])
                y_binary = (labels == "defective").astype(int)

                np.savez_compressed(
                    dev_cache_path,
                    features=X,
                    paths=paths,
                    labels=labels,
                    categories=cats,
                    splits=splits
                )
                return X, y_binary, cats, paths, splits

        # 2. If full dataset is requested
        if os.path.exists(full_cache_path):
            data = np.load(full_cache_path, allow_pickle=True)
            X = data["features"]
            cats = data["categories"]
            paths = data["paths"]
            splits = data["splits"]
            labels = data["labels"] if "labels" in data else np.array(["good" if c == "normal" else "defective" for c in cats])
            y_binary = (labels == "defective").astype(int)
            return X, y_binary, cats, paths, splits

        # 3. If full cache does not exist yet, extract using resumable extractor
        from src.quality.extract_features import extract_features
        extract_features(
            batch_size=64,
            cache_dir=cache_dir,
            data_dir=os.path.join(self.organizer_dir, "train"),
            progress_callback=lambda p: progress_bar.progress(p) if progress_bar else None,
            status_callback=lambda msg: status_text.text(msg) if status_text else None
        )
        data = np.load(full_cache_path, allow_pickle=True)
        X = data["features"]
        cats = data["categories"]
        paths = data["paths"]
        splits = data["splits"]
        labels = data["labels"]
        y_binary = (labels == "defective").astype(int)

        if dev_mode:
            return self.get_organizer_resnet18_features(use_full_dataset=False, dev_mode=True)
        return X, y_binary, cats, paths, splits

    def train_organizer_resnet18(
        self,
        use_full_dataset: bool = False,
        dev_mode: Optional[bool] = None,
        uncertainty_threshold: float = 0.70,
        cost_false_reject: float = 41.72,
        cost_escaped_defect_mult: float = 5.0,
        assumed_prevalence: float = 0.10,
        force_retrain: bool = False,
        progress_bar=None,
        status_text=None
    ) -> Dict[str, Any]:
        """
        Train class-balanced LogisticRegression on train features (5 classes) with stratified split.
        Save model to models/ via joblib. Do not retrain on every rerun.
        Test set is used ONLY for final reporting, never for training, tuning, or calibration.
        Validation split is used strictly for temperature scaling calibration and cost threshold tuning.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report, roc_curve, auc,
            precision_recall_curve
        )
        from scipy.special import softmax

        self.uncertainty_threshold = float(uncertainty_threshold)
        if dev_mode is None:
            dev_mode = not use_full_dataset

        # Load features and split metadata
        X, y_binary, cats, paths, splits = self.get_organizer_resnet18_features(
            use_full_dataset=(not dev_mode),
            dev_mode=dev_mode,
            progress_bar=progress_bar,
            status_text=status_text
        )

        train_mask = splits == "train"
        val_mask = splits == "val"
        test_mask = splits == "test"

        X_train, cat_train, path_train = X[train_mask], cats[train_mask], paths[train_mask]
        X_val, cat_val, path_val = X[val_mask], cats[val_mask], paths[val_mask]
        X_test, cat_test, path_test = X[test_mask], cats[test_mask], paths[test_mask]

        model_name = "organizer_5class_dev.joblib" if dev_mode else "organizer_5class_logistic_regression.joblib"
        model_path = os.path.join(self.model_dir, model_name)
        os.makedirs(self.model_dir, exist_ok=True)

        # Do not retrain on every rerun if model exists and force_retrain is False
        loaded_from_disk = False
        if os.path.exists(model_path) and not force_retrain:
            try:
                saved = joblib.load(model_path)
                self.clf_5class = saved["clf_5class"]
                self.classes_5class = saved.get("classes", sorted(np.unique(cats).tolist()))
                self.temperature = float(saved.get("temperature", 1.0))
                loaded_from_disk = True
            except Exception:
                loaded_from_disk = False

        if not loaded_from_disk:
            if status_text:
                status_text.text("Training class-balanced 5-class LogisticRegression on train split...")
            self.clf_5class = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            self.clf_5class.fit(X_train, cat_train)
            self.classes_5class = sorted(self.clf_5class.classes_.tolist())

            # Fit temperature scaling on validation logits strictly
            y_val_idx = np.array([self.classes_5class.index(c) for c in cat_val])
            logits_val = self.clf_5class.decision_function(X_val)
            self.temperature = fit_temperature_scaling(logits_val, y_val_idx)

            joblib.dump({
                "clf_5class": self.clf_5class,
                "classes": self.classes_5class,
                "temperature": self.temperature,
                "uncertainty_threshold": self.uncertainty_threshold,
                "dev_mode": dev_mode,
                "train_samples": int(len(X_train)),
                "val_samples": int(len(X_val)),
                "test_samples": int(len(X_test))
            }, model_path)

        # Ensure temperature is valid
        if not hasattr(self, "temperature") or self.temperature <= 0:
            y_val_idx = np.array([self.classes_5class.index(c) for c in cat_val])
            logits_val = self.clf_5class.decision_function(X_val)
            self.temperature = fit_temperature_scaling(logits_val, y_val_idx)

        # Compute raw logits on val and test
        logits_val = self.clf_5class.decision_function(X_val)
        logits_test = self.clf_5class.decision_function(X_test)
        y_val_idx = np.array([self.classes_5class.index(c) for c in cat_val])
        y_test_idx = np.array([self.classes_5class.index(c) for c in cat_test])

        # Stage B.2: Calibration - compute ECE before (T=1.0) and after (T=temperature) on TEST
        probs_test_uncal = softmax(logits_test, axis=1)
        probs_test = calibrate_logits(logits_test, self.temperature)

        ece_uncal, bin_details_uncal = compute_ece(probs_test_uncal, y_test_idx, n_bins=10)
        ece_cal, bin_details_cal = compute_ece(probs_test, y_test_idx, n_bins=10)

        # Evaluate strictly on held-out test split (used ONLY for final reporting)
        max_prob_test = np.max(probs_test, axis=1)
        pred_indices = np.argmax(probs_test, axis=1)
        y_test_pred = np.array(self.classes_5class)[pred_indices]

        # Stage B.1: 5-class metrics, per-class, and macro averages
        acc_5class = float(accuracy_score(cat_test, y_test_pred))
        f1_macro_5class = float(f1_score(cat_test, y_test_pred, average="macro", zero_division=0))
        prec_macro_5class = float(precision_score(cat_test, y_test_pred, average="macro", zero_division=0))
        rec_macro_5class = float(recall_score(cat_test, y_test_pred, average="macro", zero_division=0))
        cm_5class = confusion_matrix(cat_test, y_test_pred, labels=self.classes_5class)

        # Per-class metrics
        cls_rep = classification_report(cat_test, y_test_pred, labels=self.classes_5class, output_dict=True, zero_division=0)
        per_class_metrics = {}
        for c in self.classes_5class:
            if c in cls_rep:
                per_class_metrics[c] = {
                    "precision": round(float(cls_rep[c]["precision"]), 4),
                    "recall": round(float(cls_rep[c]["recall"]), 4),
                    "f1": round(float(cls_rep[c]["f1-score"]), 4),
                    "support": int(cls_rep[c]["support"])
                }

        # Stage B.1: Binary good-vs-defective: P(defective) = 1 - P(normal)
        idx_normal = self.classes_5class.index("normal")
        p_normal_test = probs_test[:, idx_normal]
        p_def_test = 1.0 - p_normal_test

        y_test_bin = (cat_test != "normal").astype(int)
        y_pred_bin = (p_def_test >= 0.50).astype(int)

        acc = float(accuracy_score(y_test_bin, y_pred_bin))
        prec = float(precision_score(y_test_bin, y_pred_bin, zero_division=0))
        rec = float(recall_score(y_test_bin, y_pred_bin, zero_division=0))
        f1 = float(f1_score(y_test_bin, y_pred_bin, zero_division=0))
        f1_macro = float(f1_score(y_test_bin, y_pred_bin, average="macro", zero_division=0))
        cm = confusion_matrix(y_test_bin, y_pred_bin)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        # Exact sample counts and 95% Wilson intervals for FRR and FAR
        frr, frr_ci = wilson_score_interval(fp, tn + fp)
        far, far_ci = wilson_score_interval(fn, tp + fn)

        # ROC and PR curves with AUC
        fpr, tpr, _ = roc_curve(y_test_bin, p_def_test)
        roc_auc = float(auc(fpr, tpr))
        pr_precision, pr_recall, _ = precision_recall_curve(y_test_bin, p_def_test)
        pr_auc = float(auc(pr_recall, pr_precision))

        fpr_sampled = [round(float(v), 4) for v in fpr[::max(1, len(fpr)//50)]]
        tpr_sampled = [round(float(v), 4) for v in tpr[::max(1, len(tpr)//50)]]
        pr_rec_sampled = [round(float(v), 4) for v in pr_recall[::max(1, len(pr_recall)//50)]]
        pr_prec_sampled = [round(float(v), 4) for v in pr_precision[::max(1, len(pr_precision)//50)]]

        # Stage B.3: Selective classification (UNCERTAIN flag)
        uncertain_mask = max_prob_test < self.uncertainty_threshold
        confident_mask = ~uncertain_mask
        unc_cnt = int(np.sum(uncertain_mask))
        unc_frac = float(unc_cnt / max(1, len(cat_test)))
        conf_acc = float(accuracy_score(y_test_bin[confident_mask], y_pred_bin[confident_mask])) if np.sum(confident_mask) > 0 else 0.0

        # Coverage-vs-accuracy curve data across thresholds
        cov_acc_curve = compute_coverage_accuracy_curve(probs_test, y_test_bin)

        # Stage B.4: Cost-based reject threshold optimized on VAL split
        probs_val_cal = calibrate_logits(logits_val, self.temperature)
        p_def_val = 1.0 - probs_val_cal[:, idx_normal]
        y_val_bin = (cat_val != "normal").astype(int)
        cost_fa = float(cost_false_reject * cost_escaped_defect_mult)

        cost_opt = optimize_cost_threshold(
            p_def_val=p_def_val,
            y_val_binary=y_val_bin,
            cost_fr=cost_false_reject,
            cost_fa=cost_fa,
            prevalence=assumed_prevalence
        )

        # Evaluate cost-optimal operating point on TEST split
        tau_opt = cost_opt["optimal_threshold"]
        pred_def_opt = (p_def_test >= tau_opt).astype(int)
        fp_opt = int(np.sum((y_test_bin == 0) & (pred_def_opt == 1)))
        fn_opt = int(np.sum((y_test_bin == 1) & (pred_def_opt == 0)))
        test_norm_total = max(1, tn + fp)
        test_def_total = max(1, tp + fn)
        frr_opt, frr_opt_ci = wilson_score_interval(fp_opt, test_norm_total)
        far_opt, far_opt_ci = wilson_score_interval(fn_opt, test_def_total)
        exp_cost_test = 1000.0 * (cost_false_reject * (1.0 - assumed_prevalence) * frr_opt + cost_fa * assumed_prevalence * far_opt)

        cost_operating_point = {
            "optimal_threshold": tau_opt,
            "val_expected_cost_per_1000": cost_opt["val_expected_cost_per_1000"],
            "test_expected_cost_per_1000": round(exp_cost_test, 2),
            "test_frr": round(frr_opt, 4),
            "test_frr_pct": round(frr_opt * 100.0, 2),
            "test_frr_ci": [round(frr_opt_ci[0], 4), round(frr_opt_ci[1], 4)],
            "test_far": round(far_opt, 4),
            "test_far_pct": round(far_opt * 100.0, 2),
            "test_far_ci": [round(far_opt_ci[0], 4), round(far_opt_ci[1], 4)],
            "test_fp_count": fp_opt,
            "test_fn_count": fn_opt,
            "expected_false_rejects_per_1000": round(1000.0 * (1.0 - assumed_prevalence) * frr_opt, 2),
            "expected_escaped_defects_per_1000": round(1000.0 * assumed_prevalence * far_opt, 2),
            "assumed_prevalence": assumed_prevalence,
            "assumed_prevalence_pct": round(assumed_prevalence * 100.0, 1),
            "cost_false_reject": cost_false_reject,
            "cost_escaped_defect": cost_fa,
            "cost_escaped_defect_mult": cost_escaped_defect_mult
        }

        # Stage 2 Category metrics on defective test units
        mask_test_def = cat_test != "normal"
        s2_classes = ["crack", "hole", "rust", "scratch"]
        s2_cm = confusion_matrix(cat_test[mask_test_def], y_test_pred[mask_test_def], labels=s2_classes)
        s2_acc = float(accuracy_score(cat_test[mask_test_def], y_test_pred[mask_test_def]))
        s2_f1 = float(f1_score(cat_test[mask_test_def], y_test_pred[mask_test_def], average="macro", zero_division=0))

        # Build test predictions DataFrame for inspection triage queue & audit
        test_pred_rows = []
        for i in range(len(cat_test)):
            p_conf = float(max_prob_test[i])
            is_unc = bool(uncertain_mask[i])
            act_c = cat_test[i]
            pred_c = y_test_pred[i]
            pred_lbl = "defective" if pred_c != "normal" else "good"
            act_lbl = "defective" if act_c != "normal" else "good"
            status = "UNCERTAIN" if is_unc else ("REJECT" if pred_c != "normal" else "ACCEPT")
            rec_action = (
                "Route to Human Inspector (Borderline Confidence)" if is_unc
                else ("Direct Scrappage / Rework Line" if pred_c != "normal" else "Direct Shipment (Conforming)")
            )
            test_pred_rows.append({
                "unit_id": f"TEST_{i+1:04d}",
                "image_path": path_test[i],
                "actual_label": act_lbl,
                "actual_category": act_c,
                "defect_category": act_c,
                "predicted_category": pred_c,
                "confidence": round(p_conf, 4),
                "confidence_pct": round(p_conf * 100.0, 1),
                "is_uncertain": is_unc,
                "predicted_status": status,
                "prediction": pred_lbl,
                "triage_recommendation": rec_action,
                "p_defective": round(float(p_def_test[i]), 4)
            })
        test_pred_df = pd.DataFrame(test_pred_rows)

        self.test_metrics = {
            "status": "trained",
            "feature_type": "ResNet18 (Frozen Pretrained CNN on CPU, 512-dim embedding)",
            "is_image_based": True,
            "data_mode": "Organizer",
            "is_synthetic": False,
            "dev_mode": dev_mode,
            "total_samples": len(X),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "use_full_dataset": (not dev_mode),
            "temperature": round(self.temperature, 4),
            "ece_uncalibrated": round(ece_uncal, 6),
            "ece_calibrated": round(ece_cal, 6),
            "reliability_diagram_bins": bin_details_cal,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "f1_macro": round(f1_macro, 4),
            "accuracy_5class": round(acc_5class, 4),
            "f1_macro_5class": round(f1_macro_5class, 4),
            "precision_macro_5class": round(prec_macro_5class, 4),
            "recall_macro_5class": round(rec_macro_5class, 4),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "confusion_matrix_5class": cm_5class.tolist(),
            "classes_5class": self.classes_5class,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "false_reject_count": fp,
            "false_accept_count": fn,
            "false_reject_total": tn + fp,
            "false_accept_total": tp + fn,
            "false_reject_rate": round(frr, 4),
            "false_reject_rate_pct": round(frr * 100.0, 2),
            "false_reject_ci": [round(frr_ci[0], 4), round(frr_ci[1], 4)],
            "false_accept_rate": round(far, 4),
            "false_accept_rate_pct": round(far * 100.0, 2),
            "false_accept_ci": [round(far_ci[0], 4), round(far_ci[1], 4)],
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "roc_fpr": fpr_sampled,
            "roc_tpr": tpr_sampled,
            "pr_recall": pr_rec_sampled,
            "pr_precision": pr_prec_sampled,
            "uncertainty_threshold": self.uncertainty_threshold,
            "uncertain_count": unc_cnt,
            "uncertain_fraction": round(unc_frac, 4),
            "uncertain_pct": round(unc_frac * 100.0, 1),
            "confident_accuracy": round(conf_acc, 4),
            "confident_accuracy_pct": round(conf_acc * 100.0, 1),
            "coverage_accuracy_curve": cov_acc_curve,
            "cost_operating_point": cost_operating_point,
            "stage2": {
                "available": True,
                "classes": s2_classes,
                "accuracy": round(s2_acc, 4),
                "f1_macro": round(s2_f1, 4),
                "confusion_matrix": s2_cm.tolist(),
                "test_sample_count": int(np.sum(mask_test_def))
            },
            "test_predictions": test_pred_df
        }
        return self.test_metrics

    def load_organizer_models(self) -> bool:
        """Load persisted 5-class LogisticRegression and ResNet18 linear classifiers from disk."""
        model_paths = [
            os.path.join(self.model_dir, "organizer_5class_logistic_regression.joblib"),
            os.path.join(self.model_dir, "organizer_5class_dev.joblib"),
            os.path.join(self.model_dir, "organizer_resnet18_models.joblib")
        ]
        for p in model_paths:
            if os.path.exists(p):
                try:
                    data = joblib.load(p)
                    if "clf_5class" in data:
                        self.clf_5class = data["clf_5class"]
                        self.classes_5class = data.get("classes", sorted(self.clf_5class.classes_.tolist()))
                        self.temperature = float(data.get("temperature", 1.0))
                    if "binary_clf" in data:
                        self.binary_clf = data.get("binary_clf")
                        self.family_clf = data.get("family_clf")
                        self.stage2_classes = data.get("stage2_classes", [])
                    self.uncertainty_threshold = data.get("uncertainty_threshold", 0.70)
                    return True
                except Exception:
                    continue
        return False

    def _get_resnet_components(self):
        """Retrieve or initialize cached ResNet18 model and preprocessing pipeline on CPU."""
        if not hasattr(self, "_resnet_model") or self._resnet_model is None:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval()
            self._resnet_model = model

            self._resnet_preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return self._resnet_model, self._resnet_preprocess

    def predict_image(
        self,
        image_source: Any,
        uncertainty_threshold: Optional[float] = None,
        compute_localization: bool = True
    ) -> Dict[str, Any]:
        """
        Run live inference on a single image (path, bytes, UploadedFile, or PIL Image)
        using frozen ResNet18 feature extractor + calibrated 5-class classifier + patch localization.
        """
        import torch
        from PIL import Image
        import io

        if self.clf_5class is None:
            if not self.load_organizer_models():
                self.train_organizer_resnet18(dev_mode=True)

        thresh = float(uncertainty_threshold if uncertainty_threshold is not None else self.uncertainty_threshold)

        # 1. Open image
        if isinstance(image_source, str):
            img = Image.open(image_source)
        elif isinstance(image_source, bytes):
            img = Image.open(io.BytesIO(image_source))
        elif hasattr(image_source, "read"):
            img = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            img = image_source
        else:
            raise ValueError(f"Unsupported image input type: {type(image_source)}")

        if img.mode != "RGB":
            img = img.convert("RGB")
        orig_img = img.copy()

        # 2. Extract feature
        model, preprocess = self._get_resnet_components()
        with torch.no_grad():
            tensor = preprocess(img).unsqueeze(0)
            feat = model(tensor).squeeze().numpy().reshape(1, -1)

        # 3. 5-Class Calibrated Prediction
        logits = self.clf_5class.decision_function(feat)
        temp = getattr(self, "temperature", 1.0)
        probs = calibrate_logits(logits, temp)[0]
        classes = list(self.clf_5class.classes_)
        max_prob = float(np.max(probs))
        pred_class = str(classes[np.argmax(probs)])

        idx_normal = classes.index("normal") if "normal" in classes else -1
        p_normal = float(probs[idx_normal]) if idx_normal >= 0 else 0.0
        p_good = p_normal
        p_def = 1.0 - p_normal

        # 4. Out-of-Distribution Novelty Check (Stage D Requirement 5: NOVEL / UNKNOWN blue badge)
        is_novel = False
        novelty_score = 0.0
        novelty_threshold = 0.1188
        try:
            if not hasattr(self, "novelty_detector") or self.novelty_detector is None:
                from src.quality.novelty import get_production_novelty_detector
                self.novelty_detector = get_production_novelty_detector()
            if self.novelty_detector is not None:
                novelty_score = float(self.novelty_detector.score(feat)[0])
                novelty_threshold = float(self.novelty_detector.threshold_95th)
                is_novel = bool(novelty_score > novelty_threshold)
        except Exception:
            pass

        # 5. Uncertainty & Badge Status (Consistent colors: Green Good, Red Defective, Amber Uncertain, Blue Novel)
        is_uncertain = max_prob < thresh

        if is_novel:
            status = "NOVEL"
            badge_color = "#2563EB"  # Blue Novel
            badge_text = "NOVEL / UNKNOWN"
            category_text = f"Unseen / Novel Defect (Score: {novelty_score:.3f} > {novelty_threshold:.3f})"
        elif is_uncertain:
            status = "UNCERTAIN"
            badge_color = "#F59E0B"  # Amber Uncertain
            badge_text = "UNCERTAIN"
            category_text = f"Borderline ({pred_class.capitalize()})" if pred_class != "normal" else "Borderline (Good)"
        elif pred_class == "normal":
            status = "GOOD"
            badge_color = "#10B981"  # Green Good
            badge_text = "GOOD"
            category_text = "Conforming (Normal)"
        else:
            status = "DEFECTIVE"
            badge_color = "#EF4444"  # Red Defective
            badge_text = f"DEFECTIVE ({pred_class.upper()})"
            category_text = pred_class.capitalize()

        # 6. Patch-Anomaly Localization
        anomaly_info = None
        if compute_localization:
            try:
                if not hasattr(self, "anomaly_detector") or self.anomaly_detector is None:
                    from src.quality.localization import PatchAnomalyDetector
                    self.anomaly_detector = PatchAnomalyDetector()
                anomaly_info = self.anomaly_detector.score_image(orig_img, compute_overlay=True)
            except Exception as e:
                anomaly_info = {"error": str(e), "anomaly_score": 0.0, "threshold": 2.0605}

        # 7. Top-2 Class Probabilities & Plain-Language 'Why' (Stage F Requirement 1 - No LLM)
        class_prob_pairs = sorted(zip(classes, [float(p) for p in probs]), key=lambda x: x[1], reverse=True)
        top1_cls, top1_prob = class_prob_pairs[0] if len(class_prob_pairs) > 0 else (pred_class, max_prob)
        top2_cls, top2_prob = class_prob_pairs[1] if len(class_prob_pairs) > 1 else ("none", 0.0)

        anom_score = float(anomaly_info.get("anomaly_score", 0.0)) if anomaly_info else 0.0
        anom_thresh = float(anomaly_info.get("threshold", 2.0605)) if anomaly_info else 2.0605

        why_text = generate_plain_language_why(
            status=status,
            confidence=max_prob,
            threshold=thresh,
            top1_cls=top1_cls,
            top1_prob=top1_prob,
            top2_cls=top2_cls,
            top2_prob=top2_prob,
            anomaly_score=anom_score,
            anomaly_thresh=anom_thresh,
            novelty_score=novelty_score,
            novelty_thresh=novelty_threshold
        )

        return {
            "status": status,
            "badge_label": badge_text,
            "badge_color": badge_color,
            "category_text": category_text,
            "is_defective": (pred_class != "normal"),
            "is_uncertain": is_uncertain,
            "is_novel": is_novel,
            "novelty_score": round(novelty_score, 4),
            "novelty_threshold": round(novelty_threshold, 4),
            "confidence": max_prob,
            "confidence_pct": round(max_prob * 100.0, 1),
            "defect_category": pred_class,
            "prob_good": p_good,
            "prob_defective": p_def,
            "probabilities": dict(zip(classes, [float(p) for p in probs])),
            "top2_probabilities": class_prob_pairs[:2],
            "top1_class": top1_cls,
            "top1_prob": round(top1_prob, 4),
            "top2_class": top2_cls,
            "top2_prob": round(top2_prob, 4),
            "threshold": thresh,
            "anomaly_info": anomaly_info,
            "anomaly_score": round(anom_score, 3),
            "anomaly_threshold": round(anom_thresh, 3),
            "plain_language_why": why_text
        }

    def _extract_cnn_features(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """
        Extract feature embeddings from images using frozen pretrained ResNet18 (CPU).
        Falls back to None if torchvision/torch is not installed or images are unreadable.
        """
        try:
            import torch
            import torchvision.transforms as transforms
            import torchvision.models as models
            from PIL import Image

            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval()

            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            features = []
            with torch.no_grad():
                for p in image_paths:
                    if os.path.exists(p):
                        try:
                            img = Image.open(p)
                            tensor = preprocess(img).unsqueeze(0)
                            feat = model(tensor).squeeze().numpy()
                            features.append(feat)
                        except Exception:
                            features.append(np.zeros(512, dtype=np.float32))
                    else:
                        features.append(np.zeros(512, dtype=np.float32))

            return np.array(features, dtype=np.float32)
        except Exception:
            return None

    def _extract_features(
        self,
        df_insp: pd.DataFrame,
        df_proc: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, List[str], bool]:
        """
        Extract representation for quality classification.
        
        Priority:
        1. If organizer has images and dataset is pure image dataset (no tabular measurements) -> CNN feature vector.
        2. Else (Demo mode / tabular inspection) -> Tabular features: process parameters,
           measurements (roughness_ra, dimension_delta_mm, optical_score), variant encoding.
        """
        has_images = has_organizer_images(self.organizer_dir)
        has_tabular_measurements = any(c in df_insp.columns for c in ["roughness_ra", "dimension_delta_mm", "optical_score"])
        
        # 1. Image feature extraction if available and not demo/tabular mode
        if has_images and "image_path" in df_insp.columns and not has_tabular_measurements and self.data_mode != "Demo":
            img_paths = df_insp["image_path"].tolist()
            cnn_feats = self._extract_cnn_features(img_paths)
            if cnn_feats is not None and cnn_feats.shape[0] == len(df_insp):
                names = [f"cnn_feat_{i}" for i in range(cnn_feats.shape[1])]
                return cnn_feats, names, True

        # 2. Tabular Feature Extraction (Demo Mode or Tabular Organizer Data)
        df_merged = df_insp.copy()

        # Join process telemetry by batch_id if df_proc is supplied
        if df_proc is not None and "batch_id" in df_merged.columns and "batch_id" in df_proc.columns:
            proc_cols = [c for c in ["temperature", "speed", "pressure", "vibration"] if c in df_proc.columns]
            if proc_cols:
                # Group process metrics by batch_id to get mean telemetry per batch
                proc_agg = df_proc.groupby("batch_id")[proc_cols].mean().reset_index()
                df_merged = df_merged.merge(proc_agg, on="batch_id", how="left")

        features_list = []
        feature_names = []

        # Process telemetry features
        telemetry_cols = ["temperature", "speed", "pressure", "vibration"]
        for col in telemetry_cols:
            if col in df_merged.columns:
                series = pd.to_numeric(df_merged[col], errors="coerce")
                median_v = series.median() if not np.isnan(series.median()) else 0.0
                features_list.append(series.fillna(median_v).values)
                feature_names.append(col)
            else:
                # Fill neutral defaults if process table not joined
                features_list.append(np.zeros(len(df_merged), dtype=np.float32))
                feature_names.append(f"default_{col}")

        # Variant encoding
        var_col = "variant" if "variant" in df_merged.columns else ("variant_id" if "variant_id" in df_merged.columns else None)
        if var_col:
            v_series = df_merged[var_col].astype(str)
            # One-hot like numeric mapping
            v_alpha = (v_series.str.contains("Alpha", case=False)).astype(float).values
            v_beta = (v_series.str.contains("Beta", case=False)).astype(float).values
            v_gamma = (v_series.str.contains("Gamma", case=False)).astype(float).values
            features_list.extend([v_alpha, v_beta, v_gamma])
            feature_names.extend(["variant_is_alpha", "variant_is_beta", "variant_is_gamma"])
        else:
            features_list.append(np.zeros(len(df_merged), dtype=np.float32))
            feature_names.append("variant_unknown")

        # Tabular inspection & physical measurement features (e.g. roughness_ra, dimension_delta_mm, optical_score)
        ignore_cols = {
            "label", "is_defective", "unit_id", "batch_id", "variant", "variant_id",
            "timestamp", "defect_category", "defect_family", "image_path", "bbox",
            "temperature", "speed", "pressure", "vibration"
        }
        measurement_cols = [
            c for c in df_merged.columns
            if c not in ignore_cols and pd.api.types.is_numeric_dtype(df_merged[c])
        ]
        for col in measurement_cols:
            series = pd.to_numeric(df_merged[col], errors="coerce")
            med_val = series.median() if not np.isnan(series.median()) else 0.0
            features_list.append(series.fillna(med_val).values)
            feature_names.append(col)


        # Unit sequence / index proxy within batch
        if "unit_id" in df_merged.columns:
            try:
                u_nums = df_merged["unit_id"].astype(str).str.extract(r'(\d+)')[0].astype(float)
                u_norm = (u_nums - u_nums.min()) / max(1.0, (u_nums.max() - u_nums.min()))
                features_list.append(u_norm.fillna(0.0).values)
                feature_names.append("unit_seq_norm")
            except Exception:
                pass

        is_visual = any(c in feature_names for c in ["mean_intensity", "edge_energy", "laplacian_var"])
        X = np.column_stack(features_list).astype(np.float32)
        return X, feature_names, is_visual

    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batches: np.ndarray,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a group split by batch_id to prevent data leakage across batches.
        
        Returns:
            train_idx, test_idx
        """
        n_samples = len(y)
        unique_batches = np.unique(batches)

        # If multiple batches exist, use Group splitting
        if len(unique_batches) >= 3:
            try:
                # StratifiedGroupKFold balances label proportions while strictly keeping groups intact
                n_splits = max(2, min(5, len(unique_batches)))
                sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                splits = list(sgkf.split(X, y, groups=batches))
                train_idx, test_idx = splits[0]
                return train_idx, test_idx
            except Exception:
                # Fallback to GroupShuffleSplit
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                train_idx, test_idx = next(gss.split(X, y, groups=batches))
                return train_idx, test_idx

        # Fallback for single batch or edge case
        np.random.seed(random_state)
        shuffled = np.random.permutation(n_samples)
        n_test = max(1, int(n_samples * test_size))
        return shuffled[n_test:], shuffled[:n_test]

    def train(
        self,
        df_insp: pd.DataFrame,
        df_proc: Optional[pd.DataFrame] = None,
        uncertainty_threshold: float = 0.70
    ) -> Dict[str, Any]:
        """
        Train Stage 1 (binary detector) and Stage 2 (defect category classifier),
        and compute evaluation metrics strictly on the held-out test set.
        
        Parameters:
            df_insp: Cleaned inspection DataFrame.
            df_proc: Optional process telemetry DataFrame.
            uncertainty_threshold: Confidence threshold below which samples are flagged UNCERTAIN.
            
        Returns:
            Dictionary containing test set metrics, confusion matrices, and model metadata.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        if len(df_insp) == 0:
            return {"status": "empty_data", "error": "No inspection data available"}

        self.uncertainty_threshold = float(uncertainty_threshold)

        # 1. Feature extraction
        X, self.feature_names, self.is_image_based = self._extract_features(df_insp, df_proc)
        self.data_mode = "Organizer" if (not df_insp.get("is_synthetic", False) and has_organizer_images(self.organizer_dir)) else "Demo"

        # Standardize features for anomaly z-score
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0) + 1e-5

        # 2. Extract labels and batch groups
        label_col = "label" if "label" in df_insp.columns else ("is_defective" if "is_defective" in df_insp.columns else None)
        if label_col is None:
            raise ValueError("Inspection dataset must contain 'label' or 'is_defective' column.")
        
        y_binary = is_defective(df_insp[label_col]).astype(int).values
        batch_col = "batch_id" if "batch_id" in df_insp.columns else None
        batches = df_insp[batch_col].astype(str).values if batch_col else np.array(["B_ALL"] * len(y_binary))

        # 3. Leakage-free Group Split by batch_id
        train_idx, test_idx = self._split_data(X, y_binary, batches, test_size=0.20, random_state=42)
        self.test_indices = test_idx.tolist()
        self.test_batches = np.unique(batches[test_idx]).tolist()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]

        # 4. Stage 1: Binary Good vs Defective (class-balanced)
        self.binary_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42
        )
        self.binary_clf.fit(X_train, y_train)

        # 5. Stage 2: Defect Category Classifier (on defective units with valid category labels)
        cat_col = "defect_category" if "defect_category" in df_insp.columns else ("defect_family" if "defect_family" in df_insp.columns else None)
        self.family_clf = None
        self.stage2_classes = []
        stage2_metrics: Dict[str, Any] = {"available": False}

        if cat_col is not None:
            cat_series = df_insp[cat_col].astype(str).str.strip()
            # Filter defective rows that have an informative category (exclude 'None', 'good', empty)
            valid_cat_mask = (y_binary == 1) & (~cat_series.isin(["None", "none", "good", "", "nan"]))
            
            train_cat_mask = valid_cat_mask[train_idx]
            test_cat_mask = valid_cat_mask[test_idx]

            X_train_cat = X_train[train_cat_mask]
            y_train_cat = cat_series.iloc[train_idx][train_cat_mask].values

            X_test_cat = X_test[test_cat_mask]
            y_test_cat = cat_series.iloc[test_idx][test_cat_mask].values

            unique_train_cats = np.unique(y_train_cat)
            if len(unique_train_cats) >= 2:
                self.family_clf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=42
                )
                self.family_clf.fit(X_train_cat, y_train_cat)
                self.stage2_classes = self.family_clf.classes_.tolist()

                # Evaluate Stage 2 on held-out test set
                if len(y_test_cat) > 0 and set(y_test_cat).issubset(set(self.stage2_classes)):
                    y_pred_cat = self.family_clf.predict(X_test_cat)
                    stage2_acc = float(accuracy_score(y_test_cat, y_pred_cat))
                    stage2_f1_macro = float(f1_score(y_test_cat, y_pred_cat, average="macro", zero_division=0))
                    stage2_cm = confusion_matrix(y_test_cat, y_pred_cat, labels=self.stage2_classes).tolist()
                    stage2_report = classification_report(y_test_cat, y_pred_cat, labels=self.stage2_classes, output_dict=True, zero_division=0)
                    
                    stage2_metrics = {
                        "available": True,
                        "accuracy": round(stage2_acc, 4),
                        "f1_macro": round(stage2_f1_macro, 4),
                        "classes": self.stage2_classes,
                        "confusion_matrix": stage2_cm,
                        "report": stage2_report,
                        "test_sample_count": len(y_test_cat)
                    }

        # 6. Evaluate Stage 1 Strictly on Held-Out Test Set
        y_test_pred = self.binary_clf.predict(X_test)
        y_test_prob = self.binary_clf.predict_proba(X_test) # shape: (N_test, 2)
        prob_defective_test = y_test_prob[:, 1]
        max_prob_test = np.max(y_test_prob, axis=1)

        # Standard Performance Metrics
        acc = float(accuracy_score(y_test, y_test_pred))
        prec_bin = float(precision_score(y_test, y_test_pred, zero_division=0))
        rec_bin = float(recall_score(y_test, y_test_pred, zero_division=0))
        f1_bin = float(f1_score(y_test, y_test_pred, zero_division=0))

        prec_macro = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_test_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))

        # Confusion Matrix [ [TN, FP], [FN, TP] ]
        cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        # False Reject Rate (FRR) = Good units falsely rejected: FP / (TN + FP)
        frr = float(fp / max(1, (tn + fp)))

        # False Accept Rate (FAR) = Defective units falsely accepted: FN / (TP + FN)
        far = float(fn / max(1, (tp + fn)))

        # Expected Calibration Error (ECE) for completeness
        ece_val = self._compute_ece(y_test, prob_defective_test)

        # 7. Selective Classification: UNCERTAIN Flag Evaluation
        uncertain_mask = max_prob_test < self.uncertainty_threshold
        confident_mask = ~uncertain_mask
        uncertain_count = int(np.sum(uncertain_mask))
        uncertain_fraction = float(uncertain_count / max(1, len(y_test)))
        
        if np.sum(confident_mask) > 0:
            confident_acc = float(accuracy_score(y_test[confident_mask], y_test_pred[confident_mask]))
        else:
            confident_acc = 0.0

        # Compile Complete Metrics Dictionary
        feat_type = "Tabular Telemetry (Demo)"
        if self.is_image_based:
            feat_type = "CNN (MobileNetV3 on CPU)" if any("cnn_feat" in f for f in self.feature_names) else "Computer Vision (PIL / Scipy Spatial & Texture Filters)"

        self.test_metrics = {
            "status": "trained",
            "is_synthetic": (self.data_mode == "Demo"),
            "data_mode": self.data_mode,
            "feature_type": feat_type,
            "is_image_based": self.is_image_based,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "test_batches": self.test_batches,
            
            # Stage 1 Metrics (Held-Out Test Set)
            "accuracy": round(acc, 4),
            "precision": round(prec_bin, 4),
            "recall": round(rec_bin, 4),
            "f1": round(f1_bin, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "ece": round(ece_val, 4),
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "false_reject_rate": round(frr, 4),
            "false_accept_rate": round(far, 4),
            
            # Uncertainty / Selective Classification
            "uncertainty_threshold": self.uncertainty_threshold,
            "uncertain_count": uncertain_count,
            "uncertain_fraction": round(uncertain_fraction, 4),
            "confident_count": int(np.sum(confident_mask)),
            "confident_accuracy": round(confident_acc, 4),
            
            # Stage 2 Defect Family Metrics
            "stage2": stage2_metrics,
        }

        # 8. Save Pipeline with joblib
        self.save_model()

        return self.test_metrics

    def _compute_ece(self, y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(y_true)
        if n_total == 0:
            return 0.0

        for i in range(n_bins):
            bin_mask = (y_probs >= bin_edges[i]) & (y_probs < bin_edges[i + 1])
            if np.any(bin_mask):
                bin_acc = float(np.mean(y_true[bin_mask]))
                bin_conf = float(np.mean(y_probs[bin_mask]))
                bin_weight = float(np.sum(bin_mask) / n_total)
                ece += bin_weight * abs(bin_acc - bin_conf)
        return float(ece)

    def save_model(self) -> None:
        """Save trained models and metadata using joblib."""
        bundle = {
            "binary_clf": self.binary_clf,
            "family_clf": self.family_clf,
            "feature_names": self.feature_names,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "is_image_based": self.is_image_based,
            "data_mode": self.data_mode,
            "test_metrics": self.test_metrics,
            "test_indices": self.test_indices,
            "test_batches": self.test_batches,
            "uncertainty_threshold": self.uncertainty_threshold,
            "stage2_classes": self.stage2_classes
        }
        joblib.dump(bundle, self.model_path)

    def load_model(self) -> bool:
        """Load trained models and metadata using joblib if available."""
        if not os.path.exists(self.model_path):
            return False
        try:
            bundle = joblib.load(self.model_path)
            self.binary_clf = bundle.get("binary_clf")
            self.family_clf = bundle.get("family_clf")
            self.feature_names = bundle.get("feature_names", [])
            self.feature_means = bundle.get("feature_means")
            self.feature_stds = bundle.get("feature_stds")
            self.is_image_based = bundle.get("is_image_based", False)
            self.data_mode = bundle.get("data_mode", "Demo")
            self.test_metrics = bundle.get("test_metrics", {})
            self.test_indices = bundle.get("test_indices", [])
            self.test_batches = bundle.get("test_batches", [])
            self.uncertainty_threshold = bundle.get("uncertainty_threshold", 0.70)
            self.stage2_classes = bundle.get("stage2_classes", [])
            return True
        except Exception:
            return False

    def train_or_load(
        self,
        df_insp: pd.DataFrame,
        df_proc: Optional[pd.DataFrame] = None,
        force_retrain: bool = False,
        uncertainty_threshold: float = 0.70
    ) -> Dict[str, Any]:
        """
        Load model if previously trained and compatible, otherwise train fresh.
        """
        if not force_retrain and self.load_model():
            # Verify feature dimension compatibility
            try:
                probe_X, _, _ = self._extract_features(df_insp.head(2), df_proc)
                if hasattr(self.binary_clf, "n_features_in_") and self.binary_clf.n_features_in_ == probe_X.shape[1]:
                    if self.test_metrics:
                        return self.test_metrics
            except Exception:
                pass
        return self.train(df_insp, df_proc, uncertainty_threshold=uncertainty_threshold)

    def predict_dataframe(
        self,
        df_insp: pd.DataFrame,
        df_proc: Optional[pd.DataFrame] = None,
        uncertainty_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Run inference on arbitrary inspection DataFrame, returning predictions,
        class probabilities, and UNCERTAIN triage flags.
        
        Returns:
            DataFrame with predictions and human inspector triage queue recommendations.
        """
        if self.binary_clf is None:
            raise RuntimeError("QualityClassifier has not been trained or loaded.")

        thresh = float(uncertainty_threshold if uncertainty_threshold is not None else self.uncertainty_threshold)
        X, _, _ = self._extract_features(df_insp, df_proc)

        # Stage 1 Probabilities
        probs = self.binary_clf.predict_proba(X)
        p_good = probs[:, 0]
        p_defective = probs[:, 1]
        max_probs = np.max(probs, axis=1)
        raw_pred = self.binary_clf.predict(X)

        # Stage 2 Family Predictions
        family_preds = ["None"] * len(df_insp)
        if self.family_clf is not None:
            stage2_out = self.family_clf.predict(X)
            for idx, p in enumerate(raw_pred):
                if p == 1:
                    family_preds[idx] = str(stage2_out[idx])

        # Anomaly / Outlier z-score
        z_scores = np.zeros(len(df_insp))
        if self.feature_means is not None and self.feature_stds is not None:
            z_scores = np.max(np.abs((X - self.feature_means) / self.feature_stds), axis=1)

        records = []
        for i in range(len(df_insp)):
            row = df_insp.iloc[i]
            uid = str(row.get("unit_id", f"U{i}"))
            bid = str(row.get("batch_id", "N/A"))
            var = str(row.get("variant", row.get("variant_id", "N/A")))
            actual_label = int(is_defective(row.get("label", row.get("is_defective", 0))))
            actual_cat = str(row.get("defect_category", row.get("defect_family", "None")))

            p_def = float(p_defective[i])
            m_prob = float(max_probs[i])
            z_sc = float(z_scores[i])

            is_novel = z_sc > 2.8
            is_uncertain = m_prob < thresh

            if is_novel:
                pred_label = "NOVEL"
                status = "NOVEL"
                triage = "🚨 Novel Signature: Engineering Root-Cause Review"
            elif is_uncertain:
                pred_label = "UNCERTAIN"
                status = "UNCERTAIN"
                triage = "⚠️ Borderline Confidence: Human Inspector Sign-Off Required"
            elif raw_pred[i] == 1:
                pred_label = f"Defective ({family_preds[i]})"
                status = "REJECT"
                triage = f"❌ Automated Reject: Route to {family_preds[i]} Rework/Scrap"
            else:
                pred_label = "Good"
                status = "ACCEPT"
                triage = "✅ Conforming: Direct to Downstream Packaging"

            records.append({
                "unit_id": uid,
                "batch_id": bid,
                "variant": var,
                "actual_label": actual_label,
                "actual_category": actual_cat,
                "p_good": round(float(p_good[i]), 3),
                "p_defective": round(p_def, 3),
                "confidence": round(m_prob, 3),
                "anomaly_z": round(z_sc, 2),
                "predicted_status": status,
                "prediction": pred_label,
                "is_uncertain": is_uncertain,
                "is_novel": is_novel,
                "triage_recommendation": triage
            })

        return pd.DataFrame(records)

    def evaluate_unit(self, unit_row: pd.Series, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate a single unit for backward compatibility with existing tests and pipelines.
        """
        df_single = pd.DataFrame([unit_row])
        pred_df = self.predict_dataframe(df_single, uncertainty_threshold=threshold)
        res = pred_df.iloc[0].to_dict()

        return {
            "unit_id": res["unit_id"],
            "calibrated_prob": res["p_defective"],
            "predicted_defective": bool(res["predicted_status"] in ["REJECT", "NOVEL"]),
            "defect_family": res["actual_category"] if res["predicted_status"] == "REJECT" else "None",
            "is_uncertain": res["is_uncertain"],
            "is_novel": res["is_novel"],
            "anomaly_z_score": res["anomaly_z"],
            "status": res["predicted_status"],
            "action": res["triage_recommendation"]
        }
