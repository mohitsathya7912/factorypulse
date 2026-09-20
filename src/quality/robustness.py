"""
src/quality/robustness.py
FactoryPulse Systematic Perturbation Suite and Data Augmentation Robustness Engine.

Conforms to Checkpoint 3 STAGE D ground rules:
1. Deterministic perturbation suite on the 500-image test subset (100 per class):
   - brightness x0.6 and x1.4
   - contrast x0.6 and x1.4
   - rotate 90 and 180
   - horizontal flip
   - Gaussian blur radius 1 and 2
   - Gaussian noise sigma 10 and 25
   (PIL and NumPy only)
2. Feature extraction with frozen ResNet18 (CPU). All features cached to cache/.
3. Metrics computed on test subset: accuracy, macro-F1, false-reject rate (FRR),
   false-accept rate (FAR), uncertain fraction, mean calibrated confidence.
4. Identifies drop from baseline and names the worst-case degradation.
5. Augmentation: generates ONE randomly augmented copy of every train image
   (random brightness/contrast, 90-deg rotations, flips, blur, noise with PIL/NumPy).
6. Trains a robust LogisticRegression model on original + augmented train features,
   recalibrating temperature scaling on the validation split.
7. Before/after comparison table across the full perturbation suite, reporting
   strictly measured empirical numbers (including any condition that got worse).
"""

import os
import sys
import json
import time
from typing import Dict, Any, Tuple, List, Optional, Callable
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

from src.quality.calibration import (
    wilson_score_interval,
    fit_temperature_scaling,
    calibrate_logits
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


# =============================================================================
# 1. Deterministic Perturbations (PIL / NumPy only)
# =============================================================================

PERTURBATION_NAMES = [
    "baseline",
    "bright_0.6",
    "bright_1.4",
    "contrast_0.6",
    "contrast_1.4",
    "rot_90",
    "rot_180",
    "hflip",
    "blur_1",
    "blur_2",
    "noise_10",
    "noise_25"
]

PERTURBATION_LABELS = {
    "baseline": "Baseline (Clean)",
    "bright_0.6": "Simulated Brightness x0.6",
    "bright_1.4": "Simulated Brightness x1.4",
    "contrast_0.6": "Simulated Contrast x0.6",
    "contrast_1.4": "Simulated Contrast x1.4",
    "rot_90": "Simulated Rotation 90°",
    "rot_180": "Simulated Rotation 180°",
    "hflip": "Simulated Horizontal Flip",
    "blur_1": "Simulated Gaussian Blur (r=1)",
    "blur_2": "Simulated Gaussian Blur (r=2)",
    "noise_10": "Simulated Gaussian Noise (sigma=10)",
    "noise_25": "Simulated Gaussian Noise (sigma=25)"
}


def apply_perturbation(img: Image.Image, p_type: str, seed: int = 42) -> Image.Image:
    """
    Apply a deterministic perturbation using PIL and NumPy only.
    
    Args:
        img: Input PIL Image
        p_type: Perturbation identifier
        seed: Random seed for noise generation
        
    Returns:
        Perturbed PIL Image
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    if p_type == "baseline" or p_type == "orig":
        return img
    elif p_type == "bright_0.6":
        return ImageEnhance.Brightness(img).enhance(0.6)
    elif p_type == "bright_1.4":
        return ImageEnhance.Brightness(img).enhance(1.4)
    elif p_type == "contrast_0.6":
        return ImageEnhance.Contrast(img).enhance(0.6)
    elif p_type == "contrast_1.4":
        return ImageEnhance.Contrast(img).enhance(1.4)
    elif p_type == "rot_90":
        return img.rotate(90, expand=False)
    elif p_type == "rot_180":
        return img.rotate(180, expand=False)
    elif p_type == "hflip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif p_type == "blur_1":
        return img.filter(ImageFilter.GaussianBlur(radius=1.0))
    elif p_type == "blur_2":
        return img.filter(ImageFilter.GaussianBlur(radius=2.0))
    elif p_type == "noise_10":
        arr = np.array(img, dtype=np.float32)
        rng = np.random.RandomState(seed)
        noise = rng.normal(0.0, 10.0, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    elif p_type == "noise_25":
        arr = np.array(img, dtype=np.float32)
        rng = np.random.RandomState(seed)
        noise = rng.normal(0.0, 25.0, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    else:
        raise ValueError(f"Unsupported perturbation type: '{p_type}'")


def random_augment_train_image(img: Image.Image, seed: int = 42) -> Image.Image:
    """
    Generate ONE randomly augmented copy of a train image using PIL and NumPy only.
    Stochastically samples brightness, contrast, 90-degree rotations, flips, blur, and noise.
    
    Args:
        img: Input PIL Image
        seed: Reproducible random seed per image
        
    Returns:
        Augmented PIL Image
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    rng = np.random.RandomState(seed)

    # 1. Random brightness between 0.6 and 1.4
    b_factor = float(rng.uniform(0.6, 1.4))
    img = ImageEnhance.Brightness(img).enhance(b_factor)

    # 2. Random contrast between 0.6 and 1.4
    c_factor = float(rng.uniform(0.6, 1.4))
    img = ImageEnhance.Contrast(img).enhance(c_factor)

    # 3. Random 90-degree rotation (0, 90, 180, 270)
    rot_angle = int(rng.choice([0, 90, 180, 270]))
    if rot_angle > 0:
        img = img.rotate(rot_angle, expand=False)

    # 4. Random horizontal flip (50% probability)
    if rng.rand() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 5. Random Gaussian blur (50% probability, radius in [0.5, 2.0])
    if rng.rand() > 0.5:
        blur_r = float(rng.uniform(0.5, 2.0))
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_r))

    # 6. Random Gaussian noise (50% probability, sigma in [5, 25])
    if rng.rand() > 0.5:
        sigma = float(rng.uniform(5.0, 25.0))
        arr = np.array(img, dtype=np.float32)
        noise = rng.normal(0.0, sigma, arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    return img


# =============================================================================
# 2. PyTorch Feature Extraction Pipeline
# =============================================================================

def get_feature_extractor() -> Tuple[Any, Any]:
    """
    Initialize frozen ResNet18 feature extractor and preprocessing transforms on CPU.
    """
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
    return model, preprocess


def get_500_image_test_subset(split_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Get the exact 500-image test subset (100 per class from test split)
    matching the subset used in Stage C patch localization evaluation.
    """
    if split_csv_path is None:
        split_csv_path = os.path.join(CACHE_DIR, "split.csv")
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"Split CSV not found at {split_csv_path}")

    split_df = pd.read_csv(split_csv_path)
    classes = ["normal", "crack", "hole", "rust", "scratch"]
    samples = []
    for c in classes:
        c_df = split_df[(split_df["split"] == "test") & (split_df["category"] == c)].head(100)
        samples.append(c_df)

    test_500 = pd.concat(samples, ignore_index=True).reset_index(drop=True)
    return test_500


def extract_perturbation_features_500(
    cache_path: Optional[str] = None,
    batch_size: int = 64,
    force_recompute: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extract and cache ResNet18 features for all 11 perturbations + baseline
    on the 500-image test subset.
    
    Returns:
        Dict mapping perturbation_name -> (500, 512) feature array
    """
    if cache_path is None:
        cache_path = os.path.join(CACHE_DIR, "features_perturbations_500.npz")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path) and not force_recompute:
        try:
            data = np.load(cache_path, allow_pickle=True)
            res = {k: data[k] for k in data.files}
            if all(p in res for p in PERTURBATION_NAMES):
                return res
        except Exception:
            pass

    test_500 = get_500_image_test_subset()
    model, preprocess = get_feature_extractor()

    results: Dict[str, np.ndarray] = {}
    results["categories"] = test_500["category"].values
    results["paths"] = test_500["path"].values

    total_perts = len(PERTURBATION_NAMES)
    for p_idx, p_name in enumerate(PERTURBATION_NAMES):
        t0 = time.time()
        feats_list = []
        n_imgs = len(test_500)

        for b_start in range(0, n_imgs, batch_size):
            b_end = min(b_start + batch_size, n_imgs)
            batch_tensors = []
            for i in range(b_start, b_end):
                row = test_500.iloc[i]
                p_rel = row["path"]
                p_abs = os.path.join(ROOT_DIR, p_rel) if not os.path.isabs(p_rel) else p_rel
                img = Image.open(p_abs)
                # Apply perturbation with deterministic seed per image
                img_pert = apply_perturbation(img, p_name, seed=42 + i)
                batch_tensors.append(preprocess(img_pert))

            stacked = torch.stack(batch_tensors)
            with torch.no_grad():
                f_batch = model(stacked).numpy()
            feats_list.append(f_batch)

        feats_all = np.concatenate(feats_list, axis=0)
        results[p_name] = feats_all
        elapsed = time.time() - t0
        print(f"[{p_idx + 1}/{total_perts}] Perturbation '{p_name}': 500 features extracted in {elapsed:.2f} s")

    np.savez_compressed(cache_path, **results)
    return results


def extract_augmented_train_features(
    cache_path: Optional[str] = None,
    batch_size: int = 64,
    force_recompute: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate ONE randomly augmented copy of every train image in split.csv,
    extract ResNet18 features, and cache to cache/features_augmented_train.npz.
    
    Returns:
        Dict with 'features', 'categories', 'paths'
    """
    if cache_path is None:
        cache_path = os.path.join(CACHE_DIR, "features_augmented_train.npz")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path) and not force_recompute:
        try:
            data = np.load(cache_path, allow_pickle=True)
            if "features" in data and len(data["features"]) > 0:
                return {k: data[k] for k in data.files}
        except Exception:
            pass

    split_csv_path = os.path.join(CACHE_DIR, "split.csv")
    split_df = pd.read_csv(split_csv_path)
    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
    n_train = len(train_df)

    model, preprocess = get_feature_extractor()
    t0 = time.time()
    feats_list = []

    print(f"Extracting features for ONE randomly augmented copy of {n_train:,} train images...")
    for b_start in range(0, n_train, batch_size):
        b_end = min(b_start + batch_size, n_train)
        batch_tensors = []
        for i in range(b_start, b_end):
            row = train_df.iloc[i]
            p_rel = row["path"]
            p_abs = os.path.join(ROOT_DIR, p_rel) if not os.path.isabs(p_rel) else p_rel
            img = Image.open(p_abs)
            aug_img = random_augment_train_image(img, seed=42 + i)
            batch_tensors.append(preprocess(aug_img))

        stacked = torch.stack(batch_tensors)
        with torch.no_grad():
            f_batch = model(stacked).numpy()
        feats_list.append(f_batch)

        if (b_end % 1280 == 0) or (b_end == n_train):
            print(f"  Processed {b_end:,} / {n_train:,} augmented train images ({time.time() - t0:.1f} s)...")

    feats_all = np.concatenate(feats_list, axis=0)
    result = {
        "features": feats_all,
        "categories": train_df["category"].values,
        "paths": train_df["path"].values
    }
    np.savez_compressed(cache_path, **result)
    print(f"Augmented train feature extraction complete: {len(feats_all):,} samples in {time.time() - t0:.1f} s.")
    return result


# =============================================================================
# 3. Model Evaluation on Perturbation Suite
# =============================================================================

def evaluate_model_on_perturbations(
    clf: Any,
    classes: List[str],
    temperature: float,
    pert_features: Dict[str, np.ndarray],
    categories: np.ndarray,
    uncertainty_threshold: float = 0.70
) -> Dict[str, Any]:
    """
    Evaluate a calibrated model across all 11 perturbations + baseline on the 500-image test subset.
    
    Computes:
    - 5-class accuracy
    - 5-class macro-F1
    - false-reject rate (FRR: normal predicted defective)
    - false-accept rate (FAR: defect predicted normal)
    - uncertain fraction (max probability < uncertainty_threshold)
    - mean calibrated confidence
    - drops relative to baseline
    """
    results_by_pert = {}
    idx_normal = classes.index("normal") if "normal" in classes else -1
    y_true_binary = (categories != "normal").astype(int)
    n_normal = int(np.sum(y_true_binary == 0))
    n_defect = int(np.sum(y_true_binary == 1))

    for p_name in PERTURBATION_NAMES:
        X_p = pert_features[p_name]
        logits = clf.decision_function(X_p)
        probs = calibrate_logits(logits, temperature)

        max_probs = np.max(probs, axis=1)
        pred_indices = np.argmax(probs, axis=1)
        pred_classes = np.array(classes)[pred_indices]

        # 5-class metrics
        acc = float(accuracy_score(categories, pred_classes))
        f1_macro = float(f1_score(categories, pred_classes, average="macro", zero_division=0))

        # Binary good-vs-defective: P(defective) = 1 - P(normal)
        if idx_normal >= 0:
            p_normal = probs[:, idx_normal]
            p_def = 1.0 - p_normal
            pred_binary = (p_def >= 0.50).astype(int)
        else:
            pred_binary = (pred_classes != "normal").astype(int)

        cm = confusion_matrix(y_true_binary, pred_binary, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        frr, frr_ci = wilson_score_interval(fp, n_normal)
        far, far_ci = wilson_score_interval(fn, n_defect)

        # Uncertainty triage
        is_uncertain = max_probs < uncertainty_threshold
        unc_frac = float(np.mean(is_uncertain))
        mean_conf = float(np.mean(max_probs))

        results_by_pert[p_name] = {
            "perturbation": p_name,
            "label": PERTURBATION_LABELS.get(p_name, p_name),
            "accuracy": round(acc, 4),
            "accuracy_pct": round(acc * 100.0, 2),
            "f1_macro": round(f1_macro, 4),
            "false_reject_rate": round(frr, 4),
            "false_reject_rate_pct": round(frr * 100.0, 2),
            "false_reject_ci": [round(frr_ci[0], 4), round(frr_ci[1], 4)],
            "false_accept_rate": round(far, 4),
            "false_accept_rate_pct": round(far * 100.0, 2),
            "false_accept_ci": [round(far_ci[0], 4), round(far_ci[1], 4)],
            "uncertain_fraction": round(unc_frac, 4),
            "uncertain_fraction_pct": round(unc_frac * 100.0, 2),
            "mean_confidence": round(mean_conf, 4),
            "mean_confidence_pct": round(mean_conf * 100.0, 2),
            "fp_count": fp,
            "fn_count": fn,
            "tp_count": tp,
            "tn_count": tn
        }

    # Compute drops relative to baseline
    base_acc = results_by_pert["baseline"]["accuracy"]
    base_f1 = results_by_pert["baseline"]["f1_macro"]

    worst_case_name = "baseline"
    worst_case_drop_acc = -1.0
    worst_case_drop_f1 = -1.0

    for p_name in PERTURBATION_NAMES:
        item = results_by_pert[p_name]
        drop_acc = base_acc - item["accuracy"]
        drop_f1 = base_f1 - item["f1_macro"]
        item["drop_accuracy"] = round(drop_acc, 4)
        item["drop_accuracy_pct"] = round(drop_acc * 100.0, 2)
        item["drop_f1_macro"] = round(drop_f1, 4)

        if p_name != "baseline":
            if drop_f1 > worst_case_drop_f1:
                worst_case_drop_f1 = drop_f1
                worst_case_drop_acc = drop_acc
                worst_case_name = p_name

    summary = {
        "results_by_pert": results_by_pert,
        "worst_case": {
            "perturbation": worst_case_name,
            "label": PERTURBATION_LABELS.get(worst_case_name, worst_case_name),
            "drop_accuracy": round(worst_case_drop_acc, 4),
            "drop_accuracy_pct": round(worst_case_drop_acc * 100.0, 2),
            "drop_f1_macro": round(worst_case_drop_f1, 4),
            "worst_f1_macro": results_by_pert[worst_case_name]["f1_macro"],
            "worst_accuracy": results_by_pert[worst_case_name]["accuracy"]
        },
        "sample_count": len(categories),
        "classes": classes,
        "temperature": round(temperature, 4)
    }
    return summary


# =============================================================================
# 4. Augmentation Training & Robust Model Pipeline
# =============================================================================

def train_and_evaluate_robust_pipeline(
    force_recompute: bool = False
) -> Dict[str, Any]:
    """
    Full STAGE D Robustness and Augmentation workflow:
    1. Extract / load 500-image test subset under all 11 perturbations + baseline.
    2. Evaluate baseline model on the perturbation suite.
    3. Extract / load augmented train features.
    4. Train "robust" LogisticRegression on original + augmented train features.
    5. Recalibrate robust model on VAL split (temperature scaling).
    6. Evaluate robust model on the same 500-image perturbation suite.
    7. Generate before/after comparison table, highlighting all deltas and any worse case.
    8. Cache all results to disk.
    """
    eval_cache_path = os.path.join(CACHE_DIR, "robust_model_eval.json")
    if os.path.exists(eval_cache_path) and not force_recompute:
        try:
            with open(eval_cache_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    # 1. Load 500-image test perturbation features
    pert_data = extract_perturbation_features_500(force_recompute=force_recompute)
    test_categories = pert_data["categories"]

    # 2. Load original train, val, test features
    feat_data = np.load(os.path.join(CACHE_DIR, "features_resnet18.npz"), allow_pickle=True)
    orig_X = feat_data["features"]
    orig_cats = feat_data["categories"]
    orig_splits = feat_data["splits"]

    train_mask = orig_splits == "train"
    val_mask = orig_splits == "val"

    X_train_orig = orig_X[train_mask]
    y_train_orig = orig_cats[train_mask]
    X_val = orig_X[val_mask]
    y_val = orig_cats[val_mask]

    # Load baseline model
    base_model_path = os.path.join(MODELS_DIR, "organizer_5class_logistic_regression.joblib")
    if os.path.exists(base_model_path):
        base_saved = joblib.load(base_model_path)
        base_clf = base_saved["clf_5class"]
        classes = sorted(base_saved.get("classes", np.unique(orig_cats).tolist()))
        base_temp = float(base_saved.get("temperature", 1.0))
    else:
        # Train baseline if not present
        base_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        base_clf.fit(X_train_orig, y_train_orig)
        classes = sorted(base_clf.classes_.tolist())
        y_val_idx = np.array([classes.index(c) for c in y_val])
        logits_val = base_clf.decision_function(X_val)
        base_temp = fit_temperature_scaling(logits_val, y_val_idx)

    # Evaluate baseline model
    base_eval = evaluate_model_on_perturbations(
        clf=base_clf,
        classes=classes,
        temperature=base_temp,
        pert_features=pert_data,
        categories=test_categories
    )

    # 3. Load / extract augmented train features
    aug_data = extract_augmented_train_features(force_recompute=force_recompute)
    X_train_aug = aug_data["features"]
    y_train_aug = aug_data["categories"]

    # Combine original + augmented train features
    X_train_combined = np.concatenate([X_train_orig, X_train_aug], axis=0)
    y_train_combined = np.concatenate([y_train_orig, y_train_aug], axis=0)

    print(f"Training robust model on {len(X_train_combined):,} combined features ({len(X_train_orig):,} original + {len(X_train_aug):,} augmented)...")
    robust_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    robust_clf.fit(X_train_combined, y_train_combined)

    # 4. Recalibrate robust model on VAL split
    y_val_idx = np.array([classes.index(c) for c in y_val])
    robust_logits_val = robust_clf.decision_function(X_val)
    robust_temp = fit_temperature_scaling(robust_logits_val, y_val_idx)
    print(f"Robust model calibrated on VAL split: Temperature T* = {robust_temp:.4f}")

    # Save robust model
    robust_model_path = os.path.join(MODELS_DIR, "organizer_robust_logistic_regression.joblib")
    joblib.dump({
        "clf_5class": robust_clf,
        "classes": classes,
        "temperature": robust_temp,
        "n_train_samples": len(X_train_combined)
    }, robust_model_path)

    # 5. Evaluate robust model on same perturbation suite
    robust_eval = evaluate_model_on_perturbations(
        clf=robust_clf,
        classes=classes,
        temperature=robust_temp,
        pert_features=pert_data,
        categories=test_categories
    )

    # 6. Build Before / After Comparison Table
    comparison_table = []
    cases_improved = []
    cases_worse = []
    cases_unchanged = []

    for p_name in PERTURBATION_NAMES:
        b_res = base_eval["results_by_pert"][p_name]
        r_res = robust_eval["results_by_pert"][p_name]

        delta_acc = r_res["accuracy"] - b_res["accuracy"]
        delta_f1 = r_res["f1_macro"] - b_res["f1_macro"]
        delta_frr = r_res["false_reject_rate"] - b_res["false_reject_rate"]
        delta_far = r_res["false_accept_rate"] - b_res["false_accept_rate"]
        delta_conf = r_res["mean_confidence"] - b_res["mean_confidence"]

        status = "Unchanged"
        if delta_f1 > 0.001 or delta_acc > 0.001:
            status = "Improved"
            cases_improved.append(p_name)
        elif delta_f1 < -0.001 or delta_acc < -0.001:
            status = "Degraded"
            cases_worse.append(p_name)
        else:
            cases_unchanged.append(p_name)

        row = {
            "perturbation": p_name,
            "label": PERTURBATION_LABELS.get(p_name, p_name),
            "baseline_accuracy": b_res["accuracy"],
            "baseline_accuracy_pct": b_res["accuracy_pct"],
            "robust_accuracy": r_res["accuracy"],
            "robust_accuracy_pct": r_res["accuracy_pct"],
            "delta_accuracy": round(delta_acc, 4),
            "delta_accuracy_pct": round(delta_acc * 100.0, 2),
            "baseline_f1": b_res["f1_macro"],
            "robust_f1": r_res["f1_macro"],
            "delta_f1": round(delta_f1, 4),
            "baseline_frr": b_res["false_reject_rate"],
            "baseline_frr_pct": b_res["false_reject_rate_pct"],
            "robust_frr": r_res["false_reject_rate"],
            "robust_frr_pct": r_res["false_reject_rate_pct"],
            "delta_frr": round(delta_frr, 4),
            "baseline_far": b_res["false_accept_rate"],
            "baseline_far_pct": b_res["false_accept_rate_pct"],
            "robust_far": r_res["false_accept_rate"],
            "robust_far_pct": r_res["false_accept_rate_pct"],
            "delta_far": round(delta_far, 4),
            "baseline_conf": b_res["mean_confidence"],
            "robust_conf": r_res["mean_confidence"],
            "delta_conf": round(delta_conf, 4),
            "status": status
        }
        comparison_table.append(row)

    output = {
        "baseline_evaluation": base_eval,
        "robust_evaluation": robust_eval,
        "comparison_table": comparison_table,
        "cases_improved": cases_improved,
        "cases_worse": cases_worse,
        "cases_unchanged": cases_unchanged,
        "baseline_temperature": round(base_temp, 4),
        "robust_temperature": round(robust_temp, 4),
        "train_samples_original": int(len(X_train_orig)),
        "train_samples_augmented": int(len(X_train_aug)),
        "train_samples_total": int(len(X_train_combined)),
        "test_samples_count": int(len(test_categories))
    }

    with open(eval_cache_path, "w") as f:
        json.dump(output, f, indent=2)

    return output
