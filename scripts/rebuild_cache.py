"""
scripts/rebuild_cache.py
Named CLI utility to rebuild all model checkpoints, feature caches, and memory banks from scratch.

Usage:
    python scripts/rebuild_cache.py [--all] [--classifier] [--localization] [--novelty] [--robustness] [--dev]

Examples:
    python scripts/rebuild_cache.py --all --dev      # Fast rebuild (400 images per class, ~2 mins)
    python scripts/rebuild_cache.py --all            # Full rebuild (12,000 images, ~8 mins)
    python scripts/rebuild_cache.py --classifier     # Rebuild only ResNet18 classifier & calibration
    python scripts/rebuild_cache.py --localization   # Rebuild PatchCore patch memory bank
    python scripts/rebuild_cache.py --novelty        # Rebuild novelty detector embeddings
"""

import os
import sys
import argparse
import time

# Add root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def rebuild_classifier(dev_mode: bool = False):
    print("\n" + "=" * 70)
    print(f"1. Rebuilding ResNet18 Features & Calibrated 5-Class Classifier (dev_mode={dev_mode})...")
    print("=" * 70)
    from src.quality.classifier import QualityClassifier
    t0 = time.time()
    clf = QualityClassifier()
    # Remove existing cache if rebuilding
    cache_path = os.path.join(ROOT_DIR, "cache", "features_resnet18_dev.npz" if dev_mode else "features_resnet18.npz")
    if os.path.exists(cache_path):
        print(f"Removing existing cache: {cache_path}")
        os.remove(cache_path)
    metrics = clf.train_organizer_resnet18(dev_mode=dev_mode)
    elapsed = time.time() - t0
    print(f"[OK] Classifier rebuilt in {elapsed:.2f}s!")
    print(f"  Accuracy: {metrics.get('accuracy_5class', 0)*100:.2f}% | Macro F1: {metrics.get('f1_macro_5class', 0):.4f}")
    print(f"  Calibrated ECE: {metrics.get('ece_calibrated', 0):.6f} (T*={metrics.get('temperature', 1.0):.4f})")
    return metrics


def rebuild_localization():
    print("\n" + "=" * 70)
    print("2. Rebuilding PatchCore Anomaly Localization Memory Bank...")
    print("=" * 70)
    from src.quality.localization import PatchAnomalyDetector
    t0 = time.time()
    detector = PatchAnomalyDetector()
    bank_path = os.path.join(ROOT_DIR, "cache", "patch_memory_bank.pt")
    if os.path.exists(bank_path):
        print(f"Removing existing memory bank: {bank_path}")
        os.remove(bank_path)
    bank = detector.build_or_load_memory_bank(n_normal_train=300, bank_size=30000)
    elapsed = time.time() - t0
    print(f"[OK] Patch memory bank rebuilt in {elapsed:.2f}s!")
    print(f"  Patches in bank: {bank.shape[0]:,} | Dimension: {bank.shape[1]}")
    print(f"  99th-percentile normal threshold: {detector.threshold_99th:.4f}")
    return bank


def rebuild_novelty():
    print("\n" + "=" * 70)
    print("3. Rebuilding Novelty Detector Embeddings...")
    print("=" * 70)
    from src.quality.novelty import NoveltyDetector
    t0 = time.time()
    detector = NoveltyDetector()
    res = detector.fit(force_rebuild=True)
    elapsed = time.time() - t0
    print(f"[OK] Novelty detector rebuilt in {elapsed:.2f}s!")
    print(f"  Train embeddings: {res.get('n_train_samples', 0):,} | Classes: {res.get('classes', [])}")
    print(f"  Distribution threshold (tau_nov): {res.get('tau_nov', 0.1188):.4f}")
    return res


def rebuild_robustness():
    print("\n" + "=" * 70)
    print("4. Rebuilding Robustness Perturbation Evaluation...")
    print("=" * 70)
    from src.quality.robustness import RobustnessEvaluator
    t0 = time.time()
    evaluator = RobustnessEvaluator()
    cache_path = os.path.join(ROOT_DIR, "cache", "robust_model_eval.json")
    if os.path.exists(cache_path):
        print(f"Removing existing robustness cache: {cache_path}")
        os.remove(cache_path)
    res = evaluator.evaluate_all(n_test_samples=500, use_cache=False)
    elapsed = time.time() - t0
    print(f"[OK] Robustness evaluation completed in {elapsed:.2f}s!")
    return res


def main():
    parser = argparse.ArgumentParser(description="FactoryPulse Model & Cache Rebuilder")
    parser.add_argument("--all", action="store_true", help="Rebuild all models, caches, and memory banks")
    parser.add_argument("--classifier", action="store_true", help="Rebuild ResNet18 features & 5-class classifier")
    parser.add_argument("--localization", action="store_true", help="Rebuild PatchCore patch memory bank")
    parser.add_argument("--novelty", action="store_true", help="Rebuild novelty detector embeddings")
    parser.add_argument("--robustness", action="store_true", help="Rebuild robustness perturbation suite")
    parser.add_argument("--dev", action="store_true", default=False, help="Use dev mode (400 images/class, 2,000 total) for fast rebuild")

    args = parser.parse_args()

    # If no flags given, default to --all
    rebuild_all = args.all or (not args.classifier and not args.localization and not args.novelty and not args.robustness)

    start_total = time.time()
    print("=" * 70)
    print("FACTORYPULSE CACHE & MODEL REBUILD PIPELINE")
    print("=" * 70)

    if rebuild_all or args.classifier:
        rebuild_classifier(dev_mode=args.dev)

    if rebuild_all or args.localization:
        rebuild_localization()

    if rebuild_all or args.novelty:
        rebuild_novelty()

    if rebuild_all or args.robustness:
        rebuild_robustness()

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"ALL REQUESTED CACHES & MODELS SUCCESSFULLY REBUILT IN {total_elapsed:.2f}s!")
    print("=" * 70)


if __name__ == "__main__":
    main()
