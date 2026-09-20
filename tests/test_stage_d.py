"""
tests/test_stage_d.py
Unit and Regression Tests for Checkpoint 3 STAGE D:
Robustness Perturbation Suite, Data Augmentation, and Out-of-Distribution Novelty Detection.
"""

import os
import unittest
import numpy as np
from PIL import Image

from src.quality.robustness import (
    PERTURBATION_NAMES,
    apply_perturbation,
    random_augment_train_image
)
from src.quality.novelty import (
    NoveltyDetector,
    get_production_novelty_detector
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestStageD(unittest.TestCase):

    def test_01_deterministic_perturbations(self):
        """Verify all 11 perturbations + baseline transform PIL image preserving dimensions."""
        img = Image.new("RGB", (64, 64), color=(120, 140, 160))
        for p_name in PERTURBATION_NAMES:
            p_img = apply_perturbation(img, p_name, seed=42)
            self.assertIsInstance(p_img, Image.Image, f"Failed for perturbation {p_name}")
            self.assertEqual(p_img.size, (64, 64), f"Dimensions altered by {p_name}")
            self.assertEqual(p_img.mode, "RGB", f"Mode altered by {p_name}")

        # Test reproducibility of noise perturbation
        noise1 = np.array(apply_perturbation(img, "noise_25", seed=100))
        noise2 = np.array(apply_perturbation(img, "noise_25", seed=100))
        np.testing.assert_array_equal(noise1, noise2)

        # Different seed produces different noise
        noise3 = np.array(apply_perturbation(img, "noise_25", seed=200))
        self.assertFalse(np.array_equal(noise1, noise3))

    def test_02_random_augment_train_image(self):
        """Verify stochastic augmentation produces valid PIL image and is reproducible per seed."""
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        aug1 = random_augment_train_image(img, seed=42)
        aug2 = random_augment_train_image(img, seed=42)
        self.assertIsInstance(aug1, Image.Image)
        self.assertEqual(aug1.size, (64, 64))

        np.testing.assert_array_equal(np.array(aug1), np.array(aug2))

    def test_03_novelty_detector_mathematical_properties(self):
        """Verify NoveltyDetector k-NN cosine distance logic on synthetic data."""
        rng = np.random.RandomState(42)
        # 3 classes, 10 samples each, 16 dimensions
        # Class 0: centered around unit vector [1, 0, 0, ...]
        # Class 1: centered around unit vector [0, 1, 0, ...]
        # Class 2: centered around unit vector [0, 0, 1, ...]
        v0 = np.zeros(16); v0[0] = 1.0
        v1 = np.zeros(16); v1[1] = 1.0
        v2 = np.zeros(16); v2[2] = 1.0

        X_c0 = v0 + rng.normal(0, 0.05, (10, 16))
        X_c1 = v1 + rng.normal(0, 0.05, (10, 16))
        X_c2 = v2 + rng.normal(0, 0.05, (10, 16))

        X_train = np.vstack([X_c0, X_c1, X_c2])
        y_train = np.array(["c0"] * 10 + ["c1"] * 10 + ["c2"] * 10)

        detector = NoveltyDetector(k=3)
        detector.fit(X_train, y_train)

        # In-distribution query: close to class 0
        q_known = (v0 + rng.normal(0, 0.02, 16)).reshape(1, -1)
        score_known = float(detector.score(q_known)[0])

        # Out-of-distribution query: orthogonal vector [0, 0, 0, 1, 0, ...]
        v_novel = np.zeros(16); v_novel[3] = 1.0
        q_novel = v_novel.reshape(1, -1)
        score_novel = float(detector.score(q_novel)[0])

        # Novel query should have significantly higher novelty score than known
        self.assertLess(score_known, 0.20)
        self.assertGreater(score_novel, 0.80)
        self.assertGreater(score_novel, score_known * 3.0)

    def test_04_novelty_threshold_and_prediction(self):
        """Verify 95th percentile validation calibration and prediction flag."""
        rng = np.random.RandomState(42)
        X_train = rng.normal(0, 1.0, (50, 32))
        y_train = np.array(["normal"] * 25 + ["defect"] * 25)

        detector = NoveltyDetector(k=5)
        detector.fit(X_train, y_train)

        X_val = rng.normal(0, 1.0, (100, 32))
        thresh = detector.calibrate_threshold(X_val, percentile=95.0)

        scores, is_novel = detector.predict_is_novel(X_val)
        self.assertEqual(len(scores), 100)
        self.assertEqual(len(is_novel), 100)
        # By definition of 95th percentile, ~5% of validation samples should exceed threshold
        self.assertAlmostEqual(float(np.mean(is_novel)), 0.05, delta=0.03)

    def test_05_held_out_evaluation_cache_integrity(self):
        """Verify held-out class results cache structure and honest numbers."""
        eval_path = os.path.join(ROOT_DIR, "cache", "novelty_held_out_eval.json")
        if os.path.exists(eval_path):
            import json
            with open(eval_path, "r") as f:
                data = json.load(f)

            self.assertIn("scratch_experiment", data)
            self.assertIn("rust_experiment", data)
            self.assertIn("production_detector", data)

            scratch = data["scratch_experiment"]
            self.assertEqual(scratch["held_out_class"], "scratch")
            self.assertIn("fraction_flagged_novel_or_uncertain_pct", scratch)
            self.assertIn("false_novel_rate_pct", scratch)
            self.assertGreater(scratch["fraction_flagged_novel_or_uncertain_pct"], 80.0)

            rust = data["rust_experiment"]
            self.assertEqual(rust["held_out_class"], "rust")
            self.assertIn("fraction_flagged_novel_or_uncertain_pct", rust)
            self.assertIn("false_novel_rate_pct", rust)
            self.assertGreater(rust["fraction_flagged_novel_or_uncertain_pct"], 80.0)

    def test_06_production_detector_load(self):
        """Verify production novelty detector loads and has valid parameters."""
        det_path = os.path.join(ROOT_DIR, "cache", "novelty_detector.joblib")
        if os.path.exists(det_path):
            detector = get_production_novelty_detector()
            self.assertIsInstance(detector, NoveltyDetector)
            self.assertEqual(detector.k, 5)
            self.assertGreater(detector.threshold_95th, 0.0)
            self.assertEqual(len(detector.known_classes), 5)


if __name__ == "__main__":
    unittest.main()
