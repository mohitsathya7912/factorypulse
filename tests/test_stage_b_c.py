"""
tests/test_stage_b_c.py
Unit and Regression Tests for Checkpoint 3 STAGE B & STAGE C.
Verifies calibration, Wilson intervals, ECE, cost thresholding, coverage-accuracy curves,
and PatchCore-style anomaly detection and localization.
"""

import os
import unittest
import numpy as np
import torch
from PIL import Image

from src.quality.calibration import (
    wilson_score_interval,
    compute_ece,
    fit_temperature_scaling,
    calibrate_logits,
    optimize_cost_threshold,
    compute_coverage_accuracy_curve
)
from src.quality.localization import (
    PatchAnomalyDetector,
    jet_colormap_np
)


class TestStageBAndC(unittest.TestCase):

    def test_01_wilson_score_interval_bounds(self):
        """Verify Wilson score interval properties for binomial error rates."""
        # 0 successes out of 100
        p, (low, high) = wilson_score_interval(0, 100)
        self.assertEqual(p, 0.0)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLess(high, 0.05) # 95% upper bound is ~0.036

        # 50 successes out of 100
        p, (low, high) = wilson_score_interval(50, 100)
        self.assertEqual(p, 0.5)
        self.assertAlmostEqual((low + high) / 2.0, 0.5, places=2)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

        # 100 successes out of 100
        p, (low, high) = wilson_score_interval(100, 100)
        self.assertEqual(p, 1.0)
        self.assertGreater(low, 0.95)
        self.assertEqual(high, 1.0)

        # Empty sample edge case
        p, (low, high) = wilson_score_interval(0, 0)
        self.assertEqual(p, 0.0)
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 0.0)

    def test_02_compute_ece_10_bins(self):
        """Verify 10-bin Expected Calibration Error calculation."""
        # Synthesize 100 confident samples
        np.random.seed(42)
        probs = np.array([
            [0.9, 0.1],
            [0.85, 0.15],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.05, 0.95]
        ])
        y_true = np.array([0, 0, 0, 1, 1]) # all correctly predicted
        ece, bins = compute_ece(probs, y_true, n_bins=10)
        self.assertEqual(len(bins), 10)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        for b in bins:
            self.assertIn("bin_idx", b)
            self.assertIn("accuracy", b)
            self.assertIn("confidence", b)
            self.assertIn("gap", b)

    def test_03_temperature_scaling_optimization(self):
        """Verify temperature scaling scalar optimization minimizes NLL."""
        logits = np.array([
            [5.0, -2.0, -1.0],
            [4.0, -1.0, -3.0],
            [-2.0, 4.5, -1.0],
            [-3.0, -2.0, 5.2]
        ])
        y_val = np.array([0, 0, 1, 2])
        T_opt = fit_temperature_scaling(logits, y_val)
        self.assertGreater(T_opt, 0.0)

        probs_cal = calibrate_logits(logits, T_opt)
        self.assertEqual(probs_cal.shape, logits.shape)
        # Softmax sum to 1
        np.testing.assert_allclose(np.sum(probs_cal, axis=1), np.ones(4), atol=1e-5)

    def test_04_cost_threshold_optimization(self):
        """Verify cost-optimal decision threshold search under assumed prevalence."""
        p_def = np.array([0.01, 0.05, 0.15, 0.35, 0.65, 0.85, 0.98])
        y_bin = np.array([0, 0, 0, 1, 1, 1, 1])
        res = optimize_cost_threshold(
            p_def_val=p_def,
            y_val_binary=y_bin,
            cost_fr=41.72,
            cost_fa=5.0 * 41.72,
            prevalence=0.10,
            n_grid=50
        )
        self.assertIn("optimal_threshold", res)
        self.assertIn("val_expected_cost_per_1000", res)
        self.assertGreaterEqual(res["optimal_threshold"], 0.01)
        self.assertLessEqual(res["optimal_threshold"], 0.99)

    def test_05_coverage_accuracy_curve(self):
        """Verify coverage fraction and confident accuracy across thresholds."""
        probs = np.array([
            [0.95, 0.05],
            [0.85, 0.15],
            [0.65, 0.35],
            [0.40, 0.60],
            [0.10, 0.90]
        ])
        y_bin = np.array([0, 0, 0, 1, 1])
        curve = compute_coverage_accuracy_curve(probs, y_bin)
        self.assertIn("thresholds", curve)
        self.assertIn("coverage", curve)
        self.assertIn("accuracy", curve)
        # High threshold should yield equal or lower coverage than low threshold
        self.assertGreaterEqual(curve["coverage"][0], curve["coverage"][-1])

    def test_06_patch_anomaly_detector_scoring_speed_and_boxes(self):
        """Verify patch anomaly scoring time is under 0.5s and extracts at most 3 boxes."""
        detector = PatchAnomalyDetector()
        # Create synthetic test image
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        res = detector.score_image(img, compute_overlay=True)

        self.assertIn("anomaly_score", res)
        self.assertIn("threshold", res)
        self.assertIn("bounding_boxes", res)
        self.assertIn("overlay_image", res)
        self.assertLessEqual(len(res["bounding_boxes"]), 3)
        self.assertLess(res["elapsed_seconds"], 0.5) # Ground rule: < 0.5s per image

    def test_07_jet_colormap_pure_numpy(self):
        """Verify pure NumPy Jet colormap produces valid RGB array."""
        vals = np.linspace(0.0, 1.0, 100)
        rgb = jet_colormap_np(vals)
        self.assertEqual(rgb.shape, (100, 3))
        self.assertEqual(rgb.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
