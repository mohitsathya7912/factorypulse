"""
tests/test_stage_g.py
Stage G: Hardening & Invariant Integration Tests

Verifies:
1. Split has zero overlap between train, val, and test partitions.
2. is_defective handles string labels, 0/1 numeric, booleans, and collections.
3. Calibrated ECE is finite and not worse than uncalibrated ECE.
4. Cost-optimal decision threshold is in [0, 1].
5. Heatmap produced by patch-anomaly detector has shape (256, 256).
6. Perturbation suite results table contains every expected perturbation condition.
7. Bottleneck utilization value is identical across all tabs (Overview, Production, and AI Insights).
"""

import os
import json
import re
import pytest
import numpy as np
import pandas as pd
from PIL import Image

from config.schema import is_defective
from src.data.loader import DataLoader
from src.quality.classifier import QualityClassifier, optimize_cost_threshold
from src.quality.localization import PatchAnomalyDetector
from src.production.flow import analyze_production_line
from src.insights.advisor import DecisionAdvisor


class TestStageGHardeningInvariants:
    def test_01_split_no_overlap(self):
        """Verify train, val, and test splits have strictly zero image overlap."""
        split_path = os.path.join("cache", "split.csv")
        assert os.path.exists(split_path), f"Split cache file not found at {split_path}"

        df_split = pd.read_csv(split_path)
        assert "split" in df_split.columns
        path_col = "path" if "path" in df_split.columns else "image_path"

        train_paths = set(df_split[df_split["split"] == "train"][path_col])
        val_paths = set(df_split[df_split["split"] == "val"][path_col])
        test_paths = set(df_split[df_split["split"] == "test"][path_col])

        # Assert all splits are populated
        assert len(train_paths) > 0, "Train partition must not be empty."
        assert len(val_paths) > 0, "Validation partition must not be empty."
        assert len(test_paths) > 0, "Test partition must not be empty."

        # Verify pairwise intersections are strictly empty (zero leakage)
        train_val_overlap = train_paths.intersection(val_paths)
        train_test_overlap = train_paths.intersection(test_paths)
        val_test_overlap = val_paths.intersection(test_paths)

        assert len(train_val_overlap) == 0, f"Found {len(train_val_overlap)} overlapping images between train and val."
        assert len(train_test_overlap) == 0, f"Found {len(train_test_overlap)} overlapping images between train and test."
        assert len(val_test_overlap) == 0, f"Found {len(val_test_overlap)} overlapping images between val and test."

    def test_02_is_defective_handles_all_types(self):
        """Verify is_defective handles string labels, 0/1 integers, floats, booleans, Series, and arrays."""
        # 1. Strings
        assert is_defective("defective") is True
        assert is_defective("DEFECTIVE") is True
        assert is_defective("defect") is True
        assert is_defective("fail") is True
        assert is_defective("good") is False
        assert is_defective("GOOD") is False
        assert is_defective("normal") is False
        assert is_defective("pass") is False
        assert is_defective("0") is False
        assert is_defective("1") is True
        assert is_defective("true") is True
        assert is_defective("false") is False
        assert is_defective("none") is False
        assert is_defective("") is False

        # 2. Numeric 0 / 1
        assert is_defective(1) is True
        assert is_defective(0) is False
        assert is_defective(1.0) is True
        assert is_defective(0.0) is False

        # 3. Booleans
        assert is_defective(True) is True
        assert is_defective(False) is False

        # 4. Pandas Series
        s_input = pd.Series(["good", "defective", 1, 0, True, False, "1", "0"])
        s_expected = pd.Series([False, True, True, False, True, False, True, False])
        s_out = is_defective(s_input)
        pd.testing.assert_series_equal(s_out, s_expected, check_names=False)

        # 5. NumPy Array
        arr_input = np.array([1, 0, 1, 0])
        arr_out = is_defective(arr_input)
        np.testing.assert_array_equal(arr_out, np.array([True, False, True, False]))

    def test_03_calibrated_ece_finite_and_not_worse(self):
        """Verify calibrated Expected Calibration Error (ECE) is finite and not worse than uncalibrated."""
        clf = QualityClassifier()
        clf.load_organizer_models()
        metrics = clf.train_organizer_resnet18(dev_mode=True)

        ece_uncal = metrics["ece_uncalibrated"]
        ece_cal = metrics["ece_calibrated"]

        assert np.isfinite(ece_cal), f"Calibrated ECE is non-finite: {ece_cal}"
        assert ece_cal >= 0.0, f"Calibrated ECE must be non-negative: {ece_cal}"
        # Calibrated ECE must be less than or equal to uncalibrated (with float tolerance)
        assert ece_cal <= ece_uncal + 1e-5, (
            f"Calibrated ECE ({ece_cal:.6f}) should not be worse than uncalibrated ECE ({ece_uncal:.6f})"
        )

    def test_04_cost_threshold_in_zero_to_one(self):
        """Verify cost-optimal decision threshold search returns an operating threshold in [0, 1]."""
        p_def = np.array([0.02, 0.08, 0.15, 0.35, 0.65, 0.88, 0.99])
        y_bin = np.array([0, 0, 0, 1, 1, 1, 1])

        res = optimize_cost_threshold(
            p_def_val=p_def,
            y_val_binary=y_bin,
            cost_fr=41.72,
            cost_fa=208.6,
            prevalence=0.10,
            n_grid=50
        )
        assert "optimal_threshold" in res
        thresh = res["optimal_threshold"]
        assert 0.0 <= thresh <= 1.0, f"Optimal threshold {thresh} is outside [0, 1]."
        assert np.isfinite(thresh)

    def test_05_heatmap_has_shape_256x256(self):
        """Verify patch anomaly scoring produces heatmap array and overlay image of shape (256, 256)."""
        detector = PatchAnomalyDetector()
        img = Image.new("RGB", (256, 256), color=(140, 140, 140))
        res = detector.score_image(img, compute_overlay=True)

        assert "heatmap" in res, "Missing 'heatmap' key in score_image return dictionary."
        heatmap = res["heatmap"]
        assert isinstance(heatmap, np.ndarray), "Heatmap must be a numpy ndarray."
        assert heatmap.shape == (256, 256), f"Expected heatmap shape (256, 256), got {heatmap.shape}."

        assert "overlay_image" in res
        overlay = res["overlay_image"]
        assert isinstance(overlay, Image.Image), "Overlay must be a PIL Image."
        assert overlay.size == (256, 256), f"Expected overlay size (256, 256), got {overlay.size}."

    def test_06_perturbation_table_has_every_expected_row(self):
        """Verify perturbation robustness evaluation contains every expected perturbation condition."""
        eval_path = os.path.join("cache", "robust_model_eval.json")
        assert os.path.exists(eval_path), f"Robust evaluation file not found at {eval_path}"

        with open(eval_path, "r") as f:
            data = json.load(f)

        results_by_pert = data["baseline_evaluation"]["results_by_pert"]

        expected_conditions = [
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

        for cond in expected_conditions:
            assert cond in results_by_pert, f"Missing required perturbation condition: '{cond}'"
            entry = results_by_pert[cond]
            assert "accuracy" in entry
            assert "f1_macro" in entry
            assert "false_reject_rate" in entry
            assert "false_accept_rate" in entry
            assert "mean_confidence" in entry

    def test_07_utilization_value_same_across_all_tabs(self):
        """Verify bottleneck utilization is consistent across Overview (Tab 1), Production (Tab 2), and AI Insights (Tab 6)."""
        loader = DataLoader()
        _, _, df_prod, _, _ = loader.load_all(mode="Demo")
        prod_res = analyze_production_line(df_prod)

        bn = prod_res["bottleneck_station"]
        # Tab 1 and Tab 2 calculation
        bn_util_tab1_tab2 = round(float(bn["utilization"]) * 100.0, 1)

        # Tab 6: DecisionAdvisor Impact Chain Step 3
        advisor = DecisionAdvisor(production_results=prod_res)
        impact_chain = advisor.compute_impact_chain()
        step3 = next(s for s in impact_chain["steps"] if s["stage"] == 3)
        # Parse utilization percentage from string e.g. "10.7% Utilization"
        match = re.search(r"([\d\.]+)%", step3["metric"])
        assert match is not None, f"Could not parse utilization percentage from metric: {step3['metric']}"
        bn_util_tab6_chain = round(float(match.group(1)), 1)

        # Tab 6: Computed Insights evidence object
        insights = advisor.generate_computed_insights()
        bn_insight = next(ins for ins in insights if ins["category"] == "Bottleneck & Flow")
        bn_util_tab6_evidence = round(float(bn_insight["evidence"]["utilization_pct"]), 1)

        # Verify all 3 match exactly
        assert bn_util_tab1_tab2 == bn_util_tab6_chain, (
            f"Tab 1/2 utilization ({bn_util_tab1_tab2}%) does not match Tab 6 Impact Chain ({bn_util_tab6_chain}%)"
        )
        assert bn_util_tab1_tab2 == bn_util_tab6_evidence, (
            f"Tab 1/2 utilization ({bn_util_tab1_tab2}%) does not match Tab 6 Evidence ({bn_util_tab6_evidence}%)"
        )
