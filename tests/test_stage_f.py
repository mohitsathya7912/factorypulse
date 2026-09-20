"""
Tests for Stage F:
- Evidence panel for inspected units (verdict, calibrated confidence, top-2 class probabilities,
  anomaly score, novelty score, localization heatmap, and plain-language 'why' without LLM).
- Deterministic 'why' generation across all 4 operational states (GOOD, DEFECTIVE, UNCERTAIN, NOVEL).
- Color consistency tokens and evidence dictionary schema.
"""

import os
import pytest
import numpy as np
from src.quality.classifier import QualityClassifier, generate_plain_language_why


class TestStageFEvidencePanel:
    def test_01_plain_language_why_novel(self):
        """Verify NOVEL state explanation formatting and mathematical values."""
        why = generate_plain_language_why(
            status="NOVEL",
            confidence=0.88,
            threshold=0.70,
            top1_cls="crack",
            top1_prob=0.88,
            top2_cls="hole",
            top2_prob=0.08,
            anomaly_score=2.85,
            anomaly_thresh=2.06,
            novelty_score=0.215,
            novelty_thresh=0.1188
        )
        assert "NOVEL / UNKNOWN" in why
        assert "0.215" in why
        assert "0.119" in why or "0.1188" in why
        assert "k=5" in why
        assert "out-of-distribution" in why

    def test_02_plain_language_why_uncertain(self):
        """Verify UNCERTAIN state explanation formatting when confidence < threshold."""
        why = generate_plain_language_why(
            status="UNCERTAIN",
            confidence=0.58,
            threshold=0.70,
            top1_cls="normal",
            top1_prob=0.58,
            top2_cls="crack",
            top2_prob=0.32,
            anomaly_score=1.45,
            anomaly_thresh=2.06,
            novelty_score=0.045,
            novelty_thresh=0.1188
        )
        assert "UNCERTAIN" in why
        assert "0.58" in why
        assert "0.70" in why
        assert "normal (58.0%)" in why
        assert "crack (32.0%)" in why
        assert "triage queue" in why

    def test_03_plain_language_why_defective(self):
        """Verify DEFECTIVE state explanation formatting with anomaly score."""
        why = generate_plain_language_why(
            status="DEFECTIVE",
            confidence=0.96,
            threshold=0.70,
            top1_cls="crack",
            top1_prob=0.96,
            top2_cls="hole",
            top2_prob=0.03,
            anomaly_score=3.12,
            anomaly_thresh=2.06,
            novelty_score=0.052,
            novelty_thresh=0.1188
        )
        assert "DEFECTIVE (CRACK)" in why
        assert "0.96" in why
        assert "3.12" in why
        assert "exceeds normal limit 2.06" in why

    def test_04_plain_language_why_good(self):
        """Verify GOOD (CONFORMING) state explanation formatting with normal anomaly score."""
        why = generate_plain_language_why(
            status="GOOD",
            confidence=0.92,
            threshold=0.70,
            top1_cls="normal",
            top1_prob=0.92,
            top2_cls="scratch",
            top2_prob=0.05,
            anomaly_score=1.22,
            anomaly_thresh=2.06,
            novelty_score=0.031,
            novelty_thresh=0.1188
        )
        assert "GOOD (CONFORMING)" in why
        assert "0.92" in why
        assert "normal 92.0%" in why
        assert "within normal threshold 2.06" in why

    def test_05_predict_image_evidence_dictionary_schema(self):
        """Verify that predict_image output contains all required Stage F evidence keys."""
        img_path = "data/organizer/train/crack/crack_00000.png"
        if not os.path.exists(img_path):
            pytest.skip(f"Test image {img_path} not available.")

        clf = QualityClassifier()
        clf.load_organizer_models()
        res = clf.predict_image(img_path, uncertainty_threshold=0.70, compute_localization=True)

        expected_keys = [
            "status", "badge_label", "badge_color", "category_text",
            "is_defective", "is_uncertain", "is_novel",
            "novelty_score", "novelty_threshold",
            "confidence", "confidence_pct", "defect_category",
            "prob_good", "prob_defective", "probabilities",
            "top2_probabilities", "top1_class", "top1_prob", "top2_class", "top2_prob",
            "threshold", "anomaly_info", "anomaly_score", "anomaly_threshold",
            "plain_language_why"
        ]
        for k in expected_keys:
            assert k in res, f"Missing evidence key '{k}' in predict_image output."

        # Verify plain-language why is populated and deterministic
        assert isinstance(res["plain_language_why"], str)
        assert len(res["plain_language_why"]) > 20
        assert "{" not in res["plain_language_why"], "Unformatted format string found in rationale."

        # Verify top-2 probabilities format
        top2 = res["top2_probabilities"]
        assert len(top2) == 2
        assert isinstance(top2[0], tuple)
        assert top2[0][1] >= top2[1][1], "Top-2 probabilities must be ordered descending."

    def test_06_deterministic_and_no_external_llm(self):
        """Verify generate_plain_language_why is 100% deterministic and requires no LLM."""
        args = {
            "status": "DEFECTIVE",
            "confidence": 0.85,
            "threshold": 0.70,
            "top1_cls": "hole",
            "top1_prob": 0.85,
            "top2_cls": "crack",
            "top2_prob": 0.10,
            "anomaly_score": 2.45,
            "anomaly_thresh": 2.06,
            "novelty_score": 0.065,
            "novelty_thresh": 0.1188
        }
        res1 = generate_plain_language_why(**args)
        res2 = generate_plain_language_why(**args)
        assert res1 == res2
        assert isinstance(res1, str)
