"""
tests/test_quality_ai.py
Comprehensive unit and integration tests for FactoryPulse Quality AI module.
Verifies image detection, group splitting, two-stage classification, test metrics,
uncertainty thresholding, and joblib model persistence.
"""

import os
import shutil
import unittest
import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.quality.classifier import QualityClassifier, has_organizer_images

class TestQualityAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader(organizer_dir="./data/organizer", sample_dir="./data/sample")
        cls.df_insp, cls.df_proc, cls.df_prod, cls.df_econ, cls.meta = cls.loader.load_all(mode="Demo")
        cls.test_models_dir = "./models_test_scratch"
        os.makedirs(cls.test_models_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_models_dir):
            shutil.rmtree(cls.test_models_dir, ignore_errors=True)

    def test_01_organizer_images_check(self):
        """Verify has_organizer_images returns True for organizer folder with images, and False for nonexistent directory."""
        self.assertTrue(has_organizer_images("./data/organizer"))
        self.assertFalse(has_organizer_images("./nonexistent_directory_xyz"))

    def test_02_group_split_prevents_leakage(self):
        """Verify group splitting by batch_id ensures zero batch overlap between train and test."""
        clf = QualityClassifier(model_dir=self.test_models_dir)
        metrics = clf.train(self.df_insp, self.df_proc, uncertainty_threshold=0.70)
        
        train_idx = [i for i in range(len(self.df_insp)) if i not in clf.test_indices]
        train_batches = set(self.df_insp.iloc[train_idx]["batch_id"].unique())
        test_batches = set(self.df_insp.iloc[clf.test_indices]["batch_id"].unique())

        # Zero leakage: test batches must not appear in train batches
        self.assertEqual(len(train_batches.intersection(test_batches)), 0)
        self.assertGreater(len(train_batches), 0)
        self.assertGreater(len(test_batches), 0)

    def test_03_test_metrics_calculation(self):
        """Verify accuracy, precision, recall, F1, FRR, and FAR are strictly computed."""
        clf = QualityClassifier(model_dir=self.test_models_dir)
        metrics = clf.train(self.df_insp, self.df_proc, uncertainty_threshold=0.70)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("false_reject_rate", metrics)
        self.assertIn("false_accept_rate", metrics)
        self.assertIn("confusion_matrix", metrics)

        # Sanity check rates
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["false_reject_rate"], 0.0)
        self.assertLessEqual(metrics["false_reject_rate"], 1.0)
        self.assertGreaterEqual(metrics["false_accept_rate"], 0.0)
        self.assertLessEqual(metrics["false_accept_rate"], 1.0)

        # Mathematical verification of FRR and FAR
        tn = metrics["tn"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        tp = metrics["tp"]

        expected_frr = fp / max(1, (tn + fp))
        expected_far = fn / max(1, (tp + fn))
        self.assertAlmostEqual(metrics["false_reject_rate"], expected_frr, places=3)
        self.assertAlmostEqual(metrics["false_accept_rate"], expected_far, places=3)

    def test_04_uncertainty_flag_and_triage(self):
        """Verify UNCERTAIN flag correctly isolates borderline predictions."""
        clf = QualityClassifier(model_dir=self.test_models_dir)
        clf.train(self.df_insp, self.df_proc, uncertainty_threshold=0.65)

        pred_df_low = clf.predict_dataframe(self.df_insp, self.df_proc, uncertainty_threshold=0.55)
        pred_df_high = clf.predict_dataframe(self.df_insp, self.df_proc, uncertainty_threshold=0.85)

        # Higher uncertainty threshold means more samples flagged as UNCERTAIN
        uncertain_low = int(pred_df_low["is_uncertain"].sum())
        uncertain_high = int(pred_df_high["is_uncertain"].sum())
        self.assertGreaterEqual(uncertain_high, uncertain_low)

        # Verify triage recommendation string exists
        self.assertTrue(all(pred_df_low["triage_recommendation"].str.len() > 0))

    def test_05_joblib_persistence(self):
        """Verify joblib model save and reload without recomputation."""
        clf1 = QualityClassifier(model_dir=self.test_models_dir)
        metrics1 = clf1.train(self.df_insp, self.df_proc, uncertainty_threshold=0.70)

        clf2 = QualityClassifier(model_dir=self.test_models_dir)
        loaded = clf2.load_model()
        self.assertTrue(loaded)
        self.assertEqual(clf2.test_metrics["accuracy"], metrics1["accuracy"])
        self.assertEqual(clf2.test_metrics["f1"], metrics1["f1"])
        self.assertEqual(clf2.test_batches, metrics1["test_batches"])

    def test_06_stage2_multiclass_defect_families(self):
        """Verify Stage 2 defect category classifier works when multiple categories exist."""
        clf = QualityClassifier(model_dir=self.test_models_dir)
        metrics = clf.train(self.df_insp, self.df_proc, uncertainty_threshold=0.70)
        
        stage2 = metrics.get("stage2", {})
        self.assertIn("available", stage2)
        if stage2["available"]:
            self.assertIn("classes", stage2)
            self.assertGreaterEqual(len(stage2["classes"]), 2)
            self.assertIn("confusion_matrix", stage2)
            self.assertIn("f1_macro", stage2)

    def test_07_organizer_detection_and_mode_loading(self):
        """Verify DataLoader detects organizer mode and scans organizer image classes."""
        loader = DataLoader(organizer_dir="./data/organizer")
        self.assertEqual(loader.detect_mode(), "Organizer")
        self.assertTrue(loader.has_organizer_images())
        imgs = loader.scan_organizer_images()
        for c in ["normal", "crack", "hole", "rust", "scratch"]:
            self.assertIn(c, imgs)
            self.assertGreater(len(imgs[c]), 0)

    def test_08_organizer_inspection_table_and_resnet18(self):
        """Verify Organizer inspection table schema and ResNet18 classifier evaluation."""
        loader = DataLoader(organizer_dir="./data/organizer")
        df_insp = loader.build_organizer_inspection_table()
        # Exactly 3 columns: label, defect_category, image_path (no other columns)
        self.assertEqual(list(df_insp.columns), ["label", "defect_category", "image_path"])
        
        # Verify labels: good if normal else defective
        normal_mask = df_insp["defect_category"] == "normal"
        self.assertTrue((df_insp.loc[normal_mask, "label"] == "good").all())
        self.assertTrue((df_insp.loc[~normal_mask, "label"] == "defective").all())

        # Test ResNet18 classifier evaluation
        clf = QualityClassifier(model_dir=self.test_models_dir, organizer_dir="./data/organizer")
        metrics = clf.train_organizer_resnet18(use_full_dataset=False, uncertainty_threshold=0.70)
        
        for k in ["accuracy", "precision", "recall", "f1", "f1_macro", "false_reject_rate", "false_accept_rate", "confusion_matrix"]:
            self.assertIn(k, metrics)
            if k != "confusion_matrix":
                self.assertGreaterEqual(metrics[k], 0.0)
                self.assertLessEqual(metrics[k], 1.0)


if __name__ == "__main__":
    unittest.main()
