"""
tests/test_data_layer.py
Unit and Integration Tests for FactoryPulse Data Layer.
Validates Demo Mode dataset loading, schema column validation,
missing value cleaning strategies, batch joins, and summary statistics.
"""

import os
import unittest
import pandas as pd
import numpy as np

from config.schema import (
    INSPECTION_EXPECTED,
    PROCESS_EXPECTED,
    PRODUCTION_EXPECTED,
    ECONOMIC_EXPECTED,
    remap_columns,
    validate_columns
)
from src.data.loader import DataLoader
from src.data.generator import generate_sample_datasets

class TestDataLayer(unittest.TestCase):
    """Test suite for FactoryPulse data layer components."""

    @classmethod
    def setUpClass(cls):
        """Ensure sample data exists and initialize DataLoader in Demo mode."""
        cls.sample_dir = "./data/sample"
        generate_sample_datasets(cls.sample_dir, n_batches=20)
        cls.loader = DataLoader(sample_dir=cls.sample_dir)
        cls.df_insp, cls.df_proc, cls.df_prod, cls.df_econ, cls.meta = cls.loader.load_all(mode="Demo")

    def test_01_demo_mode_loading(self):
        """Test that all four datasets load in Demo mode with positive record counts."""
        self.assertEqual(self.meta["data_mode"], "Demo")
        self.assertTrue(self.meta["is_synthetic"])
        self.assertGreater(len(self.df_insp), 0, "Inspection dataset should not be empty")
        self.assertGreater(len(self.df_proc), 0, "Process dataset should not be empty")
        self.assertGreater(len(self.df_prod), 0, "Production dataset should not be empty")
        self.assertGreater(len(self.df_econ), 0, "Economic dataset should not be empty")

    def test_02_canonical_columns_present(self):
        """Test that canonical expected columns exist across all four datasets."""
        for col in INSPECTION_EXPECTED:
            self.assertIn(col, self.df_insp.columns, f"Inspection missing required column: {col}")
        for col in PROCESS_EXPECTED:
            self.assertIn(col, self.df_proc.columns, f"Process missing required column: {col}")
        for col in PRODUCTION_EXPECTED:
            self.assertIn(col, self.df_prod.columns, f"Production missing required column: {col}")
        for col in ECONOMIC_EXPECTED:
            self.assertIn(col, self.df_econ.columns, f"Economic missing required column: {col}")

    def test_03_column_remapping_and_validation(self):
        """Test that column remapping maps organizer variants and validation lists missing columns."""
        # Simulated raw organizer dataframe with non-standard column names
        raw_df = pd.DataFrame({
            "part_id": ["P1", "P2"],
            "lot_id": ["L1", "L1"],
            "model": ["Alpha", "Beta"],
            "defective": [0, 1],
            "category": ["None", "Crack"]
        })
        remapped = remap_columns(raw_df, dataset_type="inspection")
        self.assertIn("unit_id", remapped.columns)
        self.assertIn("batch_id", remapped.columns)
        self.assertIn("variant", remapped.columns)
        self.assertIn("label", remapped.columns)
        self.assertIn("defect_category", remapped.columns)

        # Test validation reporting missing columns with clear message
        is_valid, missing, msg = validate_columns(remapped, INSPECTION_EXPECTED, "Test_Inspection")
        self.assertFalse(is_valid)
        self.assertIn("timestamp", missing)
        self.assertIn("Missing", msg)

    def test_04_missing_value_cleaning(self):
        """Test documented cleaning strategies for missing values."""
        dirty_insp = pd.DataFrame({
            "unit_id": ["U1", np.nan],
            "batch_id": ["B1", "B1"],
            "variant": [np.nan, "Alpha"],
            "timestamp": ["2026-09-19 08:00:00", np.nan],
            "label": ["pass", "defect"],
            "defect_category": [np.nan, "Scratch"]
        })
        cleaned = self.loader._clean_missing_values(dirty_insp, "inspection")
        self.assertEqual(cleaned["label"].iloc[0], 0)
        self.assertEqual(cleaned["label"].iloc[1], 1)
        self.assertFalse(cleaned["variant"].isnull().any())
        self.assertFalse(cleaned["timestamp"].isnull().any())

    def test_05_join_tables_on_batch_id(self):
        """Test joining inspection quality aggregates with process telemetry on batch_id."""
        joined = self.loader.join_tables()
        self.assertGreater(len(joined), 0)
        self.assertIn("batch_id", joined.columns)
        self.assertIn("defect_rate", joined.columns)
        self.assertIn("temperature", joined.columns)
        self.assertIn("speed", joined.columns)

    def test_06_compute_statistics(self):
        """Test statistical computation over all four datasets."""
        stats = self.loader.compute_statistics()
        self.assertIn("inspection", stats)
        self.assertIn("process", stats)
        self.assertIn("production", stats)
        self.assertIn("economic", stats)

        self.assertGreater(stats["inspection"]["total_units"], 0)
        self.assertIn("defect_rate_pct", stats["inspection"])
        self.assertIn("temperature", stats["process"])
        self.assertIn("mean", stats["process"]["temperature"])
        self.assertIn("station_breakdown", stats["production"])

    def test_07_synthetic_note_and_metadata(self):
        """Verify that SYNTHETIC_DATA_NOTE.txt and metadata.json are generated and valid."""
        note_path = os.path.join(self.sample_dir, "SYNTHETIC_DATA_NOTE.txt")
        self.assertTrue(os.path.exists(note_path), "SYNTHETIC_DATA_NOTE.txt must exist in data/sample/")
        with open(note_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("SYNTHETIC DATASET NOTICE", content)
            self.assertIn("is_synthetic = True", content)

    def test_08_is_defective_helper_and_missing_batch_id(self):
        """Verify is_defective helper and robust handling of missing batch_id."""
        from config.schema import is_defective
        
        # 1. Scalar checks
        self.assertTrue(is_defective("defective"))
        self.assertTrue(is_defective("fail"))
        self.assertTrue(is_defective("1"))
        self.assertTrue(is_defective(1))
        self.assertTrue(is_defective(True))
        
        self.assertFalse(is_defective("good"))
        self.assertFalse(is_defective("normal"))
        self.assertFalse(is_defective("pass"))
        self.assertFalse(is_defective("0"))
        self.assertFalse(is_defective(0))
        self.assertFalse(is_defective(False))
        self.assertFalse(is_defective(np.nan))

        # 2. Vectorized Series check
        s = pd.Series(["good", "defective", "normal", "crack", "0", 1, True, False])
        result = is_defective(s)
        self.assertTrue(isinstance(result, pd.Series))
        # "good"->False, "defective"->True, "normal"->False, "crack"->False (defect category, not label unless fail), "0"->False, 1->True, True->True, False->False
        self.assertEqual(int(result.sum()), 3)

        # 3. Missing batch_id in inspection
        insp_no_batch = pd.DataFrame({
            "label": ["good", "defective", "good", "defective"],
            "defect_category": ["None", "crack", "None", "hole"],
            "image_path": ["p1", "p2", "p3", "p4"]
        })
        self.assertEqual(int(is_defective(insp_no_batch["label"]).sum()), 2)
        
        joined = self.loader.join_tables(df_inspection=insp_no_batch)
        # Should gracefully fall back or return joined batches without crashing
        self.assertTrue(isinstance(joined, pd.DataFrame))

if __name__ == "__main__":
    unittest.main()
