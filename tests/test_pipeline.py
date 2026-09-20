"""
tests/test_pipeline.py
Unit and Integration Tests for FactoryPulse Analytical Modules.
Verifies Quality AI, Production Flow, Economic P&L, and Root-Cause engines.
"""

import os
import unittest
import pandas as pd
import numpy as np

from config.schema import standardize_columns, DatasetOrigin, is_defective
from src.data.loader import DataLoader
from src.quality.classifier import QualityClassifier
from src.quality.localization import render_defect_visualization
from src.production.flow import ProductionAnalyzer
from src.economics.model import EconomicModel
from src.rootcause.engine import RootCauseEngine
from src.insights.advisor import DecisionAdvisor

class TestFactoryPulsePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up data loader and test datasets."""
        cls.loader = DataLoader(organizer_dir="./data/organizer", sample_dir="./data/sample")
        cls.df_insp, cls.df_proc, cls.df_prod, cls.df_econ, cls.origin = cls.loader.load_all(mode="Demo")

    def test_01_data_loading_and_provenance(self):
        """Verify data loader correctly tracks synthetic vs organizer provenance."""
        self.assertIsNotNone(self.origin)
        self.assertGreater(len(self.df_insp), 0)
        self.assertGreater(len(self.df_proc), 0)
        self.assertGreater(len(self.df_prod), 0)
        self.assertGreater(len(self.df_econ), 0)
        self.assertIn("is_defective", self.df_insp.columns)
        self.assertIn("batch_id", self.df_proc.columns)
        self.assertIn("nominal_capacity_uph", self.df_prod.columns)

    def test_02_quality_classifier(self):
        """Verify two-stage classifier, ECE calculation, and three-tier decision routing."""
        clf = QualityClassifier(model_dir="./models")
        metrics = clf.train_or_load(self.df_insp)
        self.assertIn("f1", metrics)
        self.assertIn("ece", metrics)
        self.assertGreaterEqual(metrics["ece"], 0.0)

        # Test evaluation of a sample part
        sample_row = self.df_insp.iloc[0]
        eval_res = clf.evaluate_unit(sample_row)
        self.assertIn("action", eval_res)
        self.assertIn("calibrated_prob", eval_res)
        self.assertIn("status", eval_res)
        self.assertIn(eval_res["status"], ["ACCEPT", "REJECT", "UNCERTAIN", "NOVEL"])

    def test_03_localization_and_heatmap(self):
        """Verify rendering bounding boxes when annotated and approximate heatmaps when unannotated."""
        img_annotated, prov1 = render_defect_visualization("nonexistent.png", bbox_str="[20, 30, 40, 50]")
        self.assertIn("Verified Spatial Annotation", prov1)

        img_heatmap, prov2 = render_defect_visualization("nonexistent.png", force_heatmap=True)
        self.assertIn("Approximate", prov2)

    def test_04_production_flow_and_bottleneck(self):
        """Verify station utilization, rework loop multiplier, and Little's Law WIP sanity check."""
        analyzer = ProductionAnalyzer(self.df_prod, total_line_defects=int(is_defective(self.df_insp["is_defective"]).sum()))
        res = analyzer.analyze_line()

        self.assertIn("bottleneck_station", res)
        self.assertIn("constrained_throughput_uph", res)
        self.assertIn("littles_law_expected_wip", res)
        self.assertGreater(res["constrained_throughput_uph"], 0)
        self.assertGreater(len(res["stations"]), 0)

        # Bottleneck station should have highest effective utilization
        bn = res["bottleneck_station"]
        max_util = max(st["total_effective_utilization_pct"] for st in res["stations"])
        self.assertEqual(bn["total_effective_utilization_pct"], max_util)

    def test_05_economic_model_and_simulation(self):
        """Verify financial accounting, gross margin, and sensitivity simulation."""
        econ = EconomicModel(self.df_econ)
        pnl = econ.calculate_pnl(
            total_units_produced=len(self.df_insp),
            scrap_units=15,
            rework_units=35,
            downtime_minutes=45.0
        )
        self.assertIn("gross_revenue", pnl)
        self.assertIn("operating_profit", pnl)
        self.assertIn("gross_margin_pct", pnl)
        self.assertGreater(pnl["gross_revenue"], 0)

        # Simulation
        sim_res = econ.simulate_margin_recovery(pnl, defect_reduction_pct=25.0, throughput_increase_pct=5.0)
        self.assertIn("margin_gain_points", sim_res)
        self.assertGreater(sim_res["simulated_profit"], pnl["operating_profit"])

    def test_06_root_cause_engine(self):
        """Verify hypothesis testing, drift detection, and SHAP factor attribution."""
        joined_df = self.loader.get_joined_batch_data()
        self.assertGreater(len(joined_df), 0)

        rc = RootCauseEngine(joined_df)
        rc_res = rc.analyze_factors()
        self.assertEqual(rc_res["status"], "success")
        self.assertIn("ranked_factors", rc_res)
        self.assertIn("disclaimer", rc_res)
        self.assertIn("statistical associations", rc_res["disclaimer"])

        if rc_res["ranked_factors"]:
            top = rc_res["ranked_factors"][0]
            self.assertIn("parameter", top)
            self.assertIn("correlation", top)
            self.assertIn("shap_importance", top)

    def test_07_advisory_advisor(self):
        """Verify recommendation synthesis generates structured advisory items."""
        analyzer = ProductionAnalyzer(self.df_prod, total_line_defects=int(is_defective(self.df_insp["is_defective"]).sum()))
        prod_results = analyzer.analyze_line()
        econ = EconomicModel(self.df_econ)
        econ_results = econ.calculate_pnl(len(self.df_insp), 15, 35, 45.0)
        rc = RootCauseEngine(self.loader.get_joined_batch_data())
        rc_results = rc.analyze_factors()

        advisor = DecisionAdvisor({}, prod_results, econ_results, rc_results)
        advisories = advisor.generate_advisory_insights()
        self.assertGreater(len(advisories), 0)
        for adv in advisories:
            self.assertIn("recommended_action", adv)
            self.assertIn("SIMULATED / ADVISORY", adv["status"])

if __name__ == "__main__":
    unittest.main()
