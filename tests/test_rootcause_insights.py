"""
tests/test_rootcause_insights.py
Unit and integration tests for FactoryPulse Root-Cause Engine and AI Insights Advisor.
Verifies non-parametric statistical associations, evidence labeling, impact chain,
and rule-based recommendation generation.
"""

import unittest
import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.rootcause.engine import RootCauseEngine, build_batch_table
from src.insights.advisor import DecisionAdvisor
from src.production.flow import analyze_production_line
from src.economics.model import calculate_economics

class TestRootCauseAndInsights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader(organizer_dir="./data/organizer", sample_dir="./data/sample")
        cls.df_insp, cls.df_proc, cls.df_prod, cls.df_econ, cls.meta = cls.loader.load_all(mode="Demo")

    def test_01_build_batch_table(self):
        """Verify batch table accurately consolidates defect rates and telemetry."""
        batch_df = build_batch_table(self.df_insp, self.df_proc, self.df_prod)
        self.assertGreater(len(batch_df), 0)
        self.assertIn("batch_id", batch_df.columns)
        self.assertIn("defect_rate", batch_df.columns)
        self.assertIn("total_inspected", batch_df.columns)
        self.assertIn("temperature", batch_df.columns)
        self.assertIn("speed", batch_df.columns)
        
        # Verify defect rate is bounded [0, 1]
        self.assertTrue((batch_df["defect_rate"] >= 0.0).all())
        self.assertTrue((batch_df["defect_rate"] <= 1.0).all())

    def test_02_root_cause_engine_statistical_metrics(self):
        """Verify Spearman correlation, Mann-Whitney U, and 2-sigma outlier detection."""
        batch_df = build_batch_table(self.df_insp, self.df_proc)
        rc = RootCauseEngine(batch_df, is_synthetic=True)
        res = rc.analyze_factors()

        self.assertEqual(res["status"], "success")
        self.assertIn("ranked_factors", res)
        self.assertGreater(len(res["ranked_factors"]), 0)
        self.assertIn("disclaimer", res)
        self.assertIn("statistical associations", res["disclaimer"])
        self.assertIn("synthetic data are built into the generator", res["demo_note"])

        top = res["ranked_factors"][0]
        self.assertIn("spearman_rho", top)
        self.assertIn("spearman_p_value", top)
        self.assertIn("defect_rate_delta_pts", top)
        self.assertIn("mwu_p_value", top)
        self.assertIn("evidence_label", top)
        self.assertIn(top["evidence_label"], ["Strong", "Moderate", "Weak"])
        self.assertIn("upper_limit_2sigma", top)
        self.assertIn("lower_limit_2sigma", top)
        self.assertIn("affected_batches", top)

    def test_03_small_sample_warning(self):
        """Verify warning is issued when batch count is small."""
        batch_df = build_batch_table(self.df_insp, self.df_proc)
        small_df = batch_df.head(4).copy()
        
        rc_small = RootCauseEngine(small_df)
        res_small = rc_small.analyze_factors()
        self.assertIsNotNone(res_small["sample_warning"])
        self.assertIn("Very small sample size", res_small["sample_warning"])

    def test_04_computed_insights_sentences(self):
        """Verify insight sentences are derived strictly from computed numbers."""
        batch_df = build_batch_table(self.df_insp, self.df_proc)
        rc = RootCauseEngine(batch_df)
        rc_res = rc.analyze_factors()

        prod_res = analyze_production_line([
            {"station": "S1", "cycle_time": 0.4, "downtime_hours": 0.1, "units_in": 100, "units_out": 95, "rework_units": 5, "scrap_units": 2, "wip": 10}
        ])
        econ_res = calculate_economics(98, 85.0, 3200, 450, 240, 84, 92, 150)

        advisor = DecisionAdvisor(
            quality_summary={"accuracy": 0.95, "false_reject_rate": 0.02, "false_accept_rate": 0.04, "uncertain_count": 5, "uncertain_fraction": 0.05},
            production_results=prod_res,
            economic_results=econ_res,
            rootcause_results=rc_res,
            df_insp=self.df_insp
        )

        insights = advisor.generate_computed_insights()
        self.assertGreater(len(insights), 0)
        
        for ins in insights:
            self.assertIn("statement", ins)
            self.assertIn("evidence", ins)
            self.assertIn("confidence", ins)
            self.assertTrue(len(ins["statement"]) > 10)

    def test_05_impact_chain_end_to_end(self):
        """Verify 6-stage impact chain computation and insufficient data fallbacks."""
        batch_df = build_batch_table(self.df_insp, self.df_proc)
        rc = RootCauseEngine(batch_df)
        rc_res = rc.analyze_factors()

        prod_res = analyze_production_line([
            {"station": "Station 3 - Thermal Curing", "cycle_time": 0.6, "downtime_hours": 0.5, "units_in": 1000, "units_out": 900, "rework_units": 70, "scrap_units": 30, "wip": 42}
        ])
        econ_res = calculate_economics(970, 85.0, 32000, 4500, 2400, 1260, 1295, 750)

        advisor = DecisionAdvisor(
            production_results=prod_res,
            economic_results=econ_res,
            rootcause_results=rc_res,
            df_insp=self.df_insp
        )

        chain = advisor.compute_impact_chain()
        self.assertEqual(len(chain["steps"]), 6)
        
        # Verify 6 stages
        expected_titles = [
            "Top Defect Family",
            "Likely Contributing Factor",
            "Primary Line Bottleneck",
            "Throughput Lost",
            "Cost of Poor Quality",
            "Gross Margin Impact"
        ]
        actual_titles = [s["title"] for s in chain["steps"]]
        self.assertEqual(actual_titles, expected_titles)

        # Test fallback on empty inputs
        advisor_empty = DecisionAdvisor()
        chain_empty = advisor_empty.compute_impact_chain()
        self.assertEqual(len(chain_empty["steps"]), 6)
        self.assertEqual(chain_empty["steps"][0]["name"], "insufficient data")
        self.assertEqual(chain_empty["steps"][1]["name"], "insufficient data")

    def test_06_advisory_recommendations(self):
        """Verify rule-based advisory recommendations tagged SIMULATED / ADVISORY."""
        batch_df = build_batch_table(self.df_insp, self.df_proc)
        rc_res = RootCauseEngine(batch_df).analyze_factors()
        prod_res = analyze_production_line([
            {"station": "Stamping", "cycle_time": 0.5, "downtime_hours": 0.2, "units_in": 500, "units_out": 480, "rework_units": 20, "scrap_units": 5, "wip": 15}
        ])
        econ_res = calculate_economics(475, 85.0, 15000, 2000, 1000, 210, 370, 300)

        advisor = DecisionAdvisor(
            production_results=prod_res,
            economic_results=econ_res,
            rootcause_results=rc_res,
            df_insp=self.df_insp
        )
        recs = advisor.generate_advisory_insights()
        self.assertGreater(len(recs), 0)
        for r in recs:
            self.assertIn("recommended_action", r)
            self.assertIn("SIMULATED / ADVISORY", r["status"])


if __name__ == "__main__":
    unittest.main()
