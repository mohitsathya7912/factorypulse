"""
tests/test_stage_e.py
Unit and integration tests for Hackathon 3.0 Checkpoint 3 STAGE E:
- Direct Benjamini-Hochberg FDR p-value correction
- Bootstrap 95% confidence intervals for Spearman rho
- Statistical Root-Cause factor ranking with FDR filtering
- Defect family breakdown with top statistical associations
- Bootstrapped what-if gross margin forecast (median and 10-90% range)
"""

import pytest
import numpy as np
import pandas as pd

from src.rootcause.engine import (
    benjamini_hochberg,
    bootstrap_spearman_ci,
    build_batch_table,
    RootCauseEngine
)
from src.economics.model import (
    calculate_economics,
    simulate_what_if,
    bootstrap_what_if_margin
)
from src.data.loader import DataLoader


class TestBenjaminiHochberg:
    """Unit tests for the direct Benjamini-Hochberg (BH) FDR implementation."""

    def test_empty_and_single_input(self):
        assert benjamini_hochberg([]) == []
        assert benjamini_hochberg([0.04]) == [0.04]

    def test_monotonicity_and_ordering(self):
        # 4 p-values: [0.01, 0.04, 0.03, 0.20]
        # Sorted: p_(1)=0.01, p_(2)=0.03, p_(3)=0.04, p_(4)=0.20
        # Multipliers (m/k): 4/1=4, 4/2=2, 4/3=1.333, 4/4=1
        # Raw q: [0.04, 0.06, 0.0533, 0.20]
        # Step-down min: q_(2)=min(0.06, 0.0533)=0.0533
        # Sorted adjusted: [0.04, 0.0533, 0.0533, 0.20]
        # Restored original order: [0.04, 0.0533, 0.0533, 0.20]
        raw_p = [0.01, 0.04, 0.03, 0.20]
        adj_p = benjamini_hochberg(raw_p)

        assert len(adj_p) == 4
        assert adj_p[0] <= adj_p[2] <= adj_p[1] <= adj_p[3]
        assert round(adj_p[0], 4) == 0.04
        assert round(adj_p[2], 4) == round(0.04 * 4 / 3, 4)
        assert round(adj_p[1], 4) == round(0.04 * 4 / 3, 4)  # enforced by step-down
        assert round(adj_p[3], 4) == 0.20

    def test_upper_bound_at_one(self):
        raw_p = [0.8, 0.9, 0.95]
        adj_p = benjamini_hochberg(raw_p)
        assert all(q <= 1.0 for q in adj_p)


class TestBootstrapSpearmanCI:
    """Unit tests for the bootstrap confidence interval function."""

    def test_small_sample_fallback(self):
        ci_low, ci_high = bootstrap_spearman_ci(np.array([1, 2]), np.array([3, 4]))
        assert ci_low == -1.0
        assert ci_high == 1.0

    def test_positive_correlation_ci(self):
        rng = np.random.RandomState(42)
        x = np.linspace(10, 50, 25)
        y = 2.5 * x + rng.normal(0, 5, size=25)

        ci_low, ci_high = bootstrap_spearman_ci(x, y, n_boot=500, seed=42)
        assert -1.0 <= ci_low <= ci_high <= 1.0
        assert ci_low > 0.5  # Strong positive correlation interval

    def test_reproducibility(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([2, 1, 4, 5, 6, 8, 7, 9, 11, 10])

        ci1 = bootstrap_spearman_ci(x, y, n_boot=200, seed=42)
        ci2 = bootstrap_spearman_ci(x, y, n_boot=200, seed=42)
        assert ci1 == ci2


class TestRootCauseEngineStageE:
    """Tests for Stage E additions to RootCauseEngine."""

    @pytest.fixture
    def demo_batch_table(self):
        loader = DataLoader()
        df_insp, df_proc, df_prod, _, _ = loader.load_all(mode="Demo")
        return build_batch_table(df_insp, df_proc, df_prod)

    def test_analyze_factors_bh_and_ci(self, demo_batch_table):
        engine = RootCauseEngine(demo_batch_table, is_synthetic=True)
        results = engine.analyze_factors()

        assert results["status"] == "success"
        assert results["sample_size"] == len(demo_batch_table)
        assert "ranked_factors" in results

        factors = results["ranked_factors"]
        assert len(factors) > 0

        for f in factors:
            assert "spearman_rho" in f
            assert "spearman_p_value" in f
            assert "spearman_p_adjusted" in f
            assert "bootstrap_ci" in f
            assert "ci_str" in f
            assert "significance_status" in f
            assert "evidence_label" in f

            # Verify BH adjusted p >= raw p
            assert f["spearman_p_adjusted"] >= f["spearman_p_value"] - 1e-6

            # If not significant, evidence label must be 'not significant'
            if not f["is_significant"]:
                assert f["evidence_label"] == "not significant"

    def test_analyze_defect_families(self, demo_batch_table):
        engine = RootCauseEngine(demo_batch_table, is_synthetic=True)
        results = engine.analyze_defect_families()

        assert "family_associations" in results
        assert "top_associations" in results
        assert len(results["top_associations"]) > 0

        for top_item in results["top_associations"]:
            assert "defect_family" in top_item
            assert "top_parameter" in top_item
            assert "spearman_rho" in top_item
            assert "bootstrap_ci" in top_item
            assert "ci_str" in top_item
            assert "raw_p_value" in top_item
            assert "adjusted_p_value" in top_item
            assert "significance_status" in top_item
            assert "evidence_label" in top_item


class TestBootstrapWhatIfMargin:
    """Unit tests for bootstrapped what-if gross margin forecast."""

    @pytest.fixture
    def setup_economics(self):
        loader = DataLoader()
        df_insp, df_proc, df_prod, df_econ, _ = loader.load_all(mode="Demo")
        batch_df = build_batch_table(df_insp, df_proc, df_prod)

        econ = calculate_economics(
            good_units=950,
            unit_price=85.00,
            material_cost=32000.0,
            labor_cost=3600.0,
            overhead_cost=240.0,
            scrap_cost=2100.0,
            rework_cost=1110.0,
            downtime_cost=1800.0
        )
        return batch_df, econ

    def test_bootstrap_forecast_output_structure(self, setup_economics):
        batch_df, econ = setup_economics
        res = bootstrap_what_if_margin(
            batch_df=batch_df,
            baseline_economics=econ,
            defect_reduction_pct=20.0,
            capacity_boost_pct=10.0,
            downtime_reduction_pct=15.0,
            n_boot=300,
            seed=42
        )

        assert "median_margin_pct" in res
        assert "p10_margin_pct" in res
        assert "p90_margin_pct" in res
        assert "range_str" in res
        assert res["label"] == "Simulated / advisory"

        # Check ordering: p10 <= median <= p90
        assert res["p10_margin_pct"] <= res["median_margin_pct"] <= res["p90_margin_pct"]
        assert 40.0 <= res["median_margin_pct"] <= 70.0

    def test_empty_batch_df_fallback(self):
        econ = {"margin": 0.52}
        res = bootstrap_what_if_margin(
            batch_df=pd.DataFrame(),
            baseline_economics=econ
        )
        assert res["median_margin_pct"] == 52.0
        assert res["p10_margin_pct"] == 52.0
        assert res["p90_margin_pct"] == 52.0
        assert res["label"] == "Simulated / advisory"
