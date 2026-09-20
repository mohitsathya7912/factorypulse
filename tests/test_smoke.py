"""
tests/test_smoke.py
Comprehensive Smoke Test Suite for FactoryPulse Integration & Hardening.
Covers:
1. Schema validation, alias remapping, and missing column detection.
2. Production formulas: effective capacity, rework inflation, utilization,
   bottleneck selection with tie-breaking rules, and throughput lost to rework.
3. Economics formulas: revenue, cost breakdown, profit, gross margin, COPQ,
   cost per good unit, and What-If delta calculations.
4. End-to-end Demo-mode quality model run with selective classification and triage queue.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from config.schema import (
    remap_columns,
    validate_columns,
    INSPECTION_EXPECTED,
    PROCESS_EXPECTED,
    PRODUCTION_EXPECTED,
    ECONOMIC_EXPECTED,
    COLUMN_MAPPING
)
from src.production.flow import (
    calculate_station_metrics,
    analyze_production_line,
    ProductionAnalyzer
)
from src.economics.model import (
    calculate_economics,
    simulate_what_if,
    EconomicModel
)
from src.data.loader import DataLoader
from src.quality.classifier import QualityClassifier


# ==============================================================================
# 1. SMOKE TESTS: SCHEMA VALIDATION & COLUMN REMAPPING
# ==============================================================================

def test_smoke_schema_column_remapping_aliases():
    """Verify that organizer column aliases across all 4 domains map to canonical names."""
    # Inspection Aliases
    raw_insp = pd.DataFrame(columns=["part_id", "lot_id", "product_type", "inspection_time", "has_defect", "defect_family", "img"])
    remapped_insp = remap_columns(raw_insp, dataset_type="inspection")
    for col in ["unit_id", "batch_id", "variant", "timestamp", "label", "defect_category", "image_path"]:
        assert col in remapped_insp.columns, f"Expected canonical '{col}' in remapped inspection columns"

    # Process Aliases
    raw_proc = pd.DataFrame(columns=["lot_id", "datetime", "temp_c", "feed_rate", "pressure_bar", "vib_level"])
    remapped_proc = remap_columns(raw_proc, dataset_type="process")
    for col in ["batch_id", "timestamp", "temperature", "speed", "pressure", "vibration"]:
        assert col in remapped_proc.columns, f"Expected canonical '{col}' in remapped process columns"

    # Production Aliases
    raw_prod = pd.DataFrame(columns=["station_id", "lot_id", "ct", "nominal_capacity", "wip_count", "down_time", "arrivals", "completed", "reworked", "scrapped"])
    remapped_prod = remap_columns(raw_prod, dataset_type="production")
    for col in ["station", "batch_id", "cycle_time", "capacity", "wip", "downtime_hours", "units_in", "units_out", "rework_units", "scrap_units"]:
        assert col in remapped_prod.columns, f"Expected canonical '{col}' in remapped production columns"

    # Economic Aliases
    raw_econ = pd.DataFrame(columns=["selling_price", "raw_material", "labor_rate", "overhead_rate", "scrap_penalty", "rework_cost_per_unit", "downtime_penalty"])
    remapped_econ = remap_columns(raw_econ, dataset_type="economic")
    for col in ["unit_price", "material_cost", "labor_cost", "overhead_cost", "scrap_cost", "rework_cost", "downtime_cost_per_hour"]:
        assert col in remapped_econ.columns, f"Expected canonical '{col}' in remapped economic columns"


def test_smoke_schema_validation_success_and_failure():
    """Verify validate_columns accurately passes valid frames and returns descriptive error reports for missing ones."""
    # Success frame
    df_valid = pd.DataFrame(columns=INSPECTION_EXPECTED)
    is_valid, missing, msg = validate_columns(df_valid, INSPECTION_EXPECTED, "Inspection")
    assert is_valid is True
    assert len(missing) == 0
    assert "Validation Success" in msg

    # Incomplete frame
    df_incomplete = pd.DataFrame(columns=["unit_id", "batch_id", "variant"])
    is_valid_bad, missing_bad, msg_bad = validate_columns(df_incomplete, INSPECTION_EXPECTED, "Inspection")
    assert is_valid_bad is False
    assert "label" in missing_bad
    assert "defect_category" in missing_bad
    assert "timestamp" in missing_bad
    assert "Validation Warning" in msg_bad
    assert "Missing 3 required column(s)" in msg_bad


# ==============================================================================
# 2. SMOKE TESTS: PRODUCTION FLOW & BOTTLENECK DETECTION
# ==============================================================================

def test_smoke_production_formulas_and_station_metrics():
    """
    Verify exact production formulas:
    - effective_capacity = (60 / cycle_time) * (1 - downtime_fraction)
    - rework_rate = rework_units / units_out
    - effective_load = units_in * (1 + rework_rate)
    - utilization = effective_load / (effective_capacity * hours)
    """
    station_input = {
        "station": "Station 3 - Thermal Curing",
        "cycle_time": 0.6,       # 0.6 minutes (36 seconds)
        "downtime_hours": 0.8,   # 0.8 hours out of 8 hours -> downtime_fraction = 0.10
        "units_in": 100,
        "units_out": 90,
        "rework_units": 9,       # rework_rate = 9 / 90 = 0.10
        "scrap_units": 1,
        "wip": 45
    }
    m = calculate_station_metrics(station_input, hours=8.0)

    # 1. downtime_fraction = 0.8 / 8.0 = 0.10
    assert pytest.approx(m["downtime_fraction"], 1e-3) == 0.10

    # 2. effective_capacity = (60 / 0.6) * (1 - 0.10) = 100 * 0.90 = 90.0 UPH
    assert pytest.approx(m["effective_capacity"], 1e-2) == 90.0

    # 3. rework_rate = 9 / 90 = 0.10 (10%)
    assert pytest.approx(m["rework_rate"], 1e-3) == 0.10

    # 4. effective_load = 100 * (1 + 0.10) = 110.0
    assert pytest.approx(m["effective_load"], 1e-2) == 110.0

    # 5. utilization = 110.0 / (90.0 * 8.0) = 110 / 720 = 0.1528
    assert pytest.approx(m["utilization"], 1e-3) == 110.0 / 720.0


def test_smoke_bottleneck_detection_and_tie_breaking():
    """
    Verify bottleneck detection rules:
    - Highest utilization wins
    - Equal utilization broken by higher WIP
    - Equal utilization and equal WIP broken by longer cycle_time
    - Evidence string and rework loss calculation
    """
    stations = [
        # Station A: normal load
        {"station": "Stn_A", "cycle_time": 0.4, "downtime_hours": 0.0, "units_in": 80, "units_out": 80, "rework_units": 0, "wip": 10},
        # Station B: high utilization 80% (load=80, cap=100*1.0=100/hr, avail=800 -> 80/800 = 10%)
        # Let's craft Station Tied 1: cap = 100/hr, load = 80 -> util = 80 / (100 * 8) = 10%
        # Let's craft tied stations with high load:
        # Station B: capacity = 50 UPH (ct=1.2, dt=0). hours=8 -> avail = 400. load = 320 -> util = 320/400 = 80.0%, WIP = 25, ct=1.2
        {"station": "Stn_B", "cycle_time": 1.2, "downtime_hours": 0.0, "units_in": 320, "units_out": 320, "rework_units": 0, "wip": 25},
        # Station C: capacity = 50 UPH, avail = 400. rework_rate = 32/320 = 10% -> load = 352 -> util = 88.0%, WIP = 55
        {"station": "Stn_C", "cycle_time": 1.2, "downtime_hours": 0.0, "units_in": 320, "units_out": 320, "rework_units": 32, "wip": 55},
        # Station D: capacity = 50 UPH, avail = 400. load = 280 -> util = 70.0%, WIP = 25
        {"station": "Stn_D", "cycle_time": 1.2, "downtime_hours": 0.0, "units_in": 280, "units_out": 280, "rework_units": 0, "wip": 25},
    ]
    # For Station C: rework_units=32, units_out=320 -> rework_rate = 0.10 -> effective_load = 320 * 1.10 = 352 -> util = 352/400 = 88.0%!
    res = analyze_production_line(stations, hours=8.0)
    bn = res["bottleneck_station"]
    assert bn is not None
    assert bn["station"] == "Stn_C"
    assert "Stn_C: utilization 88%, WIP 55, rework adds 10% extra load" in res["bottleneck_evidence"]

    # Now test EXACT tie in utilization to verify WIP tie-breaker
    # Set rework=0 for C so load=320 -> exactly 80.0% utilization for both B, C, D
    stations_tied = [
        {"station": "Stn_B", "cycle_time": 1.2, "downtime_hours": 0.0, "units_in": 320, "units_out": 320, "rework_units": 0, "wip": 25},
        {"station": "Stn_C", "cycle_time": 1.2, "downtime_hours": 0.0, "units_in": 320, "units_out": 320, "rework_units": 0, "wip": 60}, # Wins via WIP!
        {"station": "Stn_D", "cycle_time": 1.2, "downtime_hours": 0.0, "units_in": 320, "units_out": 320, "rework_units": 0, "wip": 25},
    ]
    res_tied = analyze_production_line(stations_tied, hours=8.0)
    assert res_tied["bottleneck_station"]["station"] == "Stn_C"

    # Test EXACT tie in utilization AND WIP to verify cycle_time tie-breaker
    stations_ct_tie = [
        {"station": "Stn_Fast", "cycle_time": 1.0, "downtime_hours": 0.0, "units_in": 320, "units_out": 320, "rework_units": 0, "wip": 30},
        {"station": "Stn_Slow", "cycle_time": 2.0, "downtime_hours": 0.0, "units_in": 160, "units_out": 160, "rework_units": 0, "wip": 30}, # Same util (80%), same WIP (30), longer CT (2.0 vs 1.0)
    ]
    # Stn_Fast: cap = 60 UPH, 8 hrs = 480. 320 / 480 = 66.67%
    # Stn_Slow: cap = 30 UPH, 8 hrs = 240. 160 / 240 = 66.67%
    res_ct = analyze_production_line(stations_ct_tie, hours=8.0)
    assert res_ct["bottleneck_station"]["station"] == "Stn_Slow"

    # Throughput lost due to rework check
    # effective_capacity = 50. rework_rate = 0.10. without rework = 50. with rework = 50 / 1.10 = 45.45. lost = 4.55
    stations_rew = [
        {"station": "Stn_X", "cycle_time": 1.2, "downtime_hours": 0.0, "units_in": 100, "units_out": 100, "rework_units": 10, "wip": 10}
    ]
    res_rew = analyze_production_line(stations_rew, hours=8.0)
    assert pytest.approx(res_rew["throughput_without_rework"], 1e-2) == 50.0
    assert pytest.approx(res_rew["throughput_with_rework"], 1e-2) == round(50.0 / 1.10, 2)
    assert pytest.approx(res_rew["throughput_lost_due_to_rework"], 1e-2) == round(50.0 - (50.0 / 1.10), 2)


# ==============================================================================
# 3. SMOKE TESTS: ECONOMICS FORMULAS & WHAT-IF DELTAS
# ==============================================================================

def test_smoke_economics_formulas():
    """
    Verify exact economics accounting formulas:
    - Revenue = Good units x Unit price
    - Cost = Material + Labor + Overhead + Scrap + Rework + Downtime
    - Profit = Revenue - Cost
    - Margin = Profit / Revenue
    - Cost per good unit = Cost / Good units
    - COPQ = Scrap + Rework + Downtime
    """
    good_units = 1000
    unit_price = 85.0
    mat = 32000.0
    lab = 3600.0
    ovh = 240.0
    scr = 420.0
    rew = 370.0
    dt = 150.0

    econ = calculate_economics(
        good_units=good_units,
        unit_price=unit_price,
        material_cost=mat,
        labor_cost=lab,
        overhead_cost=ovh,
        scrap_cost=scr,
        rework_cost=rew,
        downtime_cost=dt
    )

    # 1. Revenue
    expected_rev = 1000 * 85.0 # 85,000.0
    assert econ["revenue"] == expected_rev

    # 2. Total Cost
    expected_cost = mat + lab + ovh + scr + rew + dt # 36,780.0
    assert econ["cost"] == expected_cost

    # 3. Profit = Revenue - Cost
    expected_profit = expected_rev - expected_cost # 48,220.0
    assert econ["profit"] == expected_profit

    # 4. Margin = Profit / Revenue
    expected_margin = expected_profit / expected_rev
    assert pytest.approx(econ["margin"], 1e-4) == expected_margin

    # 5. Cost per good unit = Cost / good_units
    expected_cpg = expected_cost / good_units
    assert pytest.approx(econ["cost_per_good_unit"], 1e-2) == expected_cpg

    # 6. Cost of poor quality = Scrap + Rework + Downtime
    expected_copq = scr + rew + dt # 940.0
    assert econ["cost_of_poor_quality"] == expected_copq


def test_smoke_economics_what_if_simulation_deltas():
    """Verify simulate_what_if returns exact deltas vs baseline and is labeled 'Simulated / advisory'."""
    base_prod = {"line_throughput": 100.0}
    base_econ = {
        "good_units": 800,
        "unit_price": 85.0,
        "revenue": 68000.0,
        "material_cost": 25600.0,
        "labor_cost": 3600.0,
        "overhead_cost": 240.0,
        "scrap_cost": 840.0,
        "rework_cost": 555.0,
        "downtime_cost": 300.0,
        "cost": 31135.0,
        "profit": 36865.0,
        "margin": 0.5421,
        "cost_of_poor_quality": 1695.0
    }

    sim = simulate_what_if(
        baseline_production=base_prod,
        baseline_economics=base_econ,
        defect_reduction_pct=20.0,
        capacity_boost_pct=10.0,
        downtime_reduction_pct=15.0
    )

    assert sim["label"] == "Simulated / advisory"
    deltas = sim["deltas"]
    baseline = sim["baseline"]
    simulated = sim["simulated"]

    # Verify deltas equal (simulated - baseline)
    assert pytest.approx(deltas["throughput_delta"], 1e-2) == simulated["throughput"] - baseline["throughput"]
    assert deltas["good_units_delta"] == simulated["good_units"] - baseline["good_units"]
    assert pytest.approx(deltas["revenue_delta"], 1e-2) == simulated["revenue"] - baseline["revenue"]
    assert pytest.approx(deltas["cost_delta"], 1e-2) == simulated["cost"] - baseline["cost"]
    assert pytest.approx(deltas["profit_delta"], 1e-2) == simulated["profit"] - baseline["profit"]
    assert pytest.approx(deltas["margin_delta"], 1e-4) == simulated["margin"] - baseline["margin"]
    assert pytest.approx(deltas["copq_delta"], 1e-2) == simulated["copq"] - baseline["copq"]


# ==============================================================================
# 4. SMOKE TESTS: END-TO-END DEMO-MODE QUALITY MODEL RUN
# ==============================================================================

def test_smoke_end_to_end_demo_quality_model_run():
    """
    Load synthetic Demo dataset, train two-stage QualityClassifier with GroupSplit by batch_id,
    evaluate held-out metrics, test selective classification with uncertainty threshold,
    and generate triage queue.
    """
    loader = DataLoader()
    df_insp, df_proc, df_prod, df_econ, meta = loader.load_all(mode="Demo")

    assert len(df_insp) > 0, "Demo inspection data must load"
    assert len(df_proc) > 0, "Demo process data must load"
    assert meta["is_synthetic"] is True

    with tempfile.TemporaryDirectory() as tmp_model_dir:
        clf = QualityClassifier(model_dir=tmp_model_dir)

        # Train on Demo data
        metrics = clf.train_or_load(
            df_insp=df_insp,
            df_proc=df_proc,
            force_retrain=True,
            uncertainty_threshold=0.70
        )

        assert metrics["status"] in ["trained", "loaded", "success"]
        assert "Tabular Telemetry" in metrics["feature_type"]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0
        assert 0.0 <= metrics["false_reject_rate"] <= 1.0
        assert 0.0 <= metrics["false_accept_rate"] <= 1.0
        assert len(metrics["confusion_matrix"]) == 2
        assert len(metrics["test_batches"]) > 0

        # Predict with thresholding on a sample of batches
        sample_df = df_insp.head(50)
        preds = clf.predict_dataframe(sample_df, df_proc=df_proc, uncertainty_threshold=0.70)

        assert len(preds) == 50
        assert "is_uncertain" in preds.columns
        assert "predicted_status" in preds.columns
        assert "confidence" in preds.columns
        assert "triage_recommendation" in preds.columns

        # Verify uncertainty behavior
        for _, row in preds.iterrows():
            if row["confidence"] < 0.70:
                assert row["is_uncertain"] is True
                assert row["predicted_status"] == "UNCERTAIN"
                assert "Inspector" in row["triage_recommendation"]
