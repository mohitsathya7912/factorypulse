"""
tests/test_production_economics.py
Pytest Test Suite for FactoryPulse Production and Economics Calculation Modules.
Includes hand-computed toy datasets to verify all mathematical formulas.

Formulas Tested:
- effective_capacity = (60 / cycle_time) * (1 - downtime_fraction)
- rework_rate = rework_units / units_out
- effective_load = units_in * (1 + rework_rate)
- utilization = effective_load / (effective_capacity * hours)
- Bottleneck = highest utilization (tie-break: higher WIP, then longer cycle_time)
- Throughput lost due to rework = throughput without rework - throughput with rework
- Revenue = Good units * Unit price
- Cost = Material + Labor + Overhead + Scrap + Rework + Downtime
- Profit = Revenue - Cost, Margin = Profit / Revenue
- Cost per good unit = Cost / Good units
- Cost of poor quality (COPQ) = Scrap + Rework + Downtime
- What-If deltas vs baseline labeled 'Simulated / advisory'
"""

import pytest
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

# ==============================================================================
# 1. PRODUCTION METRICS (HAND-COMPUTED TOY DATA)
# ==============================================================================

def test_hand_computed_station_metrics():
    """
    Hand-computed single station test:
    - cycle_time = 0.5 minutes (30 sec)
    - downtime_fraction = 0.10 (10% downtime)
    - hours = 8.0
    - units_in = 800
    - units_out = 780
    - rework_units = 78
    - wip = 15

    Hand Calculations:
    - effective_capacity = (60 / 0.5) * (1 - 0.10) = 120 * 0.9 = 108.0 units/hr
    - rework_rate = 78 / 780 = 0.10 (10%)
    - effective_load = 800 * (1 + 0.10) = 880.0 units
    - total_capacity_8h = 108.0 * 8 = 864.0 units
    - utilization = 880.0 / 864.0 = 1.018518... (~101.85%)
    """
    toy_station = {
        "station": "Station 1",
        "cycle_time": 0.5,
        "downtime_fraction": 0.10,
        "units_in": 800,
        "units_out": 780,
        "rework_units": 78,
        "scrap_units": 20,
        "wip": 15
    }

    result = calculate_station_metrics(toy_station, hours=8.0)

    assert result["effective_capacity"] == pytest.approx(108.0, rel=1e-3)
    assert result["rework_rate"] == pytest.approx(0.10, rel=1e-3)
    assert result["effective_load"] == pytest.approx(880.0, rel=1e-3)
    assert result["utilization"] == pytest.approx(880.0 / 864.0, rel=1e-3)
    assert result["wip"] == 15


def test_hand_computed_bottleneck_and_evidence():
    """
    Hand-computed 3-station line to verify bottleneck identification and evidence string:
    - Station 1: util = 1.0185, WIP = 15
    - Station 2: util = 1.3477, WIP = 42, rework_rate = 0.15  -> Clear Bottleneck!
    - Station 3: util = 0.5987, WIP = 8

    Station 2 Details:
    - cycle_time = 0.6 min, downtime_fraction = 0.20
    - effective_capacity = (60 / 0.6) * 0.8 = 80.0 UPH
    - units_in = 750, units_out = 700, rework_units = 105
    - rework_rate = 105 / 700 = 0.15
    - effective_load = 750 * 1.15 = 862.5
    - 8h capacity = 80.0 * 8 = 640.0
    - utilization = 862.5 / 640.0 = 1.347656...
    """
    toy_line = [
        {
            "station": "Station 1",
            "cycle_time": 0.5,
            "downtime_fraction": 0.10,
            "units_in": 800,
            "units_out": 780,
            "rework_units": 78,
            "scrap_units": 20,
            "wip": 15
        },
        {
            "station": "Station 2",
            "cycle_time": 0.6,
            "downtime_fraction": 0.20,
            "units_in": 750,
            "units_out": 700,
            "rework_units": 105,
            "scrap_units": 50,
            "wip": 42
        },
        {
            "station": "Station 3",
            "cycle_time": 0.4,
            "downtime_fraction": 0.05,
            "units_in": 650,
            "units_out": 640,
            "rework_units": 32,
            "scrap_units": 10,
            "wip": 8
        }
    ]

    res = analyze_production_line(toy_line, hours=8.0)

    # Verify Station 2 is chosen as primary bottleneck
    bn = res["bottleneck_station"]
    assert bn["station"] == "Station 2"
    assert bn["utilization"] == pytest.approx(862.5 / 640.0, rel=1e-3)
    assert bn["wip"] == 42
    assert bn["rework_rate"] == pytest.approx(0.15, rel=1e-3)

    # Verify evidence string format
    # Expected: "Station 2: utilization 135%, WIP 42, rework adds 15% extra load"
    assert "Station 2: utilization" in res["bottleneck_evidence"]
    assert "WIP 42" in res["bottleneck_evidence"]
    assert "rework adds 15% extra load" in res["bottleneck_evidence"]

    # Line throughput = capacity of the most constrained station = 80.0 UPH
    assert res["line_throughput"] == pytest.approx(80.0, rel=1e-3)


def test_bottleneck_tie_breaking():
    """
    Verify tie-breaking:
    1. Highest utilization
    2. Ties broken by higher WIP
    3. Ties broken by longer cycle time
    """
    # Case A: Same utilization, Station B has higher WIP (45 vs 20)
    line_tie_wip = [
        {"station": "StA", "cycle_time": 0.5, "downtime_fraction": 0.0, "units_in": 120, "units_out": 120, "rework_units": 0, "wip": 20},
        {"station": "StB", "cycle_time": 0.5, "downtime_fraction": 0.0, "units_in": 120, "units_out": 120, "rework_units": 0, "wip": 45}
    ]
    res_wip = analyze_production_line(line_tie_wip, hours=1.0)
    assert res_wip["bottleneck_station"]["station"] == "StB"

    # Case B: Same utilization and same WIP, Station D has longer cycle time (0.6 min vs 0.4 min)
    # StC: cycle_time = 0.4 min (150 uph) -> units_in = 150 -> util = 1.0, WIP = 30
    # StD: cycle_time = 0.6 min (100 uph) -> units_in = 100 -> util = 1.0, WIP = 30
    line_tie_ct = [
        {"station": "StC", "cycle_time": 0.4, "downtime_fraction": 0.0, "units_in": 150, "units_out": 150, "rework_units": 0, "wip": 30},
        {"station": "StD", "cycle_time": 0.6, "downtime_fraction": 0.0, "units_in": 100, "units_out": 100, "rework_units": 0, "wip": 30}
    ]
    res_ct = analyze_production_line(line_tie_ct, hours=1.0)
    assert res_ct["bottleneck_station"]["station"] == "StD"


def test_throughput_lost_due_to_rework():
    """
    Hand-computed test for throughput lost due to rework:
    - Bottleneck capacity without rework = 80.0 UPH
    - Bottleneck rework_rate = 0.15 (15%)
    - Throughput with rework = 80.0 / (1 + 0.15) = 80.0 / 1.15 = 69.5652 UPH
    - Throughput lost due to rework = 80.0 - 69.5652 = 10.4348 UPH
    """
    station_data = [
        {
            "station": "Bottleneck",
            "cycle_time": 0.6,
            "downtime_fraction": 0.20,
            "units_in": 750,
            "units_out": 700,
            "rework_units": 105,
            "wip": 25
        }
    ]
    res = analyze_production_line(station_data, hours=8.0)
    expected_lost = 80.0 - (80.0 / 1.15)
    assert res["throughput_lost_due_to_rework"] == pytest.approx(expected_lost, rel=1e-3)


# ==============================================================================
# 2. ECONOMICS (HAND-COMPUTED TOY DATA)
# ==============================================================================

def test_hand_computed_economics():
    """
    Hand-computed financial model test:
    - Good units = 600
    - Unit price = $100.00
    - Material cost = $20,000.00
    - Labor cost = $12,000.00
    - Overhead cost = $8,000.00
    - Scrap cost = $3,000.00
    - Rework cost = $2,500.00
    - Downtime cost = $1,500.00

    Hand Calculations:
    - Revenue = 600 * 100 = $60,000.00
    - Cost = 20000 + 12000 + 8000 + 3000 + 2500 + 1500 = $47,000.00
    - Profit = 60000 - 47000 = $13,000.00
    - Margin = 13000 / 60000 = 0.216666... (21.67%)
    - Cost per good unit = 47000 / 600 = $78.3333...
    - COPQ (Scrap + Rework + Downtime) = 3000 + 2500 + 1500 = $7,000.00
    """
    res = calculate_economics(
        good_units=600,
        unit_price=100.00,
        material_cost=20000.00,
        labor_cost=12000.00,
        overhead_cost=8000.00,
        scrap_cost=3000.00,
        rework_cost=2500.00,
        downtime_cost=1500.00
    )

    assert res["revenue"] == 60000.00
    assert res["cost"] == 47000.00
    assert res["profit"] == 13000.00
    assert res["margin"] == pytest.approx(13000.0 / 60000.0, rel=1e-3)
    assert res["cost_per_good_unit"] == pytest.approx(47000.0 / 600.0, rel=1e-3)
    assert res["cost_of_poor_quality"] == 7000.00


# ==============================================================================
# 3. WHAT-IF FUNCTION WITH RELATIVE INPUTS & DELTAS
# ==============================================================================

def test_what_if_simulation_deltas():
    """
    Verify What-If simulation with relative inputs:
    - reduce defect rate by 20%
    - increase bottleneck capacity by 10%
    - reduce downtime by 15%

    Verifies:
    - Label is 'Simulated / advisory'
    - Recomputes throughput, good units, revenue, cost, profit, margin
    - Returns DELTAS dictionary vs baseline
    - Simulated profit and margin exceed baseline
    """
    base_prod = {"line_throughput": 80.0}
    base_econ = {
        "good_units": 600,
        "unit_price": 100.00,
        "gross_revenue": 60000.00,
        "material_cost": 20000.00,
        "labor_cost": 12000.00,
        "overhead_cost": 8000.00,
        "scrap_cost": 3000.00,
        "rework_cost": 2500.00,
        "downtime_cost": 1500.00,
        "total_production_cost": 47000.00,
        "operating_profit": 13000.00,
        "gross_margin_pct": 21.67,
        "cost_of_poor_quality": 7000.00
    }

    sim = simulate_what_if(
        baseline_production=base_prod,
        baseline_economics=base_econ,
        defect_reduction_pct=20.0,
        capacity_boost_pct=10.0,
        downtime_reduction_pct=15.0
    )

    # Verify Label
    assert sim["label"] == "Simulated / advisory"

    # Verify Deltas exist
    deltas = sim["deltas"]
    assert "throughput_delta" in deltas
    assert "good_units_delta" in deltas
    assert "revenue_delta" in deltas
    assert "cost_delta" in deltas
    assert "profit_delta" in deltas
    assert "margin_delta" in deltas

    # Positive improvements
    assert deltas["throughput_delta"] > 0.0
    assert deltas["profit_delta"] > 0.0
    assert deltas["margin_delta"] > 0.0

    # Simulated values exceed baseline
    assert sim["simulated"]["throughput"] > sim["baseline"]["throughput"]
    assert sim["simulated"]["profit"] > sim["baseline"]["profit"]
    assert sim["simulated"]["margin"] > sim["baseline"]["margin"]
    assert sim["simulated"]["copq"] < sim["baseline"]["copq"]
