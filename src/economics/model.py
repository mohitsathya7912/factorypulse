"""
src/economics/model.py
FactoryPulse Economic Calculation and What-If Simulation Module.
Calculates period revenue, cost breakdown, profit, gross margin,
cost per good unit, and cost of poor quality (COPQ).

Provides the What-If simulation function with relative inputs,
computing recomputed metrics and explicit DELTAS vs baseline.

All mathematical formulas are documented below:
- Revenue = Good units x Unit price
- Cost = Material + Labor + Overhead + Scrap cost + Rework cost + Downtime cost
- Profit = Revenue - Cost
- Margin = Profit / Revenue
- Cost per good unit = Cost / Good units
- Cost of poor quality = Scrap cost + Rework cost + Downtime cost
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np

def calculate_economics(
    good_units: float,
    unit_price: float,
    material_cost: float,
    labor_cost: float,
    overhead_cost: float,
    scrap_cost: float,
    rework_cost: float,
    downtime_cost: float
) -> Dict[str, Any]:
    """
    Compute period totals for revenue, cost breakdown, profit, margin,
    cost per good unit, and cost of poor quality.

    Formulas:
    - Revenue = Good units x Unit price
    - Cost = Material + Labor + Overhead + Scrap cost + Rework cost + Downtime cost
    - Profit = Revenue - Cost
    - Margin = Profit / Revenue
    - Cost per good unit = Cost / Good units
    - Cost of poor quality = Scrap cost + Rework cost + Downtime cost

    Returns:
        Dictionary of exact financial totals.
    """
    # 1. Revenue
    # FORMULA: Revenue = Good units x Unit price
    revenue = float(good_units) * float(unit_price)

    # 2. Total Cost
    # FORMULA: Cost = Material + Labor + Overhead + Scrap cost + Rework cost + Downtime cost
    cost = (
        float(material_cost) +
        float(labor_cost) +
        float(overhead_cost) +
        float(scrap_cost) +
        float(rework_cost) +
        float(downtime_cost)
    )

    # 3. Profit
    # FORMULA: Profit = Revenue - Cost
    profit = revenue - cost

    # 4. Gross Margin
    # FORMULA: Margin = Profit / Revenue
    margin = (profit / revenue) if revenue > 0.0 else 0.0

    # 5. Cost per Good Unit
    # FORMULA: Cost per good unit = Cost / Good units
    cost_per_good_unit = (cost / good_units) if good_units > 0.0 else 0.0

    # 6. Cost of Poor Quality (COPQ)
    # FORMULA: Cost of poor quality = Scrap cost + Rework cost + Downtime cost
    cost_of_poor_quality = float(scrap_cost) + float(rework_cost) + float(downtime_cost)

    return {
        "good_units": int(good_units),
        "unit_price": round(float(unit_price), 2),
        "revenue": round(revenue, 2),
        "material_cost": round(float(material_cost), 2),
        "labor_cost": round(float(labor_cost), 2),
        "overhead_cost": round(float(overhead_cost), 2),
        "scrap_cost": round(float(scrap_cost), 2),
        "rework_cost": round(float(rework_cost), 2),
        "downtime_cost": round(float(downtime_cost), 2),
        "cost": round(cost, 2),
        "profit": round(profit, 2),
        "margin": round(margin, 4),
        "cost_per_good_unit": round(cost_per_good_unit, 2),
        "cost_of_poor_quality": round(cost_of_poor_quality, 2)
    }


def simulate_what_if(
    baseline_production: Dict[str, Any],
    baseline_economics: Dict[str, Any],
    defect_reduction_pct: float = 0.0,
    capacity_boost_pct: float = 0.0,
    downtime_reduction_pct: float = 0.0,
    hours: float = 8.0
) -> Dict[str, Any]:
    """
    Evaluate What-If simulation with relative inputs:
    - reduce defect rate by X% (defect_reduction_pct)
    - increase bottleneck capacity by Y% (capacity_boost_pct)
    - reduce downtime by Z% (downtime_reduction_pct)

    Recomputes throughput, good units, revenue, cost, profit, margin,
    and returns DELTAS vs baseline.

    Label: 'Simulated / advisory'

    Returns:
        Dictionary containing baseline, simulated, and deltas dictionaries.
    """
    # Baseline values
    base_tp = float(baseline_production.get("line_throughput", baseline_production.get("constrained_throughput_uph", 0.0)))
    base_good = int(baseline_economics.get("good_units", 0))
    base_rev = float(baseline_economics.get("revenue", baseline_economics.get("gross_revenue", 0.0)))
    base_cost = float(baseline_economics.get("cost", baseline_economics.get("total_production_cost", 0.0)))
    base_profit = float(baseline_economics.get("profit", baseline_economics.get("operating_profit", 0.0)))
    base_margin = float(baseline_economics.get("margin", baseline_economics.get("gross_margin_pct", 0.0) / 100.0 if "gross_margin_pct" in baseline_economics else 0.0))
    base_unit_price = float(baseline_economics.get("unit_price", 85.0))
    base_copq = float(baseline_economics.get("cost_of_poor_quality", baseline_economics.get("total_quality_loss", 0.0)))

    # Relative adjustment multipliers
    defect_mult = max(0.0, 1.0 - (defect_reduction_pct / 100.0))
    cap_mult = 1.0 + (capacity_boost_pct / 100.0)
    dt_mult = max(0.0, 1.0 - (downtime_reduction_pct / 100.0))

    # Recompute Throughput:
    # Bottleneck capacity increases by cap_mult; downtime reduction further unlocks availability
    # sim_throughput = base_tp * cap_mult * (1 + dt_recovery)
    dt_headroom_boost = 1.0 + ((downtime_reduction_pct / 100.0) * 0.10)
    sim_tp = round(base_tp * cap_mult * dt_headroom_boost, 2)

    # Recompute Units:
    # Good units scale with throughput and saved scrap
    tp_ratio = (sim_tp / base_tp) if base_tp > 0.0 else 1.0
    scrap_saved = float(baseline_economics.get("scrap_cost", 0.0)) * (1.0 - defect_mult) / max(1.0, float(baseline_economics.get("scrap_cost", 1.0)) / max(1, base_good)) if base_good > 0 else 0.0
    sim_good = int(round(base_good * tp_ratio + scrap_saved * 0.5))

    # Recompute Financials
    sim_rev = round(sim_good * base_unit_price, 2)

    # Material scales with production throughput
    base_mat = float(baseline_economics.get("material_cost", 0.0))
    sim_mat = round(base_mat * tp_ratio, 2)

    # Labor and overhead remain largely fixed for the period
    sim_labor = float(baseline_economics.get("labor_cost", 0.0))
    sim_overhead = float(baseline_economics.get("overhead_cost", 0.0))

    # Scrap, Rework, and Downtime costs reduced by relative inputs
    base_scrap_cost = float(baseline_economics.get("scrap_cost", 0.0))
    base_rework_cost = float(baseline_economics.get("rework_cost", 0.0))
    base_downtime_cost = float(baseline_economics.get("downtime_cost", 0.0))

    sim_scrap_cost = round(base_scrap_cost * defect_mult, 2)
    sim_rework_cost = round(base_rework_cost * defect_mult, 2)
    sim_downtime_cost = round(base_downtime_cost * dt_mult, 2)

    sim_cost = round(
        sim_mat + sim_labor + sim_overhead + sim_scrap_cost + sim_rework_cost + sim_downtime_cost, 2
    )
    sim_profit = round(sim_rev - sim_cost, 2)
    sim_margin = round((sim_profit / sim_rev), 4) if sim_rev > 0.0 else 0.0
    sim_cost_per_good = round((sim_cost / sim_good), 2) if sim_good > 0 else 0.0
    sim_copq = round(sim_scrap_cost + sim_rework_cost + sim_downtime_cost, 2)

    # DELTAS vs Baseline
    deltas = {
        "throughput_delta": round(sim_tp - base_tp, 2),
        "good_units_delta": int(sim_good - base_good),
        "revenue_delta": round(sim_rev - base_rev, 2),
        "cost_delta": round(sim_cost - base_cost, 2),
        "profit_delta": round(sim_profit - base_profit, 2),
        "margin_delta": round(sim_margin - base_margin, 4),
        "copq_delta": round(sim_copq - base_copq, 2)
    }

    baseline_dict = {
        "throughput": base_tp,
        "good_units": base_good,
        "revenue": base_rev,
        "cost": base_cost,
        "profit": base_profit,
        "margin": base_margin,
        "copq": base_copq
    }

    simulated_dict = {
        "throughput": sim_tp,
        "good_units": sim_good,
        "revenue": sim_rev,
        "cost": sim_cost,
        "profit": sim_profit,
        "margin": sim_margin,
        "cost_per_good_unit": sim_cost_per_good,
        "copq": sim_copq
    }

    return {
        "label": "Simulated / advisory",
        "inputs": {
            "defect_reduction_pct": defect_reduction_pct,
            "capacity_boost_pct": capacity_boost_pct,
            "downtime_reduction_pct": downtime_reduction_pct
        },
        "baseline": baseline_dict,
        "simulated": simulated_dict,
        "deltas": deltas
    }


class EconomicModel:
    """
    Object-oriented economic wrapper providing P&L calculations and
    What-If sensitivity models for DataFrame and dataset inputs.
    """

    def __init__(self, df_economics: pd.DataFrame):
        self.df_econ = df_economics.copy()
        self.params = {
            "unit_price": 85.00,
            "material_cost": 32.00,
            "labor_cost": 45.00,
            "overhead_cost": 30.00,
            "scrap_cost": 42.00,
            "rework_cost": 18.50,
            "downtime_cost_per_hour": 150.00,
            # Aliases
            "labor_rate_hourly": 45.00,
            "overhead_rate_hourly": 30.00,
            "scrap_penalty_per_unit": 42.00,
            "rework_cost_per_unit": 18.50,
            "downtime_penalty_hourly": 150.00
        }
        if len(df_economics) > 0:
            first_row = df_economics.iloc[0].to_dict()
            for k, v in first_row.items():
                if pd.notnull(v):
                    self.params[k] = float(v)

    def calculate_pnl(
        self,
        total_units_produced: int,
        scrap_units: int,
        rework_units: int,
        downtime_minutes: float,
        shift_hours: float = 8.0,
        active_workers: int = 10
    ) -> Dict[str, Any]:
        """Compute full profit and loss breakdown matching canonical formulations."""
        good_units = max(0, total_units_produced - scrap_units)

        # Standard costs
        mat_rate = self.params.get("material_cost", 32.00)
        labor_rate = self.params.get("labor_cost", self.params.get("labor_rate_hourly", 45.00))
        overhead_rate = self.params.get("overhead_cost", self.params.get("overhead_rate_hourly", 30.00))
        scrap_rate = self.params.get("scrap_cost", self.params.get("scrap_penalty_per_unit", 42.00))
        rework_rate = self.params.get("rework_cost", self.params.get("rework_cost_per_unit", 18.50))
        dt_rate = self.params.get("downtime_cost_per_hour", self.params.get("downtime_penalty_hourly", 150.00))

        tot_mat = total_units_produced * mat_rate
        tot_labor = shift_hours * active_workers * labor_rate
        tot_overhead = shift_hours * overhead_rate
        tot_scrap = scrap_units * scrap_rate
        tot_rework = rework_units * rework_rate
        tot_downtime = (downtime_minutes / 60.0) * dt_rate

        econ_dict = calculate_economics(
            good_units=good_units,
            unit_price=self.params.get("unit_price", 85.00),
            material_cost=tot_mat,
            labor_cost=tot_labor,
            overhead_cost=tot_overhead,
            scrap_cost=tot_scrap,
            rework_cost=tot_rework,
            downtime_cost=tot_downtime
        )

        # Map to dashboard expected keys
        econ_dict["gross_revenue"] = econ_dict["revenue"]
        econ_dict["total_production_cost"] = econ_dict["cost"]
        econ_dict["operating_profit"] = econ_dict["profit"]
        econ_dict["gross_margin_pct"] = round(econ_dict["margin"] * 100.0, 2)
        econ_dict["total_quality_loss"] = econ_dict["cost_of_poor_quality"]
        econ_dict["unit_standard_cost"] = econ_dict["cost_per_good_unit"]
        econ_dict["total_units_produced"] = int(total_units_produced)
        econ_dict["scrap_units"] = int(scrap_units)
        econ_dict["rework_units"] = int(rework_units)
        econ_dict["downtime_minutes"] = round(downtime_minutes, 1)

        return econ_dict

    def simulate_margin_recovery(
        self,
        base_pnl: Dict[str, Any],
        defect_reduction_pct: float = 0.0,
        throughput_increase_pct: float = 0.0
    ) -> Dict[str, Any]:
        """Wrapper for What-If scenario simulations with relative inputs."""
        sim_res = simulate_what_if(
            baseline_production={"line_throughput": base_pnl.get("good_units", 100) / 8.0},
            baseline_economics=base_pnl,
            defect_reduction_pct=defect_reduction_pct,
            capacity_boost_pct=throughput_increase_pct,
            downtime_reduction_pct=defect_reduction_pct * 0.5
        )
        return {
            "label": sim_res["label"],
            "baseline_profit": sim_res["baseline"]["profit"],
            "simulated_profit": sim_res["simulated"]["profit"],
            "profit_gain": sim_res["deltas"]["profit_delta"],
            "baseline_margin": round(sim_res["baseline"]["margin"] * 100.0, 2),
            "simulated_margin": round(sim_res["simulated"]["margin"] * 100.0, 2),
            "margin_gain_points": round(sim_res["deltas"]["margin_delta"] * 100.0, 2),
            "cost_savings": round(-sim_res["deltas"]["cost_delta"], 2)
        }


def bootstrap_what_if_margin(
    batch_df: pd.DataFrame,
    baseline_economics: Dict[str, Any],
    defect_reduction_pct: float = 0.0,
    capacity_boost_pct: float = 0.0,
    downtime_reduction_pct: float = 0.0,
    n_boot: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Bootstrap over manufacturing batches to give the what-if margin as a median and
    10-90% empirical range instead of a single point estimate.

    Label: 'Simulated / advisory'

    Args:
        batch_df: Consolidated batch DataFrame (from build_batch_table)
        baseline_economics: Baseline financial totals dictionary
        defect_reduction_pct: Relative defect reduction %
        capacity_boost_pct: Relative capacity boost %
        downtime_reduction_pct: Relative downtime reduction %
        n_boot: Number of bootstrap resamples (default 1000)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing median_margin_pct, p10_margin_pct, p90_margin_pct,
        range_str, and label 'Simulated / advisory'.
    """
    if batch_df is None or len(batch_df) == 0:
        base_m = float(baseline_economics.get("margin", 0.0)) * 100.0
        return {
            "median_margin_pct": round(base_m, 2),
            "p10_margin_pct": round(base_m, 2),
            "p90_margin_pct": round(base_m, 2),
            "range_str": f"[{base_m:.1f}%, {base_m:.1f}%]",
            "label": "Simulated / advisory"
        }

    n_batches = len(batch_df)
    rng = np.random.RandomState(seed)

    has_scrap = "scrap_units" in batch_df.columns and batch_df["scrap_units"].sum() > 0
    has_rework = "rework_units" in batch_df.columns and batch_df["rework_units"].sum() > 0
    has_dt = "downtime_hours" in batch_df.columns and batch_df["downtime_hours"].sum() > 0

    base_unit_price = float(baseline_economics.get("unit_price", 85.0))
    base_mat_cost = float(baseline_economics.get("material_cost", 0.0))
    base_labor = float(baseline_economics.get("labor_cost", 0.0))
    base_overhead = float(baseline_economics.get("overhead_cost", 0.0))
    base_scrap_cost = float(baseline_economics.get("scrap_cost", 0.0))
    base_rework_cost = float(baseline_economics.get("rework_cost", 0.0))
    base_dt_cost = float(baseline_economics.get("downtime_cost", 0.0))

    total_batch_inspected = float(batch_df["total_inspected"].sum()) if "total_inspected" in batch_df.columns else float(n_batches * 50)
    mat_per_unit = base_mat_cost / max(1.0, total_batch_inspected)

    total_batch_scrap = float(batch_df["scrap_units"].sum()) if has_scrap else max(1.0, float(batch_df["defect_count"].sum()) * 0.5 if "defect_count" in batch_df.columns else 1.0)
    total_batch_rework = float(batch_df["rework_units"].sum()) if has_rework else max(1.0, float(batch_df["defect_count"].sum()) * 0.5 if "defect_count" in batch_df.columns else 1.0)
    total_batch_dt = float(batch_df["downtime_hours"].sum()) if has_dt else max(1.0, float(n_batches))

    scrap_rate = base_scrap_cost / max(1.0, total_batch_scrap)
    rework_rate = base_rework_cost / max(1.0, total_batch_rework)
    dt_rate = base_dt_cost / max(1.0, total_batch_dt)

    boot_margins = []

    for _ in range(n_boot):
        idx = rng.choice(n_batches, size=n_batches, replace=True)
        samp = batch_df.iloc[idx]

        samp_total = float(samp["total_inspected"].sum()) if "total_inspected" in samp.columns else float(len(samp) * 50)
        if "conforming_count" in samp.columns:
            samp_good = float(samp["conforming_count"].sum())
        elif "defect_count" in samp.columns:
            samp_good = float(samp_total - samp["defect_count"].sum())
        else:
            samp_good = samp_total

        samp_scrap_u = float(samp["scrap_units"].sum()) if has_scrap else (float(samp["defect_count"].sum()) * 0.5 if "defect_count" in samp.columns else 0.0)
        samp_rework_u = float(samp["rework_units"].sum()) if has_rework else (float(samp["defect_count"].sum()) * 0.5 if "defect_count" in samp.columns else 0.0)
        samp_dt_h = float(samp["downtime_hours"].sum()) if has_dt else (float(len(samp)))

        samp_econ = calculate_economics(
            good_units=max(0, samp_good),
            unit_price=base_unit_price,
            material_cost=samp_total * mat_per_unit,
            labor_cost=base_labor,
            overhead_cost=base_overhead,
            scrap_cost=samp_scrap_u * scrap_rate,
            rework_cost=samp_rework_u * rework_rate,
            downtime_cost=samp_dt_h * dt_rate
        )

        res = simulate_what_if(
            baseline_production={"line_throughput": samp_good / 8.0},
            baseline_economics=samp_econ,
            defect_reduction_pct=defect_reduction_pct,
            capacity_boost_pct=capacity_boost_pct,
            downtime_reduction_pct=downtime_reduction_pct
        )
        boot_margins.append(res["simulated"]["margin"] * 100.0)

    med = float(np.median(boot_margins))
    p10 = float(np.percentile(boot_margins, 10))
    p90 = float(np.percentile(boot_margins, 90))

    return {
        "median_margin_pct": round(med, 2),
        "p10_margin_pct": round(p10, 2),
        "p90_margin_pct": round(p90, 2),
        "range_str": f"[{p10:.1f}%, {p90:.1f}%]",
        "label": "Simulated / advisory"
    }

