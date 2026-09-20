"""
src/production/flow.py
FactoryPulse Production Flow and Bottleneck Analysis Module.
Calculates station effective capacity, rework loop inflation, utilization,
bottleneck identification with tie-breaking, and throughput lost due to rework.

All mathematical formulas are documented below:
- effective_capacity = (60 / cycle_time) * (1 - downtime_fraction) [units/hour]
- rework_rate = rework_units / units_out
- effective_load = units_in * (1 + rework_rate)   # rework re-enters the line
- utilization = effective_load / (effective_capacity * hours)
- line throughput = capacity of the most constrained station
- Bottleneck = highest utilization; ties broken by higher WIP then longer cycle time
- Throughput lost due to rework = throughput without rework loop - throughput with it
"""

from typing import Dict, Any, List, Union, Optional
import pandas as pd
import numpy as np

def calculate_station_metrics(
    station_data: Dict[str, Any],
    hours: float = 8.0
) -> Dict[str, Any]:
    """
    Compute individual station capacity, load, rework multiplier, and utilization.

    Formulas:
    - downtime_fraction: Fraction of shift lost to downtime [0.0 to 1.0]
    - effective_capacity = (60 / cycle_time) * (1 - downtime_fraction) [units/hour]
    - rework_rate = rework_units / units_out
    - effective_load = units_in * (1 + rework_rate)
    - utilization = effective_load / (effective_capacity * hours)

    Parameters:
        station_data: Dictionary containing:
            - 'station': Name or identifier of the station
            - 'cycle_time': Process cycle time in minutes
            - 'downtime_fraction' or 'downtime_hours': Station downtime
            - 'units_in': Units entering the station
            - 'units_out': Completed units leaving the station
            - 'rework_units': Defective units sent for rework
            - 'wip': Work-in-process units currently buffered at station
        hours: Total operating hours in the period (default: 8.0)

    Returns:
        Dictionary of calculated station operational metrics.
    """
    station_name = str(station_data.get("station", station_data.get("station_name", "Station")))
    
    # 1. Cycle Time (in minutes per unit)
    # If cycle_time > 10 (e.g. given in seconds like 24s), convert to minutes: 24 / 60 = 0.4 min
    ct_raw = float(station_data.get("cycle_time", station_data.get("cycle_time_sec", 1.0)))
    cycle_time = ct_raw / 60.0 if ct_raw > 10.0 else max(1e-4, ct_raw)

    # 2. Downtime Fraction
    if "downtime_fraction" in station_data:
        dt_fraction = float(station_data["downtime_fraction"])
    elif "downtime_hours" in station_data:
        dt_fraction = float(station_data["downtime_hours"]) / max(1e-4, hours)
    elif "downtime_min" in station_data:
        dt_fraction = (float(station_data["downtime_min"]) / 60.0) / max(1e-4, hours)
    else:
        dt_fraction = 0.0
    dt_fraction = min(1.0, max(0.0, dt_fraction))

    # 3. Effective Capacity [units/hour]
    # FORMULA: effective_capacity = (60 / cycle_time) * (1 - downtime_fraction)
    effective_capacity = (60.0 / cycle_time) * (1.0 - dt_fraction)

    # 4. Units and Rework Rate
    units_in = float(station_data.get("units_in", 0.0))
    units_out = float(station_data.get("units_out", 0.0))
    rework_units = float(station_data.get("rework_units", station_data.get("rework_count", 0.0)))
    scrap_units = float(station_data.get("scrap_units", station_data.get("scrap_count", 0.0)))

    # FORMULA: rework_rate = rework_units / units_out
    rework_rate = (rework_units / units_out) if units_out > 0.0 else 0.0

    # 5. Effective Load (Rework re-enters the line)
    # FORMULA: effective_load = units_in * (1 + rework_rate)
    effective_load = units_in * (1.0 + rework_rate)

    # 6. Utilization
    # FORMULA: utilization = effective_load / (effective_capacity * hours)
    total_available_capacity = effective_capacity * hours
    utilization = (effective_load / total_available_capacity) if total_available_capacity > 0.0 else 0.0

    # 7. Work-In-Process (WIP)
    wip = float(station_data.get("wip", station_data.get("wip_count", 0.0)))

    return {
        "station": station_name,
        "cycle_time": round(cycle_time, 4),
        "downtime_fraction": round(dt_fraction, 4),
        "effective_capacity": round(effective_capacity, 2),
        "units_in": int(units_in),
        "units_out": int(units_out),
        "rework_units": int(rework_units),
        "scrap_units": int(scrap_units),
        "rework_rate": round(rework_rate, 4),
        "effective_load": round(effective_load, 2),
        "utilization": round(utilization, 4),
        "wip": int(wip)
    }


def analyze_production_line(
    stations: Union[List[Dict[str, Any]], pd.DataFrame],
    hours: float = 8.0
) -> Dict[str, Any]:
    """
    Evaluate the full production line, identify primary bottleneck,
    and calculate rework-induced throughput loss.

    Bottleneck Criteria:
    - Highest utilization.
    - Ties broken by higher WIP, then longer cycle_time.

    Throughput Lost Due to Rework:
    - throughput_without_rework = effective_capacity of bottleneck
    - throughput_with_rework = throughput_without_rework / (1 + rework_rate)
    - throughput_lost_due_to_rework = throughput_without_rework - throughput_with_rework

    Returns:
        Dictionary with all station metrics, bottleneck evidence string,
        line throughput, and throughput lost due to rework.
    """
    if isinstance(stations, pd.DataFrame):
        records = stations.to_dict(orient="records")
    else:
        records = [dict(s) for s in stations]

    if not records:
        return {
            "stations": [],
            "bottleneck_station": None,
            "bottleneck_evidence": "No station data available",
            "line_throughput": 0.0,
            "throughput_without_rework": 0.0,
            "throughput_with_rework": 0.0,
            "throughput_lost_due_to_rework": 0.0,
            "total_wip": 0,
            "total_scrap": 0,
            "total_rework": 0
        }

    # Calculate metrics for each station
    computed_stations = [calculate_station_metrics(st, hours=hours) for st in records]

    # Find Bottleneck:
    # Highest utilization; ties broken by higher WIP then longer cycle time
    sorted_stations = sorted(
        computed_stations,
        key=lambda s: (round(s["utilization"], 4), s["wip"], s["cycle_time"]),
        reverse=True
    )
    bottleneck = sorted_stations[0]

    # Evidence String, e.g.:
    # "Station 3: utilization 94%, WIP 42, rework adds 11% extra load"
    rework_extra_load_pct = bottleneck["rework_rate"] * 100.0
    evidence_str = (
        f"{bottleneck['station']}: utilization {bottleneck['utilization'] * 100.0:.0f}%, "
        f"WIP {bottleneck['wip']}, rework adds {rework_extra_load_pct:.0f}% extra load"
    )

    # Line throughput = capacity of the most constrained station
    # FORMULA: line throughput = capacity of the most constrained station
    line_throughput = bottleneck["effective_capacity"]

    # Throughput lost due to rework:
    # FORMULA: throughput without rework loop - throughput with it
    throughput_without_rework = bottleneck["effective_capacity"]
    throughput_with_rework = throughput_without_rework / (1.0 + bottleneck["rework_rate"])
    throughput_lost_due_to_rework = throughput_without_rework - throughput_with_rework

    # Total WIP across line
    total_wip = sum(s["wip"] for s in computed_stations)
    total_scrap = sum(s["scrap_units"] for s in computed_stations)
    total_rework = sum(s["rework_units"] for s in computed_stations)

    return {
        "stations": computed_stations,
        "bottleneck_station": bottleneck,
        "bottleneck_evidence": evidence_str,
        "line_throughput": round(line_throughput, 2),
        "throughput_without_rework": round(throughput_without_rework, 2),
        "throughput_with_rework": round(throughput_with_rework, 2),
        "throughput_lost_due_to_rework": round(throughput_lost_due_to_rework, 2),
        "total_wip": int(total_wip),
        "total_scrap": int(total_scrap),
        "total_rework": int(total_rework)
    }


class ProductionAnalyzer:
    """
    High-level analyzer wrapping production flow calculations for DataFrame
    and multi-batch datasets.
    """

    def __init__(self, df_production: pd.DataFrame, total_line_defects: int = 0, hours: float = 8.0):
        self.df_prod = df_production.copy()
        self.total_defects = total_line_defects
        self.hours = hours

    def analyze_line(self) -> Dict[str, Any]:
        """Run line analysis across stations in df_production."""
        df = self.df_prod.copy()
        if len(df) == 0:
            return {
                "stations": [],
                "bottleneck_station": {},
                "constrained_throughput_uph": 0.0,
                "total_actual_wip": 0,
                "total_scrap_units": 0,
                "total_rework_units": 0,
                "throughput_lost_due_to_rework": 0.0
            }

        # Aggregate across batches per station if multi-batch table
        station_rows = []
        for st_name, st_group in df.groupby("station" if "station" in df.columns else "station_id"):
            ct = float(st_group["cycle_time"].mean()) if "cycle_time" in st_group.columns else float(st_group.get("cycle_time_sec", pd.Series([24.0])).mean())
            dt_h = float(st_group["downtime_hours"].sum()) if "downtime_hours" in st_group.columns else float(st_group.get("downtime_min", pd.Series([0.0])).sum()) / 60.0
            u_in = float(st_group["units_in"].sum()) if "units_in" in st_group.columns else 100.0
            u_out = float(st_group["units_out"].sum()) if "units_out" in st_group.columns else 95.0
            rew = float(st_group["rework_units"].sum()) if "rework_units" in st_group.columns else float(st_group.get("rework_count", pd.Series([0.0])).sum())
            scr = float(st_group["scrap_units"].sum()) if "scrap_units" in st_group.columns else float(st_group.get("scrap_count", pd.Series([0.0])).sum())
            wip = float(st_group["wip"].mean()) if "wip" in st_group.columns else float(st_group.get("wip_count", pd.Series([20.0])).mean())

            station_rows.append({
                "station": str(st_name),
                "cycle_time": ct,
                "downtime_hours": dt_h,
                "units_in": u_in,
                "units_out": u_out,
                "rework_units": rew,
                "scrap_units": scr,
                "wip": wip
            })

        res = analyze_production_line(station_rows, hours=self.hours)

        # Build backward-compatible fields for dashboard and existing components
        bn = res["bottleneck_station"] if res["bottleneck_station"] else {}
        stations_rep = []
        for s in res["stations"]:
            stations_rep.append({
                "station_id": s["station"],
                "station_name": s["station"],
                "nominal_capacity_uph": round(60.0 / max(1e-4, s["cycle_time"]), 1),
                "effective_capacity_uph": s["effective_capacity"],
                "cycle_time_sec": round(s["cycle_time"] * 60.0, 1),
                "wip_count": s["wip"],
                "downtime_min": round(s["downtime_fraction"] * self.hours * 60.0, 1),
                "rework_count": s["rework_units"],
                "nominal_utilization_pct": round(min(100.0, (s["units_in"] / max(1.0, s["effective_capacity"] * self.hours)) * 100.0), 1),
                "rework_utilization_add": round(s["rework_rate"] * 100.0, 1),
                "total_effective_utilization_pct": round(s["utilization"] * 100.0, 1),
                "headroom_uph": max(0.0, round(s["effective_capacity"] - (s["effective_load"] / self.hours), 1))
            })

        bn_compat = {
            "station_id": bn.get("station", "N/A"),
            "station_name": bn.get("station", "N/A"),
            "effective_capacity_uph": bn.get("effective_capacity", 0.0),
            "total_effective_utilization_pct": round(bn.get("utilization", 0.0) * 100.0, 1),
            "rework_utilization_add": round(bn.get("rework_rate", 0.0) * 100.0, 1),
            "headroom_uph": max(0.0, round(bn.get("effective_capacity", 0.0) - (bn.get("effective_load", 0.0) / self.hours), 1)),
            "downtime_min": round(bn.get("downtime_fraction", 0.0) * self.hours * 60.0, 1)
        }

        total_line_ct_sec = sum(s["cycle_time_sec"] for s in stations_rep)
        littles_law_wip = res["line_throughput"] * (total_line_ct_sec / 3600.0)

        return {
            "stations": stations_rep,
            "bottleneck_station": bn_compat,
            "bottleneck_evidence": res["bottleneck_evidence"],
            "constrained_throughput_uph": res["line_throughput"],
            "line_throughput": res["line_throughput"],
            "throughput_lost_due_to_rework": res["throughput_lost_due_to_rework"],
            "total_line_cycle_time_sec": round(total_line_ct_sec, 1),
            "total_actual_wip": res["total_wip"],
            "littles_law_expected_wip": round(littles_law_wip, 1),
            "wip_sanity_delta": round(res["total_wip"] - littles_law_wip, 1),
            "total_scrap_units": res["total_scrap"],
            "total_rework_units": res["total_rework"]
        }
