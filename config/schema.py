"""
config/schema.py
FactoryPulse Data Schemas, Canonical Column Definitions, and Central Column-Mapping Dictionary.
Provides centralized column remapping so real organizer column names can be remapped in ONE place.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

def is_defective(val_or_series: Any) -> Any:
    """
    Return boolean indicator (or Series/array of booleans) indicating defective status.
    Accepts:
    - String labels: 'defective', 'defect', 'fail', 'failed', 'ng', 'bad', '1', 'true' -> True
                     'good', 'normal', 'pass', 'ok', '0', 'false', 'none', '', NaN -> False
    - Numeric values: 1, 1.0 -> True; 0, 0.0 -> False
    - Boolean values: True -> True; False -> False
    - pandas Series: returns boolean pd.Series
    - numpy array: returns boolean np.ndarray
    """
    if isinstance(val_or_series, pd.Series):
        s = val_or_series.astype(str).str.strip().str.lower()
        return s.isin(["1", "1.0", "true", "t", "defective", "defect", "fail", "failed", "ng", "bad"])
    elif isinstance(val_or_series, np.ndarray):
        s = pd.Series(val_or_series).astype(str).str.strip().str.lower()
        return s.isin(["1", "1.0", "true", "t", "defective", "defect", "fail", "failed", "ng", "bad"]).values
    elif pd.isnull(val_or_series):
        return False
    elif isinstance(val_or_series, (bool, np.bool_)):
        return bool(val_or_series)
    elif isinstance(val_or_series, (int, float, np.number)):
        return bool(val_or_series == 1 or val_or_series == 1.0)
    else:
        s = str(val_or_series).strip().lower()
        return s in ["1", "1.0", "true", "t", "defective", "defect", "fail", "failed", "ng", "bad"]

# ==============================================================================
# 1. CANONICAL EXPECTED COLUMNS FOR THE FOUR DATASETS
# ==============================================================================

# Inspection: unit_id, batch_id, variant, timestamp, label (good/defective), defect_category, plus optional image_path
INSPECTION_EXPECTED = [
    "unit_id",
    "batch_id",
    "variant",
    "timestamp",
    "label",
    "defect_category"
]
INSPECTION_OPTIONAL = ["image_path", "bbox"]

# Process: batch_id, timestamp, several numeric process parameters (e.g. temperature, speed, pressure, vibration)
PROCESS_EXPECTED = [
    "batch_id",
    "timestamp",
    "temperature",
    "speed",
    "pressure",
    "vibration"
]

# Production: station, batch_id, cycle_time, capacity, wip, downtime_hours, units_in, units_out, rework_units, scrap_units
PRODUCTION_EXPECTED = [
    "station",
    "batch_id",
    "cycle_time",
    "capacity",
    "wip",
    "downtime_hours",
    "units_in",
    "units_out",
    "rework_units",
    "scrap_units"
]

# Economic: unit_price, material_cost, labor_cost, overhead_cost, scrap_cost, rework_cost, downtime_cost_per_hour
ECONOMIC_EXPECTED = [
    "unit_price",
    "material_cost",
    "labor_cost",
    "overhead_cost",
    "scrap_cost",
    "rework_cost",
    "downtime_cost_per_hour"
]

# ==============================================================================
# 2. CENTRAL COLUMN-MAPPING DICTIONARY (Remap organizer names in ONE place)
# ==============================================================================

COLUMN_MAPPING: Dict[str, Dict[str, str]] = {
    "inspection": {
        # Organizer column variants -> Canonical column
        "id": "unit_id",
        "part_id": "unit_id",
        "item_id": "unit_id",
        "serial_number": "unit_id",
        "batch": "batch_id",
        "lot_id": "batch_id",
        "model": "variant",
        "variant_id": "variant",
        "product_type": "variant",
        "product": "variant",
        "time": "timestamp",
        "date": "timestamp",
        "datetime": "timestamp",
        "inspection_time": "timestamp",
        "is_defective": "label",
        "defective": "label",
        "has_defect": "label",
        "defect": "label",
        "status": "label",
        "pass_fail": "label",
        "result": "label",
        "defect_family": "defect_category",
        "defect_type": "defect_category",
        "defect_class": "defect_category",
        "category": "defect_category",
        "failure_mode": "defect_category",
        "image": "image_path",
        "file_path": "image_path",
        "filename": "image_path",
        "img": "image_path",
        "bounding_box": "bbox",
        "box": "bbox",
    },
    "process": {
        "batch": "batch_id",
        "lot_id": "batch_id",
        "time": "timestamp",
        "date": "timestamp",
        "datetime": "timestamp",
        "temp": "temperature",
        "temp_c": "temperature",
        "heater_temp": "temperature",
        "temp_zone_1": "temperature",
        "temp_zone_2": "temperature",
        "feed_rate": "speed",
        "feed_rate_m_min": "speed",
        "line_speed": "speed",
        "rpm": "speed",
        "press": "pressure",
        "pressure_bar": "pressure",
        "hydraulic_pressure": "pressure",
        "vib": "vibration",
        "vibration_mm_s": "vibration",
        "vib_level": "vibration",
    },
    "production": {
        "station_id": "station",
        "station_name": "station",
        "machine": "station",
        "workstation": "station",
        "line_station": "station",
        "batch": "batch_id",
        "lot_id": "batch_id",
        "ct": "cycle_time",
        "cycle_time_sec": "cycle_time",
        "process_time": "cycle_time",
        "nominal_capacity": "capacity",
        "nominal_capacity_uph": "capacity",
        "uph": "capacity",
        "max_capacity": "capacity",
        "wip_count": "wip",
        "in_process": "wip",
        "queue_count": "wip",
        "buffer": "wip",
        "downtime": "downtime_hours",
        "down_time": "downtime_hours",
        "downtime_min": "downtime_hours", # converted if needed
        "input_units": "units_in",
        "arrivals": "units_in",
        "output_units": "units_out",
        "completed": "units_out",
        "rework": "rework_units",
        "reworked": "rework_units",
        "scrap": "scrap_units",
        "scrapped": "scrap_units",
    },
    "economic": {
        "price": "unit_price",
        "selling_price": "unit_price",
        "revenue_per_unit": "unit_price",
        "material": "material_cost",
        "raw_material": "material_cost",
        "labor": "labor_cost",
        "labor_rate": "labor_cost",
        "labor_rate_hourly": "labor_cost",
        "overhead": "overhead_cost",
        "overhead_rate": "overhead_cost",
        "overhead_rate_hourly": "overhead_cost",
        "scrap_penalty": "scrap_cost",
        "scrap_penalty_per_unit": "scrap_cost",
        "rework": "rework_cost",
        "rework_cost_per_unit": "rework_cost",
        "downtime_penalty": "downtime_cost_per_hour",
        "downtime_cost": "downtime_cost_per_hour",
        "downtime_penalty_hourly": "downtime_cost_per_hour",
    }
}

# ==============================================================================
# 3. SCHEMA MAPPING & VALIDATION UTILITIES
# ==============================================================================

def remap_columns(df: pd.DataFrame, dataset_type: str = "generic") -> pd.DataFrame:
    """
    Standardize DataFrame column names by trimming, converting to lowercase,
    and mapping aliases according to COLUMN_MAPPING.
    """
    cleaned_cols = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    renamed_df = df.rename(columns=cleaned_cols)

    mapping = COLUMN_MAPPING.get(dataset_type.lower(), {})
    # Also combine with all other domain mappings for maximum resilience
    combined_mapping = {}
    for sub_map in COLUMN_MAPPING.values():
        combined_mapping.update(sub_map)
    combined_mapping.update(mapping)

    final_cols = {col: combined_mapping.get(col, col) for col in renamed_df.columns}
    return renamed_df.rename(columns=final_cols)


def validate_columns(
    df: pd.DataFrame,
    expected_cols: List[str],
    dataset_name: str
) -> Tuple[bool, List[str], str]:
    """
    Validate that required columns exist in the DataFrame.
    Returns:
        (is_valid: bool, missing_columns: List[str], message: str)
    """
    actual_cols = set(df.columns)
    missing = [c for c in expected_cols if c not in actual_cols]
    if missing:
        msg = (
            f"Validation Warning for '{dataset_name}': Missing {len(missing)} required column(s): {missing}. "
            f"Present columns: {list(df.columns)}."
        )
        return False, missing, msg
    msg = f"Validation Success for '{dataset_name}': All {len(expected_cols)} required columns present."
    return True, [], msg

from dataclasses import dataclass

@dataclass
class DatasetOrigin:
    """Metadata regarding data provenance (Organizer vs Synthetic Demo vs Upload)."""
    is_synthetic: bool
    source_name: str
    record_counts: Dict[str, int]
    warning_banner: Optional[str] = None

    def get_banner_text(self) -> Optional[str]:
        if self.is_synthetic:
            return "⚠️ DEMO / SYNTHETIC DATA ACTIVE — Organizer dataset not yet detected in data/organizer/."
        return None

# Alias for backwards compatibility
standardize_columns = remap_columns
