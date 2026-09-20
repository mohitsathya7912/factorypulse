"""
src/data/loader.py
FactoryPulse Universal Data Loader and Preprocessing Pipeline.
Strictly relies on Pandas and NumPy only.

Implements three operational data modes:
1. 'Organizer': Loads from data/organizer/ (auto-detects files & remaps columns)
2. 'Demo': Loads from data/sample/ (fallback synthetic baseline with is_synthetic=True)
3. 'Upload': Accepts in-memory/uploaded CSV streams from the Streamlit UI

Includes automated schema validation, missing value cleaning strategies,
batch_id table joining, and multi-domain summary statistics.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import pandas as pd

from config.schema import (
    INSPECTION_EXPECTED,
    PROCESS_EXPECTED,
    PRODUCTION_EXPECTED,
    ECONOMIC_EXPECTED,
    remap_columns,
    validate_columns,
    is_defective,
)
from src.data.generator import generate_sample_datasets

class DataLoader:
    """
    Universal manufacturing dataset loader, cleaner, and statistical engine.
    Supports Organizer, Demo, and Upload data modes with transparent provenance tracking.
    """

    def __init__(
        self,
        organizer_dir: str = "./data/organizer",
        sample_dir: str = "./data/sample",
        db_path: str = "./data/factorypulse.db"
    ):
        """
        Initialize the DataLoader with directory paths and default state.

        Parameters:
            organizer_dir: Path to directory where official organizer datasets reside.
            sample_dir: Path to directory where demo synthetic datasets reside.
            db_path: Optional SQLite path for local data persistence.
        """
        # Robust path resolution to repo root if current working directory differs
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if not os.path.isabs(organizer_dir) and not os.path.exists(organizer_dir):
            candidate = os.path.join(root_dir, organizer_dir.lstrip("./").lstrip(".\\"))
            if os.path.exists(candidate) or os.path.exists(os.path.dirname(candidate)):
                organizer_dir = candidate
        if not os.path.isabs(sample_dir) and not os.path.exists(sample_dir):
            candidate = os.path.join(root_dir, sample_dir.lstrip("./").lstrip(".\\"))
            if os.path.exists(candidate) or os.path.exists(os.path.dirname(candidate)):
                sample_dir = candidate

        self.organizer_dir = organizer_dir
        self.sample_dir = sample_dir
        self.db_path = db_path

        self.df_inspection: Optional[pd.DataFrame] = None
        self.df_process: Optional[pd.DataFrame] = None
        self.df_production: Optional[pd.DataFrame] = None
        self.df_economics: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def has_organizer_images(self) -> bool:
        """Check if organizer_dir contains image files or class subfolders."""
        if not os.path.exists(self.organizer_dir):
            return False
        class_names = {"normal", "crack", "hole", "rust", "scratch"}
        for sub in [self.organizer_dir, os.path.join(self.organizer_dir, "train"), os.path.join(self.organizer_dir, "train", "train")]:
            if os.path.exists(sub):
                try:
                    for d in os.listdir(sub):
                        if d.lower() in class_names:
                            return True
                except Exception:
                    pass
        for root, dirs, files in os.walk(self.organizer_dir, followlinks=False):
            for d in dirs:
                if d.lower() in class_names:
                    return True
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    return True
        return False

    def scan_organizer_images(self) -> Dict[str, List[str]]:
        """
        Scan organizer directory for the 5 official class folders:
        'normal', 'crack', 'hole', 'rust', 'scratch'.
        Returns dictionary mapping class name to list of file paths.
        """
        classes = ["normal", "crack", "hole", "rust", "scratch"]
        results: Dict[str, List[str]] = {c: [] for c in classes}
        if not os.path.exists(self.organizer_dir):
            return results

        candidates = [
            os.path.join(self.organizer_dir, "train"),
            os.path.join(self.organizer_dir, "train", "train"),
            self.organizer_dir
        ]
        found_dir = None
        for cand in candidates:
            if os.path.exists(cand):
                try:
                    if any(os.path.exists(os.path.join(cand, c)) for c in classes):
                        found_dir = cand
                        break
                except Exception:
                    pass

        if found_dir:
            for c in classes:
                c_path = os.path.join(found_dir, c)
                if os.path.exists(c_path):
                    files = sorted([
                        os.path.join(c_path, f)
                        for f in os.listdir(c_path)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
                    ])
                    results[c] = files
        else:
            for root, dirs, files in os.walk(self.organizer_dir, followlinks=False):
                folder_name = os.path.basename(root).lower()
                if folder_name in results:
                    img_files = sorted([
                        os.path.join(root, f)
                        for f in files
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
                    ])
                    results[folder_name].extend(img_files)

        return results

    def build_organizer_inspection_table(self, sample_per_class: Optional[int] = None) -> pd.DataFrame:
        """
        Build the inspection table strictly from the 5 class folders:
        label = good if normal else defective; defect_category = folder name; image_path.
        Invent no other columns.
        """
        classes = ["normal", "crack", "hole", "rust", "scratch"]
        train_dir = os.path.join(self.organizer_dir, "train")
        if not os.path.exists(train_dir):
            train_dir = self.organizer_dir

        rows = []
        for c in classes:
            c_dir = os.path.join(train_dir, c)
            if os.path.exists(c_dir):
                files = sorted([f for f in os.listdir(c_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                if sample_per_class is not None:
                    files = files[:sample_per_class]
                for f in files:
                    rel_p = f"data/organizer/train/{c}/{f}"
                    rows.append({
                        "label": "good" if c == "normal" else "defective",
                        "defect_category": c,
                        "image_path": rel_p
                    })

        df = pd.DataFrame(rows, columns=["label", "defect_category", "image_path"])
        os.makedirs(self.organizer_dir, exist_ok=True)
        df.to_csv(os.path.join(self.organizer_dir, "inspection.csv"), index=False)
        return df

    def build_organizer_tables(self) -> bool:
        """Backwards-compatible alias for building organizer inspection table."""
        self.build_organizer_inspection_table()
        return True

    def detect_mode(self) -> str:
        """
        Detect active data mode based on filesystem state.
        Returns 'Organizer' if files exist in data/organizer/ (CSVs or images), else 'Demo'.
        """
        if os.path.exists(self.organizer_dir):
            files = [
                f for f in os.listdir(self.organizer_dir)
                if not f.startswith(".") and f.endswith((".csv", ".tsv", ".txt"))
            ]
            if len(files) > 0:
                return "Organizer"
            if self.has_organizer_images():
                return "Organizer"
        return "Demo"

    def load_csv_data(
        self,
        source: Union[str, Any],
        dataset_type: str,
        expected_cols: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Read, remap columns, validate schema, and clean missing values for a dataset.

        Parameters:
            source: Filepath string or file-like buffer (e.g. UploadedFile).
            dataset_type: One of 'inspection', 'process', 'production', 'economic'.
            expected_cols: List of required canonical column names.

        Returns:
            Tuple of (cleaned_df, validation_report_dict)
        """
        try:
            df_raw = pd.read_csv(source)
        except Exception as e:
            df_raw = pd.DataFrame(columns=expected_cols)

        # 1. Remap organizer column names via centralized config/schema.py
        df_remapped = remap_columns(df_raw, dataset_type=dataset_type)

        # 2. Validate columns and generate descriptive missing column message
        is_valid, missing_cols, val_msg = validate_columns(df_remapped, expected_cols, dataset_type)

        # Ensure missing expected columns exist as NaN to allow graceful degradation
        for col in missing_cols:
            df_remapped[col] = np.nan

        # 3. Clean missing values according to documented strategy
        df_cleaned = self._clean_missing_values(df_remapped, dataset_type)

        report = {
            "dataset": dataset_type,
            "is_valid": is_valid,
            "missing_columns": missing_cols,
            "validation_message": val_msg,
            "initial_rows": len(df_raw),
            "cleaned_rows": len(df_cleaned)
        }
        return df_cleaned, report

    def _clean_missing_values(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Apply rigorous domain-specific missing value cleaning strategies:

        Strategy:
        - label (Quality): Convert text strings ('pass','good','ok' -> 0; 'fail','ng','defect' -> 1)
          and fill NaNs with 0 (assume conforming unless marked defective).
        - Numeric process/production metrics: Fill NaNs with column median; if column is all NaN, fill with 0.0.
        - Categorical identifiers (variant, station, defect_category): Fill NaNs with 'Unknown' or 'None'.
        - Timestamps: Fill missing timestamps using forward-fill then backward-fill.
        """
        cleaned = df.copy()

        # Quality Inspection cleaning
        if dataset_type == "inspection":
            if "label" in cleaned.columns:
                cleaned["label"] = cleaned["label"].apply(self._standardize_label)
            if "defect_category" in cleaned.columns:
                cleaned["defect_category"] = cleaned["defect_category"].fillna("None").astype(str)
                # Ensure conforming units have category 'None'
                if "label" in cleaned.columns:
                    cleaned.loc[cleaned["label"] == 0, "defect_category"] = "None"
            if "variant" in cleaned.columns:
                cleaned["variant"] = cleaned["variant"].fillna("Standard-Variant").astype(str)
            if "unit_id" in cleaned.columns:
                if cleaned["unit_id"].isnull().any():
                    fallback_ids = pd.Series([f"U{1000 + i}" for i in range(len(cleaned))], index=cleaned.index)
                    cleaned["unit_id"] = cleaned["unit_id"].fillna(fallback_ids).astype(str)
                else:
                    cleaned["unit_id"] = cleaned["unit_id"].astype(str)

        # Process Telemetry cleaning
        elif dataset_type == "process":
            numeric_cols = ["temperature", "speed", "pressure", "vibration"]
            for col in numeric_cols:
                if col in cleaned.columns:
                    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                    median_val = cleaned[col].median()
                    cleaned[col] = cleaned[col].fillna(median_val if pd.notnull(median_val) else 0.0)

        # Production Flow cleaning
        elif dataset_type == "production":
            numeric_cols = [
                "cycle_time", "capacity", "wip", "downtime_hours",
                "units_in", "units_out", "rework_units", "scrap_units"
            ]
            for col in numeric_cols:
                if col in cleaned.columns:
                    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                    median_val = cleaned[col].median()
                    cleaned[col] = cleaned[col].fillna(median_val if pd.notnull(median_val) else 0.0)
            if "station" in cleaned.columns:
                cleaned["station"] = cleaned["station"].fillna("Unassigned Station").astype(str)

        # Economic parameters cleaning
        elif dataset_type == "economic":
            for col in ECONOMIC_EXPECTED:
                if col in cleaned.columns:
                    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0.0)

        # General timestamp cleaning across all datasets
        if "timestamp" in cleaned.columns:
            cleaned["timestamp"] = cleaned["timestamp"].ffill().bfill().fillna("2026-09-19 08:00:00")

        return cleaned

    @staticmethod
    def _standardize_label(val: Any) -> int:
        """Convert arbitrary pass/fail or binary representations into integer 0 or 1."""
        return 1 if is_defective(val) else 0

    def load_all(
        self,
        mode: Optional[str] = None,
        uploaded_files: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Load all four manufacturing domains under specified or detected mode.

        Parameters:
            mode: 'Organizer', 'Demo', or 'Upload'. If None, auto-detected.
            uploaded_files: Dictionary of file buffers if mode='Upload'.
                           Keys: 'inspection', 'process', 'production', 'economic'.

        Returns:
            Tuple of (df_inspection, df_process, df_production, df_economics, metadata)
        """
        validation_reports = {}

        if mode == "Upload" and uploaded_files:
            active_mode = "Upload"
            is_synthetic = False
            source_name = "User Uploaded CSV Files"

            df_insp, r1 = self.load_csv_data(uploaded_files.get("inspection"), "inspection", INSPECTION_EXPECTED)
            df_proc, r2 = self.load_csv_data(uploaded_files.get("process"), "process", PROCESS_EXPECTED)
            df_prod, r3 = self.load_csv_data(uploaded_files.get("production"), "production", PRODUCTION_EXPECTED)
            df_econ, r4 = self.load_csv_data(uploaded_files.get("economic"), "economic", ECONOMIC_EXPECTED)
            validation_reports = {"inspection": r1, "process": r2, "production": r3, "economic": r4}

        else:
            detected = self.detect_mode() if mode is None else mode
            if detected == "Organizer":
                active_mode = "Organizer"
                is_synthetic = False
                source_name = f"Official Organizer Image Dataset ({self.organizer_dir})"
                target_dir = self.organizer_dir

                # 1. Build or load inspection table strictly with label, defect_category, image_path
                insp_path = os.path.join(self.organizer_dir, "inspection.csv")
                if not os.path.exists(insp_path):
                    df_insp = self.build_organizer_inspection_table()
                else:
                    df_insp = pd.read_csv(insp_path)

                # 2. Keep other tabs on Demo data without faking links (Requirement 4)
                if not os.path.exists(os.path.join(self.sample_dir, "process.csv")):
                    generate_sample_datasets(self.sample_dir, n_batches=20)

                df_demo_insp, _ = self.load_csv_data(os.path.join(self.sample_dir, "inspection.csv"), "inspection", INSPECTION_EXPECTED)
                df_proc, r2 = self.load_csv_data(os.path.join(self.sample_dir, "process.csv"), "process", PROCESS_EXPECTED)
                df_prod, r3 = self.load_csv_data(os.path.join(self.sample_dir, "production.csv"), "production", PRODUCTION_EXPECTED)
                df_econ, r4 = self.load_csv_data(os.path.join(self.sample_dir, "economics.csv"), "economic", ECONOMIC_EXPECTED)
                validation_reports = {
                    "inspection": {"dataset": "inspection", "is_valid": True, "initial_rows": len(df_insp), "cleaned_rows": len(df_insp)},
                    "process": r2,
                    "production": r3,
                    "economic": r4
                }
            else:
                active_mode = "Demo"
                is_synthetic = True
                source_name = f"DEMO / SYNTHETIC DATA ({self.sample_dir})"
                target_dir = self.sample_dir
                # Ensure synthetic sample dataset exists
                if not os.path.exists(os.path.join(self.sample_dir, "inspection.csv")):
                    generate_sample_datasets(self.sample_dir, n_batches=20)

                insp_path = self._find_file(target_dir, ["inspection", "quality", "defects"])
                proc_path = self._find_file(target_dir, ["process", "telemetry", "sensors"])
                prod_path = self._find_file(target_dir, ["production", "flow", "stations"])
                econ_path = self._find_file(target_dir, ["economics", "cost", "financial"])

                df_insp, r1 = self.load_csv_data(insp_path, "inspection", INSPECTION_EXPECTED)
                df_proc, r2 = self.load_csv_data(proc_path, "process", PROCESS_EXPECTED)
                df_prod, r3 = self.load_csv_data(prod_path, "production", PRODUCTION_EXPECTED)
                df_econ, r4 = self.load_csv_data(econ_path, "economic", ECONOMIC_EXPECTED)
                validation_reports = {"inspection": r1, "process": r2, "production": r3, "economic": r4}

        metadata = {
            "data_mode": active_mode,
            "is_synthetic": is_synthetic,
            "source_name": source_name,
            "other_tabs_demo": (active_mode == "Organizer"),
            "demo_inspection": df_demo_insp if active_mode == "Organizer" else None,
            "validation_reports": validation_reports,
            "record_counts": {
                "inspection": len(df_insp),
                "process": len(df_proc),
                "production": len(df_prod),
                "economics": len(df_econ),
            }
        }

        # Provide convenient backwards-compatible aliases for all downstream consumers
        if "label" in df_insp.columns and "is_defective" not in df_insp.columns:
            df_insp["is_defective"] = is_defective(df_insp["label"]).astype(int)
        if "defect_category" in df_insp.columns and "defect_family" not in df_insp.columns:
            df_insp["defect_family"] = df_insp["defect_category"]
        if "variant" in df_insp.columns and "variant_id" not in df_insp.columns:
            df_insp["variant_id"] = df_insp["variant"]

        if "station" in df_prod.columns:
            if "station_id" not in df_prod.columns:
                df_prod["station_id"] = df_prod["station"]
            if "station_name" not in df_prod.columns:
                df_prod["station_name"] = df_prod["station"]
        if "capacity" in df_prod.columns and "nominal_capacity_uph" not in df_prod.columns:
            df_prod["nominal_capacity_uph"] = df_prod["capacity"]
        if "cycle_time" in df_prod.columns and "cycle_time_sec" not in df_prod.columns:
            df_prod["cycle_time_sec"] = df_prod["cycle_time"]
        if "wip" in df_prod.columns and "wip_count" not in df_prod.columns:
            df_prod["wip_count"] = df_prod["wip"]
        if "downtime_hours" in df_prod.columns and "downtime_min" not in df_prod.columns:
            df_prod["downtime_min"] = df_prod["downtime_hours"] * 60.0
        if "rework_units" in df_prod.columns and "rework_count" not in df_prod.columns:
            df_prod["rework_count"] = df_prod["rework_units"]
        if "scrap_units" in df_prod.columns and "scrap_count" not in df_prod.columns:
            df_prod["scrap_count"] = df_prod["scrap_units"]

        if len(df_econ) > 0:
            if "scrap_penalty_per_unit" not in df_econ.columns and "scrap_cost" in df_econ.columns:
                df_econ["scrap_penalty_per_unit"] = df_econ["scrap_cost"]
            if "rework_cost_per_unit" not in df_econ.columns and "rework_cost" in df_econ.columns:
                df_econ["rework_cost_per_unit"] = df_econ["rework_cost"]
            if "downtime_penalty_hourly" not in df_econ.columns and "downtime_cost_per_hour" in df_econ.columns:
                df_econ["downtime_penalty_hourly"] = df_econ["downtime_cost_per_hour"]
            if "labor_rate_hourly" not in df_econ.columns and "labor_cost" in df_econ.columns:
                df_econ["labor_rate_hourly"] = df_econ["labor_cost"]
            if "overhead_rate_hourly" not in df_econ.columns and "overhead_cost" in df_econ.columns:
                df_econ["overhead_rate_hourly"] = df_econ["overhead_cost"]

        self.df_inspection = df_insp
        self.df_process = df_proc
        self.df_production = df_prod
        self.df_economics = df_econ
        self.metadata = metadata

        return df_insp, df_proc, df_prod, df_econ, metadata

    def get_joined_batch_data(self) -> pd.DataFrame:
        """Backwards-compatible alias for join_tables()."""
        return self.join_tables()

    def _find_file(self, directory: str, keywords: List[str]) -> str:
        """Locate CSV file matching keywords within target directory, or fallback gracefully."""
        if not os.path.exists(directory):
            return ""
        for f in os.listdir(directory):
            if f.lower().endswith((".csv", ".tsv")):
                for kw in keywords:
                    if kw in f.lower():
                        return os.path.join(directory, f)
        # Fallback to any CSV if single file exists
        csvs = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".csv")]
        return csvs[0] if csvs else ""

    def join_tables(
        self,
        df_inspection: Optional[pd.DataFrame] = None,
        df_process: Optional[pd.DataFrame] = None,
        df_production: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Join inspection, process, and production tables on batch_id.
        Aggregates unit inspections into batch defect rates before merging.

        Returns:
            Merged pandas DataFrame with batch quality, telemetry, and station metrics.
        """
        df_insp = df_inspection if df_inspection is not None else self.df_inspection
        df_proc = df_process if df_process is not None else self.df_process
        df_prod = df_production if df_production is not None else self.df_production

        if df_insp is None or df_proc is None or len(df_insp) == 0 or len(df_proc) == 0:
            return pd.DataFrame()

        # If df_insp lacks batch_id (pure organizer image dataset), use demo inspection for joined telemetry
        if "batch_id" not in df_insp.columns:
            if hasattr(self, "metadata") and self.metadata.get("demo_inspection") is not None:
                df_insp = self.metadata["demo_inspection"]
            elif os.path.exists(os.path.join(self.sample_dir, "inspection.csv")):
                df_insp, _ = self.load_csv_data(os.path.join(self.sample_dir, "inspection.csv"), "inspection", INSPECTION_EXPECTED)
            else:
                return pd.DataFrame()

        # Aggregate inspection data at the batch level
        lbl_col = "label" if "label" in df_insp.columns else ("is_defective" if "is_defective" in df_insp.columns else None)
        uid_col = "unit_id" if "unit_id" in df_insp.columns else "batch_id"
        temp_insp = df_insp.copy()
        temp_insp["_is_def"] = is_defective(temp_insp[lbl_col]).astype(int) if lbl_col else 0
        batch_quality = temp_insp.groupby("batch_id").agg(
            total_units=(uid_col, "count"),
            defect_count=("_is_def", "sum"),
        ).reset_index()
        batch_quality["defect_rate"] = batch_quality["defect_count"] / np.maximum(1, batch_quality["total_units"])

        # Join Quality with Process Telemetry on batch_id
        joined = pd.merge(batch_quality, df_proc, on="batch_id", how="inner")

        # Optionally join aggregated production metrics if production data provided
        if df_prod is not None and len(df_prod) > 0 and "batch_id" in df_prod.columns:
            prod_batch = df_prod.groupby("batch_id").agg(
                total_wip=("wip", "sum"),
                total_downtime=("downtime_hours", "sum"),
                batch_rework=("rework_units", "sum"),
                batch_scrap=("scrap_units", "sum")
            ).reset_index()
            joined = pd.merge(joined, prod_batch, on="batch_id", how="left")

        return joined

    def compute_statistics(
        self,
        df_inspection: Optional[pd.DataFrame] = None,
        df_process: Optional[pd.DataFrame] = None,
        df_production: Optional[pd.DataFrame] = None,
        df_economics: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Compute basic and advanced statistics across all four manufacturing domains.
        Relies on Pandas and NumPy only.

        Returns:
            Dictionary containing descriptive statistics for each domain.
        """
        insp = df_inspection if df_inspection is not None else self.df_inspection
        proc = df_process if df_process is not None else self.df_process
        prod = df_production if df_production is not None else self.df_production
        econ = df_economics if df_economics is not None else self.df_economics

        stats: Dict[str, Any] = {}

        # 1. Inspection Statistics
        if insp is not None and len(insp) > 0:
            total_u = len(insp)
            lbl_col = "label" if "label" in insp.columns else ("is_defective" if "is_defective" in insp.columns else None)
            def_mask = is_defective(insp[lbl_col]) if lbl_col else pd.Series(False, index=insp.index)
            defects = int(def_mask.sum())
            rate = float(defects / max(1, total_u))
            variant_counts = insp["variant"].value_counts().to_dict() if "variant" in insp.columns else {}
            category_counts = (
                insp[def_mask]["defect_category"].value_counts().to_dict()
                if "defect_category" in insp.columns else {}
            )
            stats["inspection"] = {
                "total_units": total_u,
                "defective_units": defects,
                "defect_rate_pct": round(rate * 100.0, 2),
                "variant_distribution": variant_counts,
                "defect_category_distribution": category_counts,
                "unique_batches": int(insp["batch_id"].nunique()) if "batch_id" in insp.columns else 0
            }

        # 2. Process Telemetry Statistics
        if proc is not None and len(proc) > 0:
            numeric_params = [c for c in proc.columns if c not in ["batch_id", "timestamp"] and pd.api.types.is_numeric_dtype(proc[c])]
            proc_stats = {}
            for p in numeric_params:
                vals = proc[p].dropna()
                if len(vals) > 0:
                    proc_stats[p] = {
                        "mean": round(float(np.mean(vals)), 2),
                        "std": round(float(np.std(vals)), 2),
                        "min": round(float(np.min(vals)), 2),
                        "max": round(float(np.max(vals)), 2),
                        "median": round(float(np.median(vals)), 2)
                    }
            stats["process"] = proc_stats

        # 3. Production Statistics
        if prod is not None and len(prod) > 0:
            total_in = int(prod["units_in"].sum()) if "units_in" in prod.columns else 0
            total_out = int(prod["units_out"].sum()) if "units_out" in prod.columns else 0
            total_scrap = int(prod["scrap_units"].sum()) if "scrap_units" in prod.columns else 0
            total_rework = int(prod["rework_units"].sum()) if "rework_units" in prod.columns else 0
            total_downtime = float(prod["downtime_hours"].sum()) if "downtime_hours" in prod.columns else 0.0

            # Station bottlenecks: group by station and find station with lowest capacity / highest WIP
            station_summary = {}
            if "station" in prod.columns:
                for st_name, st_df in prod.groupby("station"):
                    station_summary[st_name] = {
                        "mean_capacity": round(float(st_df["capacity"].mean()), 1) if "capacity" in st_df.columns else 0,
                        "mean_wip": round(float(st_df["wip"].mean()), 1) if "wip" in st_df.columns else 0,
                        "total_rework": int(st_df["rework_units"].sum()) if "rework_units" in st_df.columns else 0,
                    }

            stats["production"] = {
                "total_units_in": total_in,
                "total_units_out": total_out,
                "total_scrap": total_scrap,
                "total_rework": total_rework,
                "total_downtime_hours": round(total_downtime, 2),
                "yield_pct": round((total_out / max(1, total_in)) * 100.0, 2),
                "station_breakdown": station_summary
            }

        # 4. Economic Statistics
        if econ is not None and len(econ) > 0:
            first_row = econ.iloc[0].to_dict()
            stats["economic"] = {k: round(float(v), 2) for k, v in first_row.items() if pd.notnull(v)}

        return stats
