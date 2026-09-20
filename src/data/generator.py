"""
src/data/generator.py
Synthetic Data Generator for FactoryPulse DEMO Mode.
Generates ~20 batches, 2-3 product variants, 5 stations with one primary bottleneck,
3-4 defect families (~8-12% overall defect rate), and drifting process telemetry.

Strictly relies on Pandas and NumPy only.
All outputs are saved with explicit synthetic metadata and warning notices.
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from config.schema import is_defective

def generate_sample_datasets(output_dir: str = "./data/sample", n_batches: int = 20) -> None:
    """
    Generate complete, realistic synthetic manufacturing datasets for Demo Mode.

    Parameters:
        output_dir: Path to directory where CSVs and metadata will be saved.
        n_batches: Number of manufacturing batches to simulate (default: 20).
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    np.random.seed(42)
    units_per_batch = 50
    total_units = n_batches * units_per_batch
    batches = [f"B{101 + i}" for i in range(n_batches)]
    variants = ["Variant-Alpha", "Variant-Beta", "Variant-Gamma"]
    base_timestamp = pd.Timestamp("2026-09-15 08:00:00")

    # ==========================================================================
    # 1. PROCESS TELEMETRY (~20 batches, 4 drifting parameters)
    # ==========================================================================
    # Drift is introduced in batches 11 to 15 (B112 to B116)
    drift_batch_indices = set(range(11, 16))
    process_rows = []

    for i, b_id in enumerate(batches):
        batch_time = base_timestamp + pd.Timedelta(hours=i * 4)
        is_drifting = i in drift_batch_indices

        # Drifting parameters: temperature, speed, pressure, vibration
        temp = np.random.normal(218.0 if is_drifting else 195.0, 2.5)
        speed = np.random.normal(18.2 if is_drifting else 15.0, 0.6)
        pressure = np.random.normal(4.85 if is_drifting else 4.20, 0.15)
        vibration = np.random.normal(2.15 if is_drifting else 1.20, 0.12)

        process_rows.append({
            "batch_id": b_id,
            "timestamp": batch_time.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": round(float(temp), 2),
            "speed": round(float(speed), 2),
            "pressure": round(float(pressure), 2),
            "vibration": round(float(vibration), 2),
        })

    df_process = pd.DataFrame(process_rows)
    df_process.to_csv(os.path.join(output_dir, "process.csv"), index=False)

    # ==========================================================================
    # 2. INSPECTION DATA (~1000 units, 2-3 variants, ~8-12% defect rate)
    # ==========================================================================
    # Defect families: None, Surface Scratch, Pinhole Void, Crack, Contamination
    defect_families = ["Surface Scratch", "Pinhole Void", "Crack", "Contamination"]
    inspection_rows = []
    unit_counter = 1000

    for i, b_id in enumerate(batches):
        is_drifting = i in drift_batch_indices
        # Normal batches have ~6% defects; drifting batches jump to ~22% (yielding ~10% overall average)
        batch_defect_prob = 0.22 if is_drifting else 0.06

        for u in range(units_per_batch):
            unit_id = f"U{unit_counter}"
            unit_counter += 1
            unit_time = base_timestamp + pd.Timedelta(hours=i * 4, minutes=u * 4)
            variant = str(np.random.choice(variants, p=[0.50, 0.30, 0.20]))

            is_def_unit = int(np.random.rand() < batch_defect_prob)
            if is_def_unit == 1:
                # Thermal drift induces mostly Pinhole Void and Crack defects
                if is_drifting:
                    category = str(np.random.choice(
                        defect_families, p=[0.15, 0.60, 0.15, 0.10]
                    ))
                else:
                    category = str(np.random.choice(
                        defect_families, p=[0.45, 0.25, 0.15, 0.15]
                    ))
                bx = int(np.random.randint(25, 65))
                by = int(np.random.randint(25, 65))
                bw = int(np.random.randint(15, 30))
                bh = int(np.random.randint(15, 30))
                bbox_str = f"[{bx},{by},{bw},{bh}]"
            else:
                category = "None"
                bbox_str = "[]"

            img_rel_path = f"data/sample/images/{unit_id}.png"
            if u < 2 or (is_def_unit == 1 and u < 8):
                img_path = os.path.join(images_dir, f"{unit_id}.png")
                _generate_sample_image(img_path, unit_id, is_def_unit, category)

            # Realistic tabular inspection measurements (Demo Mode)
            if is_def_unit == 1:
                roughness = round(float(np.random.normal(2.65, 0.45)), 2)
                dim_dev = round(float(np.random.normal(0.14, 0.05)), 3)
                optical_score = round(float(np.random.normal(81.0, 5.5)), 1)
            else:
                roughness = round(float(np.random.normal(1.25, 0.18)), 2)
                dim_dev = round(float(np.random.normal(0.015, 0.035)), 3)
                optical_score = round(float(np.random.normal(94.5, 2.8)), 1)

            inspection_rows.append({
                "unit_id": unit_id,
                "batch_id": b_id,
                "variant": variant,
                "timestamp": unit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "label": is_def_unit, # 0 = good, 1 = defective
                "defect_category": category,
                "image_path": img_rel_path,
                "bbox": bbox_str,
                "roughness_ra": roughness,
                "dimension_delta_mm": dim_dev,
                "optical_score": optical_score
            })


    df_inspection = pd.DataFrame(inspection_rows)
    df_inspection.to_csv(os.path.join(output_dir, "inspection.csv"), index=False)

    # ==========================================================================
    # 3. PRODUCTION FLOW (5 stations, Station 3 is the clear primary bottleneck)
    # ==========================================================================
    # Stations:
    # 1. Feed (Cap: 150 UPH, CT: 24s)
    # 2. Press & Stamp (Cap: 140 UPH, CT: 25.7s)
    # 3. Thermal Curing (Cap: 100 UPH, CT: 36.0s) -> PRIMARY BOTTLENECK
    # 4. Surface Coating (Cap: 130 UPH, CT: 27.7s)
    # 5. Final Inspection (Cap: 160 UPH, CT: 22.5s)
    stations_config = [
        {"name": "Station 1 - Feed", "capacity": 150, "cycle_time": 24.0, "base_downtime": 0.20},
        {"name": "Station 2 - Stamping", "capacity": 140, "cycle_time": 25.7, "base_downtime": 0.35},
        {"name": "Station 3 - Thermal Curing", "capacity": 100, "cycle_time": 36.0, "base_downtime": 0.85}, # Constraint!
        {"name": "Station 4 - Coating", "capacity": 130, "cycle_time": 27.7, "base_downtime": 0.25},
        {"name": "Station 5 - Final Quality", "capacity": 160, "cycle_time": 22.5, "base_downtime": 0.15},
    ]

    production_rows = []
    for b_id in batches:
        b_defects = int(is_defective(df_inspection[df_inspection["batch_id"] == b_id]["label"]).sum())
        # 75% of defects undergo rework (cycling back through Curing), 25% scrapped
        b_rework = int(b_defects * 0.75)
        b_scrap = b_defects - b_rework

        for st in stations_config:
            is_bn = "Station 3" in st["name"]
            # Station 3 absorbs full rework loop load, leading to high WIP buffer build-up
            st_rework = b_rework if is_bn else int(b_rework * 0.25)
            st_scrap = int(b_scrap / 5)
            wip = int(np.random.randint(65, 95)) if is_bn else int(np.random.randint(12, 28))
            units_in = units_per_batch + st_rework
            units_out = units_in - st_scrap

            production_rows.append({
                "station": st["name"],
                "batch_id": b_id,
                "cycle_time": st["cycle_time"],
                "capacity": st["capacity"],
                "wip": wip,
                "downtime_hours": round(st["base_downtime"] + (0.4 if is_bn and b_defects > 8 else 0.0), 2),
                "units_in": units_in,
                "units_out": units_out,
                "rework_units": st_rework,
                "scrap_units": st_scrap
            })

    df_production = pd.DataFrame(production_rows)
    df_production.to_csv(os.path.join(output_dir, "production.csv"), index=False)

    # ==========================================================================
    # 4. ECONOMIC DATA
    # ==========================================================================
    economic_rows = [{
        "unit_price": 85.00,
        "material_cost": 32.00,
        "labor_cost": 45.00,
        "overhead_cost": 30.00,
        "scrap_cost": 42.00,
        "rework_cost": 18.50,
        "downtime_cost_per_hour": 150.00,
    }]
    df_economics = pd.DataFrame(economic_rows)
    df_economics.to_csv(os.path.join(output_dir, "economics.csv"), index=False)

    # ==========================================================================
    # 5. METADATA & SYNTHETIC NOTICE NOTE
    # ==========================================================================
    overall_defect_rate = float(is_defective(df_inspection["label"]).mean() * 100.0)

    metadata = {
        "is_synthetic": True,
        "data_mode": "Demo",
        "batches_count": n_batches,
        "total_units": total_units,
        "defect_rate_pct": round(overall_defect_rate, 2),
        "primary_bottleneck": "Station 3 - Thermal Curing (100 UPH)",
        "drifting_parameters": ["temperature", "speed", "pressure", "vibration"],
        "notice": "DEMO / SYNTHETIC DATA — Generated for hackathon evaluation baseline."
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    note_text = (
        "=====================================================================\n"
        "FACTORYPULSE SYNTHETIC DATASET NOTICE (DEMO MODE)\n"
        "=====================================================================\n"
        "This dataset was generated by src/data/generator.py for demonstration,\n"
        "testing, and baseline evaluation of FactoryPulse.\n\n"
        f"- Batches Simulated: {n_batches}\n"
        f"- Product Variants: {', '.join(variants)}\n"
        f"- Overall Defect Rate: {overall_defect_rate:.2f}%\n"
        "- Stations Simulated: 5 (Primary Bottleneck: Station 3 - Thermal Curing)\n"
        "- Drifting Telemetry Parameters: temperature, speed, pressure, vibration\n"
        "- Metadata Flag: is_synthetic = True\n\n"
        "When real organizer datasets are dropped into data/organizer/,\n"
        "FactoryPulse automatically switches to 'Organizer' mode.\n"
        "=====================================================================\n"
    )
    with open(os.path.join(output_dir, "SYNTHETIC_DATA_NOTE.txt"), "w", encoding="utf-8") as f:
        f.write(note_text)


def _generate_sample_image(path: str, unit_id: str, unit_is_defective: int, defect_category: str) -> None:
    """Draw a 256x256 simulated manufacturing component with optional visual defect."""
    img = Image.new("RGB", (256, 256), color=(222, 226, 232))
    draw = ImageDraw.Draw(img)

    # Base component outline
    draw.rounded_rectangle([25, 25, 231, 231], radius=14, fill=(188, 194, 204), outline=(120, 130, 145), width=3)
    for hx, hy in [(45, 45), (211, 45), (45, 211), (211, 211)]:
        draw.ellipse([hx - 8, hy - 8, hx + 8, hy + 8], fill=(85, 90, 100), outline=(50, 55, 65))
    draw.rectangle([65, 65, 191, 191], fill=(208, 215, 225), outline=(145, 155, 170), width=2)

    # Defect visual rendering
    if unit_is_defective == 1:
        if defect_category == "Surface Scratch":
            draw.line([(85, 90), (170, 160)], fill=(60, 20, 20), width=3)
            draw.line([(88, 92), (167, 158)], fill=(225, 40, 40), width=1)
        elif defect_category == "Pinhole Void":
            for ox, oy in [(110, 120), (116, 126), (124, 118), (119, 132), (128, 128)]:
                draw.ellipse([ox - 4, oy - 4, ox + 4, oy + 4], fill=(25, 25, 25))
        elif defect_category == "Crack":
            draw.line([(100, 80), (115, 110), (108, 135), (128, 170)], fill=(35, 15, 15), width=3)
        else: # Contamination
            draw.ellipse([95, 95, 155, 145], fill=(85, 80, 55), outline=(55, 50, 35))

    img.save(path)

if __name__ == "__main__":
    generate_sample_datasets()
    print("FactoryPulse synthetic dataset generated successfully.")
