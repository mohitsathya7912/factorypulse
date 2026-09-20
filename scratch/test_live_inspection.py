"""
scratch/test_live_inspection.py
Test and measure all metrics for the Live Inspection section, ResNet18 classifier,
utilization alignment, impact chain narrative bolding, and economics scaling.
"""
import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.loader import DataLoader
from src.quality.classifier import QualityClassifier, has_organizer_images
from src.production.flow import analyze_production_line
from src.economics.model import calculate_economics
from src.rootcause.engine import RootCauseEngine, build_batch_table
from src.insights.advisor import DecisionAdvisor


def main():
    print("=================================================================")
    print("1. EVALUATING RESNET18 QUALITY CLASSIFIER & HELD-OUT TEST SPLIT")
    print("=================================================================")
    clf = QualityClassifier(model_dir="./models", organizer_dir="./data/organizer")
    metrics = clf.train_organizer_resnet18(use_full_dataset=False, uncertainty_threshold=0.65)
    
    print(f"Status: {metrics.get('status')}")
    print(f"Feature Type: {metrics.get('feature_type')}")
    print(f"Train samples: {metrics.get('train_samples')}")
    print(f"Test samples (held-out): {metrics.get('test_samples')}")
    print(f"Test Accuracy: {metrics.get('accuracy') * 100:.2f}%")
    print(f"Macro F1-Score: {metrics.get('f1_macro'):.4f}")
    print(f"Precision: {metrics.get('precision') * 100:.2f}%")
    print(f"Recall: {metrics.get('recall') * 100:.2f}%")
    print(f"False-Reject Rate (FRR): {metrics.get('false_reject_rate') * 100:.2f}%")
    print(f"False-Accept Rate (FAR): {metrics.get('false_accept_rate') * 100:.2f}%")
    print(f"Stage 1 Confusion Matrix (TN, FP / FN, TP):\n{np.array(metrics.get('confusion_matrix'))}")
    
    stage2 = metrics.get("stage2", {})
    if stage2.get("available"):
        print(f"\nStage 2 Defect Categories: {stage2.get('classes')}")
        print(f"Stage 2 Test Accuracy: {stage2.get('accuracy') * 100:.2f}%")
        print(f"Stage 2 Macro F1: {stage2.get('f1_macro'):.4f}")
        print(f"Stage 2 Confusion Matrix:\n{np.array(stage2.get('confusion_matrix'))}")

    # Check held-out test predictions
    test_pred_df = metrics.get("test_predictions", pd.DataFrame())
    print(f"\nHeld-out test set size: {len(test_pred_df)} records")
    print(f"Test split columns: {list(test_pred_df.columns)}")
    print(f"Distribution of test true categories:\n{test_pred_df['actual_category'].value_counts()}")
    print(f"Distribution of test predicted categories:\n{test_pred_df['predicted_category'].value_counts()}")
    
    # 2. Test Live File Upload & Persistence
    print("\n=================================================================")
    print("2. TESTING LIVE IMAGE PERSISTENCE & RESNET18 INFERENCE")
    print("=================================================================")
    upload_dir = os.path.join(ROOT_DIR, "data", "uploads", "images")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Pick a sample test image from each class to simulate uploads
    categories = ["normal", "crack", "hole", "rust", "scratch"]
    sample_uploads = []
    for cat in categories:
        img_files = glob.glob(os.path.join(ROOT_DIR, "data", "organizer", "train", cat, "*.png"))
        if img_files:
            sample_uploads.append((cat, img_files[0]))
            
    print(f"Found {len(sample_uploads)} sample images to upload and test.")
    live_results = []
    for cat, img_path in sample_uploads:
        dest_name = f"test_live_{cat}_{os.path.basename(img_path)}"
        dest_path = os.path.join(upload_dir, dest_name)
        # Copy to uploads to simulate persistence
        with open(img_path, "rb") as fin:
            with open(dest_path, "wb") as fout:
                fout.write(fin.read())
        assert os.path.exists(dest_path), f"File {dest_path} was not persisted!"
        
        # Predict
        res = clf.predict_image(dest_path, uncertainty_threshold=0.65)
        live_results.append({
            "filename": dest_name,
            "true_cat": cat,
            "status": res["status"],
            "badge_label": res["badge_label"],
            "badge_color": res["badge_color"],
            "defect_category": res["defect_category"],
            "confidence_pct": res["confidence_pct"]
        })
        print(f"Image: {dest_name[:25]}... -> Status: {res['status']}, Badge: {res['badge_label']}, Cat: {res['defect_category']}, Conf: {res['confidence_pct']:.1f}%")

    n_total = len(live_results)
    n_good = sum(1 for p in live_results if p["status"] == "GOOD")
    n_def = sum(1 for p in live_results if p["status"] == "DEFECTIVE")
    n_unc = sum(1 for p in live_results if p["status"] == "UNCERTAIN")
    print(f"\nLive Inspection Summary: Total={n_total}, Good={n_good}, Defective={n_def}, Uncertain={n_unc}")

    # 3. Test Random Test Samples (Held-out only)
    print("\n=================================================================")
    print("3. TESTING RANDOM SAMPLES FROM HELD-OUT SPLIT")
    print("=================================================================")
    sampled = test_pred_df.sample(min(4, len(test_pred_df)), random_state=42)
    # Verify these image paths belong to the test set
    for _, row in sampled.iterrows():
        print(f"Sample: {os.path.basename(row['image_path'])} | Pred: {row['prediction']} ({row['predicted_category']}) | True: {row['actual_label']} ({row['actual_category']}) | Conf: {row['confidence']*100:.1f}%")

    # 4. Test Utilization Scaling & Impact Chain Consistency
    print("\n=================================================================")
    print("4. VERIFYING UTILIZATION SCALING & IMPACT CHAIN CONSISTENCY")
    print("=================================================================")
    loader = DataLoader()
    df_insp, df_proc, df_prod, df_econ, demo_meta = loader.load_all(mode="Demo")
    
    # Station flow analysis
    station_col = "station" if "station" in df_prod.columns else "station_id"
    station_records = []
    for st_name, st_group in df_prod.groupby(station_col):
        ct_val = float(st_group["cycle_time"].mean())
        dt_hrs = float(st_group["downtime_hours"].sum())
        u_in = float(st_group["units_in"].sum())
        u_out = float(st_group["units_out"].sum())
        rew = float(st_group["rework_units"].sum())
        scr = float(st_group["scrap_units"].sum())
        wip = float(st_group["wip"].mean())
        station_records.append({
            "station": str(st_name), "cycle_time": ct_val, "downtime_hours": dt_hrs,
            "units_in": u_in, "units_out": u_out, "rework_units": rew, "scrap_units": scr, "wip": wip
        })
    prod_res = analyze_production_line(station_records, hours=8.0)
    bn = prod_res.get("bottleneck_station", {})
    kpi_bn_util = bn.get("utilization", 0.0) * 100.0
    print(f"Production flow bottleneck: {bn.get('station')} with utilization: {kpi_bn_util:.1f}% (raw: {bn.get('utilization')})")

    # Economics calculation
    econ_res = calculate_economics(
        good_units=970,
        unit_price=85.00,
        material_cost=32000.0,
        labor_cost=7200.0,
        overhead_cost=240.0,
        scrap_cost=0.0,
        rework_cost=0.0,
        downtime_cost=2280.0
    )
    batch_df = build_batch_table(df_insp, df_proc, df_prod)
    rc_engine = RootCauseEngine(batch_df, is_synthetic=True)
    rc_results = rc_engine.analyze_factors()

    advisor = DecisionAdvisor(
        production_results=prod_res,
        economic_results=econ_res,
        rootcause_results=rc_results,
        df_insp=df_insp
    )
    ic = advisor.compute_impact_chain()
    bn_step = [s for s in ic["steps"] if s["title"] == "Primary Line Bottleneck"][0]
    print(f"Impact Chain Bottleneck Step: Name={bn_step['name']}, Metric={bn_step['metric']}")
    print(f"Impact Narrative:\n{ic['narrative']}")

    # Check for bold tags
    assert "<b>" in ic["narrative"] and "</b>" in ic["narrative"], "Impact Narrative must contain proper <b> HTML bold tags!"
    assert "**" not in ic["narrative"], "Impact Narrative should not contain unparsed ** markdown!"
    # Check utilization alignment
    assert f"{kpi_bn_util:.1f}% Utilization" == bn_step["metric"], f"Utilization mismatch! KPI has {kpi_bn_util:.1f}% but Impact Chain has {bn_step['metric']}"
    print("SUCCESS: Utilization percentage matches exactly between KPI and Impact Chain!")

    # 5. Check Organizer Economics Scaling
    print("\n=================================================================")
    print("5. VERIFYING DEMO REVENUE/COST SCALING IN ORGANIZER MODE")
    print("=================================================================")
    org_insp, org_proc, org_prod, org_econ, org_meta = loader.load_all(mode="Organizer")
    org_demo_insp = org_meta["demo_inspection"]
    print(f"Organizer total image count: {len(org_insp):,} images")
    print(f"Organizer demo units for telemetry: {len(org_demo_insp):,} units")
    
    # In app.py:
    # ov_units = len(filtered_demo_insp) if data_mode == "Organizer" else total_units
    # econ_units = ov_units if data_mode == "Organizer" else total_units
    econ_units = len(org_demo_insp)
    good_units = max(0, econ_units - 0)
    tot_mat = econ_units * 32.00
    demo_econ = calculate_economics(
        good_units=good_units,
        unit_price=85.00,
        material_cost=tot_mat,
        labor_cost=7200.0,
        overhead_cost=240.0,
        scrap_cost=0.0,
        rework_cost=0.0,
        downtime_cost=2280.0
    )
    print(f"Organizer Mode Economics Revenue: ${demo_econ['revenue']:,.2f}")
    print(f"Organizer Mode Economics Production Cost: ${demo_econ['cost']:,.2f}")
    print(f"Organizer Mode Operating Profit: ${demo_econ['profit']:,.2f}")
    assert demo_econ["revenue"] == 85000.0, f"Revenue should be $85,000 for 1,000 units, but got {demo_econ['revenue']}"
    assert demo_econ["cost"] < 50000.0, f"Cost should be ~$41,720, but got {demo_econ['cost']}"
    print("SUCCESS: Demo revenue/cost is NOT scaled by the 12,000 image count!")


if __name__ == "__main__":
    main()
