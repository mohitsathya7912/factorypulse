# Running FactoryPulse (NEURAX Hackathon 3.0)

FactoryPulse is an AI-powered manufacturing decision support system designed for high-throughput lines. It unifies visual inspection, production bottleneck flow, plant economics, and statistical root-cause attribution.

---

## 1. Quickstart & Installation

### Step 1: Clone or Open Project Directory
```bash
cd neurax
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
# Tip: If script execution is disabled in PowerShell, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
```bash
# Windows
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

---

## 2. Launching the Dashboard

Run the Streamlit application:
```bash
streamlit run dashboard/app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 3. Switching from Demo Mode to Real Organizer Data

FactoryPulse is built to adapt dynamically to real hackathon datasets with zero code changes:

- **DEMO / SYNTHETIC DATA MODE (Default):**
  When `data/organizer/` contains no dataset files, FactoryPulse automatically generates and loads a realistic 20-batch multi-stage dataset (`data/sample/`) modeling thermal curing bottlenecks and process drift. A visible red **DEMO / SYNTHETIC DATA** banner is displayed on all views.

- **REAL ORGANIZER DATA MODE:**
  Simply place the organizer's provided data files (CSVs, Excel, images) into:
  ```
  data/organizer/
  ```
  FactoryPulse automatically detects these files, turns the status banner green (**OFFICIAL ORGANIZER DATA LOADED**), and maps incoming column names to internal schemas via `config/schema.py`.

- **UPLOAD MODE:**
  Select "Upload" in the sidebar to upload individual CSV files directly through the Streamlit interface.

---

## 4. How to Remap Columns in `config/schema.py`

FactoryPulse standardizes all incoming datasets into canonical internal schemas via a central alias dictionary in `config/schema.py`. You only ever need to edit **ONE file** to support new column naming conventions.

### Canonical Column Schemas:
| Dataset | Canonical Columns |
| :--- | :--- |
| **Inspection** | `unit_id`, `batch_id`, `variant`, `timestamp`, `label`, `defect_category`, optional: `image_path`, `bbox` |
| **Process** | `batch_id`, `timestamp`, `temperature`, `speed`, `pressure`, `vibration` |
| **Production** | `station`, `batch_id`, `cycle_time`, `capacity`, `wip`, `downtime_hours`, `units_in`, `units_out`, `rework_units`, `scrap_units` |
| **Economic** | `unit_price`, `material_cost`, `labor_cost`, `overhead_cost`, `scrap_cost`, `rework_cost`, `downtime_cost_per_hour` |

### Adding a New Column Alias:
Open [`config/schema.py`](file:///c:/Users/vivek/OneDrive/Desktop/neurax/config/schema.py) and locate `COLUMN_MAPPING`. Add your organizer column name as the key and the canonical name as the value:

```python
COLUMN_MAPPING = {
    "inspection": {
        "your_custom_part_id": "unit_id",     # Maps your custom ID to canonical unit_id
        "pass_or_fail": "label",              # Maps pass/fail flag to label (0/1)
        "anomaly_class": "defect_category",   # Maps defect name to defect_category
        ...
    },
    "process": {
        "furnace_temp_deg": "temperature",
        "extrusion_bar": "pressure",
        ...
    },
    "production": {
        "machine_code": "station",
        "idle_minutes": "downtime_hours",
        ...
    },
    "economic": {
        "sales_price_eur": "unit_price",
        ...
    }
}
```

The loader automatically trims whitespace, converts names to lowercase, and translates all mapped aliases into the canonical schema before downstream validation.

---

## 5. Dashboard Views (6 Tabs)

1. **📊 Overview:** Executive summary with 8 KPI cards (total units, defect rate, throughput, bottleneck station, revenue, cost, profit, margin) plus the **End-to-End Operational Impact Chain Panel** tracing defect root causes to bottom-line margin.
2. **⚙️ Production:** Station utilization bar chart with the bottleneck highlighted, cycle times, WIP, downtime, rework load, and throughput lost to rework.
3. **💰 Economics:** Revenue vs cost waterfall breakdown, operating profit, margin, cost per good unit, COPQ, and interactive What-If scenario simulator displaying DELTAS vs baseline (labeled *Simulated / advisory*).
4. **🔍 Quality AI:** Two-Stage Visual & Tabular Defect Classifier.
   - Stage 1: Binary Good vs Defective with class-balanced weighting.
   - Stage 2: Defect Category / Family classification on defective parts.
   - Held-Out Evaluation: Stratified Group Split by `batch_id` to strictly prevent telemetry leakage across batches. Computes Accuracy, Precision, Recall, Macro F1, False-Reject Rate (FRR: $FP/(TN+FP)$), False-Accept Rate (FAR: $FN/(TP+FN)$), and Confusion Matrices.
   - Selective Classification: Predictions below confidence threshold ($T_{conf}$) are flagged as `UNCERTAIN` and routed to the Human Inspector Triage Queue with CSV export.
   - Model Persistence: Saved/loaded with `joblib` (`models/quality_model.joblib`) to eliminate redundant retraining.
5. **🔬 Root Cause:** Statistical Process Association & Drift Attribution (simple, honest non-parametric statistics).
   - Spearman rank correlation ($\rho$) with batch defect rate and exact p-values.
   - High-vs-Low median split comparison with $\Delta$ defect rate and Mann-Whitney U test.
   - $2\sigma$ control limit outlier detection and affected batch defect rate calculation.
   - Parameter drift charts across batches and defect family association matrix.
   - Strict language: *"statistical associations / likely contributing factors, not proven causes"*.
6. **💡 AI Insights:** Rule-Based Decision Support & End-to-End Impact Chain.
   - Computed operational finding cards derived strictly from empirical numbers.
   - Full 6-stage Impact Chain connecting Top Defect Family ➔ Process Factor ➔ Bottleneck Rework ➔ Lost Throughput ➔ COPQ ➔ Operating Margin.
   - Targeted countermeasure recommendations labeled *SIMULATED / ADVISORY* with empirical evidence basis and estimated margin recovery.

---

## 6. Rebuilding Cache and Models from Scratch (Named Commands)

All models and feature representations can be completely rebuilt from raw files with a single named command. Precomputed artifacts are stored in `cache/` and `models/`:

### Option A: Using the Central CLI Script
```bash
# 1. Fast Dev-Mode Rebuild (400 images per class, 2,000 total, ~2 minutes on CPU)
python scripts/rebuild_cache.py --all --dev

# 2. Full Dataset Rebuild (All 12,000 images, ~8 minutes on CPU)
python scripts/rebuild_cache.py --all

# 3. Modular Rebuild Commands (Target individual components)
python scripts/rebuild_cache.py --classifier     # Rebuilds ResNet18 features & 5-class Logistic Regression
python scripts/rebuild_cache.py --localization   # Rebuilds 30,000-patch PatchCore memory bank
python scripts/rebuild_cache.py --novelty        # Rebuilds k=5 cosine novelty detector embeddings
python scripts/rebuild_cache.py --robustness     # Rebuilds perturbation suite evaluation on 500 test images
```

### Option B: Named One-Liner Commands via Python
```bash
# Rebuild ResNet18 Feature Cache & Calibrated Classifier:
python -c "from src.quality.classifier import QualityClassifier; QualityClassifier().train_organizer_resnet18(dev_mode=True)"

# Rebuild PatchCore Localization Memory Bank (30,000 normal patches):
python -c "from src.quality.localization import PatchAnomalyDetector; PatchAnomalyDetector().build_or_load_memory_bank()"

# Rebuild Novelty Detector Embeddings (k=5 nearest-neighbor train embeddings):
python -c "from src.quality.novelty import NoveltyDetector; NoveltyDetector().fit(force_rebuild=True)"
```

---

## 7. Running Automated Tests

Run the full pytest suite (**78 automated tests** covering data layer, production flow, economics, Quality AI, calibration, localization, robustness, novelty, root cause with BH correction, evidence panels, and hardening invariants):
```bash
pytest
```
Or run individual test modules:
```bash
pytest tests/test_stage_g.py -v         # Invariant & hardening tests (7/7 pass)
pytest tests/test_stage_f.py -v         # Stage F evidence panel tests (6/6 pass)
pytest tests/test_stage_e.py -v         # Benjamini-Hochberg & bootstrap CIs (10/10 pass)
pytest tests/test_smoke.py -v           # Integration smoke tests (7/7 pass)
```

---

## 8. Compliance & Ground Rules

- **Software-Only:** No live camera feeds, PLCs, robotic sorting, or production machine controls are required or used.
- **Data Provenance:** Any screen utilizing synthetic data displays an unmissable banner. Synthetic data is never misrepresented as organizer data.
- **Empirical Numbers:** Every metric shown on the dashboard is calculated live from active data tables.
- **Statistical Associations:** Root-cause findings are explicitly reported as *statistical associations / likely contributing factors*, never *proven physical causes*.
- **Advisory Guidance:** Recommendations and what-if simulation results are explicitly tagged *simulated / advisory*.
