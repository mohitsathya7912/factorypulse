# Model Card: FactoryPulse Quality & Anomaly AI System

## 1. Model Details

- **Model Name:** FactoryPulse Industrial Visual Quality AI & Defect Localization
- **Organization / Competition:** NEURAX Hackathon 3.0 (Domain 2: AI in Industry and Automation)
- **Model Version:** Checkpoint 3 (Stages A through G)
- **Model Date:** September 2026
- **Architecture Overview:**
  1. **Feature Extractor:** Frozen pretrained ResNet18 backbone on CPU (`torchvision.models.resnet18(weights=DEFAULT)`), extracting 512-dimensional pen-ultimate feature representations from global average pooling.
  2. **5-Class Classifier:** L2-regularized multinomial Logistic Regression with `class_weight='balanced'` trained via L-BFGS optimizer.
  3. **Confidence Calibration:** Post-hoc Temperature Scaling ($T^* = 0.8468$) fitted strictly on validation set logits using negative log-likelihood minimization via `scipy.optimize.minimize_scalar`.
  4. **Cost-Optimal Decision Gate:** Cost-weighted rejection threshold ($\tau^* = 0.0100$ at assumed 10% defect prevalence) minimizing Bayes risk: $\text{Loss}(\tau) = C_{\text{FA}} \cdot P(\text{Defect} \mid \hat{y} < \tau) + C_{\text{FR}} \cdot P(\text{Good} \mid \hat{y} \ge \tau)$.
  5. **PatchCore Anomaly Localization:** Joint `layer2 + layer3` spatial feature maps average-pooled and projected to 128 dimensions via fixed random orthogonal projection. Compares image patches against a 30,000-patch memory bank sampled from >=300 normal training images using batched CPU `torch.cdist`, smoothed with a 2D Gaussian filter ($\sigma = 4.0$) and upsampled to $256 \times 256$.
  6. **Novelty Detector:** Minimum mean cosine distance across known classes to the $k=5$ nearest training feature embeddings, thresholded at 99.5th percentile of normal validation embeddings ($\tau_{\text{nov}} = 0.1188$).
  7. **Decision Rationale:** 100% deterministic plain-language 'why' constructed directly from computed mathematical quantities (confidence, top-2 class probabilities, anomaly score, novelty score). Strictly **NO LLM**.

---

## 2. Intended Use

- **Primary Intended Use:** High-throughput automated visual inspection, defect category triage, localized anomaly visualization, and statistical root-cause attribution for discrete manufacturing lines.
- **Selective Classification Routing:** Units with maximum calibrated class probability below the decision threshold ($T_{\text{conf}}$) are automatically routed to a Human Inspector Triage Queue.
- **Out-of-Distribution Handling:** Parts with feature novelty exceeding $\tau_{\text{nov}}$ are flagged with a blue badge (`🔵 NOVEL / UNKNOWN`) for specialist metallurgical/process characterization.
- **Out-of-Scope Uses:** Autonomous critical safety shutoff without human oversight; real-time physical robotic actuator control (software-only decision support platform).

---

## 3. Training Data & Data Splits

### 3.1 Dataset Summary
- **Source:** Official organizer image dataset (`data/organizer/train/`).
- **Total Images:** 12,000 PNG files ($256 \times 256$ pixels, RGB).
- **Class Balance:** Perfectly balanced across 5 classes (2,400 images each):
  - `normal` (conforming)
  - `crack` (defective)
  - `hole` (defective)
  - `rust` (defective)
  - `scratch` (defective)
- **Binary Mapping:** `normal` $\rightarrow$ Conforming (`good`); all other classes $\rightarrow$ `defective` (4:1 defect-to-normal ratio in the full pool).

### 3.2 Partitioning & Leakage Prevention
- **Split Strategy:** Stratified 60% Train / 20% Validation / 20% Test partition with fixed random seed (`seed = 42`).
- **Sample Counts:**
  - **Train:** 7,200 images (1,440 per class)
  - **Validation:** 2,400 images (480 per class)
  - **Test (Held-Out):** 2,400 images (480 per class)
- **Leakage Invariant:** Strictly verified zero image overlap:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- **Split Persistence:** Persisted to disk at `cache/split.csv` to ensure identical partitioning across all modules.

---

## 4. Measured Performance Metrics (Held-Out Test Split Only)

All performance metrics below were computed strictly on the held-out test split (never seen during feature scaling, logistic regression training, or temperature fitting).

### 4.1 5-Class Defect Classification
- **5-Class Test Accuracy:** **100.00%** (2,400 / 2,400 correct)
- **5-Class Macro F1:** **1.0000**
- **Per-Class Metrics:**
  - `normal`: Precision 100.00%, Recall 100.00%, F1 1.0000 (Support: 480)
  - `crack`: Precision 100.00%, Recall 100.00%, F1 1.0000 (Support: 480)
  - `hole`: Precision 100.00%, Recall 100.00%, F1 1.0000 (Support: 480)
  - `rust`: Precision 100.00%, Recall 100.00%, F1 1.0000 (Support: 480)
  - `scratch`: Precision 100.00%, Recall 100.00%, F1 1.0000 (Support: 480)

### 4.2 Binary Quality Gating (Good vs. Defective)
- **Binary Precision:** **100.00%**
- **Binary Recall (Sensitivity):** **100.00%**
- **ROC-AUC:** **1.0000**
- **PR-AUC:** **1.0000**
- **False-Reject Rate (FRR):** **0.00%** (0 / 480 normal test images, 95% Wilson CI: `[0.00%, 0.79%]`)
- **False-Accept Rate (FAR):** **0.00%** (0 / 1,920 defect test images, 95% Wilson CI: `[0.00%, 0.20%]`)

### 4.3 Probability Calibration (Temperature Scaling)
- **Optimal Temperature Parameter:** $T^* = 0.8468$
- **Expected Calibration Error (ECE, 10 Bins):**
  - Uncalibrated: **0.0482%**
  - Calibrated: **0.0284%** (41.1% error reduction)
- **Finite and Non-Worse Invariant:** $\text{ECE}_{\text{cal}} \le \text{ECE}_{\text{uncal}}$ confirmed by automated regression test.

### 4.4 PatchCore Anomaly Localization
- **Memory Bank Size:** 30,000 patches from 300 normal training images (`cache/patch_memory_bank.pt`).
- **99th-Percentile Normal Threshold:** $\tau_{\text{anom}} = 2.0605$
- **Image-Level Anomaly ROC-AUC:** **100.00%** (evaluated on 500-image test set: 100 per class).
- **Normal Sanity Check:** **99.0%** of normal test images remain below threshold.
- **Scoring Latency:** **0.1335 seconds per image** on standard CPU (well below 0.5s limit).
- **Output Resolution:** $256 \times 256$ float anomaly map and blended RGB overlay with up to 3 bounding boxes.

### 4.5 Robustness & Data Augmentation
- **Perturbation Benchmark (500 held-out test images):**
  - Worst Case: Gaussian Noise ($\sigma = 25$) dropped accuracy by **-6.60% pts** (to 93.40%).
  - Runner Up: Gaussian Blur ($r = 2$) dropped accuracy by **-3.80% pts** (to 96.20%).
- **Data Augmentation Effect (7,200 augmented train features + recalibration $T^* = 0.7800$):**
  - Gaussian Noise accuracy improved from **93.40% to 99.20%** (+5.80% pts recovery).
  - Gaussian Blur accuracy improved from **96.20% to 99.80%** (+3.60% pts recovery).
  - Clean baseline accuracy maintained at **100.00%**. Zero conditions degraded.

### 4.6 Novelty Detection ($k=5$ Cosine Distance)
- **Held-Out Class Stress Test (Simulated Unseen Classes):**
  - **100.0%** of held-out `scratch` images flagged as Novel / Uncertain.
  - **100.0%** of held-out `rust` images flagged as Novel / Uncertain.
- **False-Novel Rate on Known Classes:** **5.94%** (95% Wilson CI: `[4.97%, 7.09%]`).

---

## 5. Statistical Root-Cause Analysis (Demo Process Telemetry)

- **Method:** Non-parametric Spearman rank correlation ($\rho$) between batch process parameters and batch defect rate across $N = 20$ manufacturing batches.
- **Multiple Testing Correction:** Direct implementation of Benjamini-Hochberg (BH) False Discovery Rate procedure ($q_{(k)} = \min(1, p_{(k)} \cdot m / k)$ with step-down monotonicity).
- **Uncertainty Quantification:** 1,000-iteration bootstrap 95% confidence intervals.
- **Measured Findings ($N = 20$ Batches):**
  - `vibration`: $\rho = +0.623$, 95% CI $[+0.117, +0.886]$, $p = 0.0033$, $p_{\text{adj}} = 0.0072$ (**Significant / Strong**)
  - `speed`: $\rho = +0.619$, 95% CI $[+0.168, +0.864]$, $p = 0.0036$, $p_{\text{adj}} = 0.0072$ (**Significant / Strong**)
  - `pressure`: $\rho = +0.503$, 95% CI $[-0.035, +0.829]$, $p = 0.0239$, $p_{\text{adj}} = 0.0318$ (**Significant / Moderate**)
  - `temperature`: $\rho = +0.450$, 95% CI $[-0.127, +0.749]$, $p = 0.0467$, $p_{\text{adj}} = 0.0467$ (**Significant / Moderate**)
- **Defect Family Breakdown:** `pressure` is strongly associated with `Pinhole Void` ($\rho = +0.673$, $p_{\text{adj}} = 0.0040$); other defect family correlations do not survive FDR correction under $N=20$ and are labeled `not significant`.

---

## 6. Limitations & Ethical Considerations

1. **Synthetic-Looking Data Characteristics:**
   The organizer image dataset exhibits clean background renders and uniform illumination typical of synthetic CAD/procedural generation rather than high-noise real-world optical factory cameras (e.g. lens distortion, oil smudges, particulate debris).
2. **Same-Source Test Split:**
   Both train and test sets originate from the same collection run. While split leakage is strictly zero, cross-factory domain shift (different lighting angles, camera sensors, or surface finishes) is not captured.
3. **Statistical Associations, Not Physical Causes:**
   Root-cause findings represent rank correlations and likely contributing factors derived non-parametrically. They do not constitute proven physical causality. Physical plant interventions should be verified through controlled Design of Experiments (DOE).
4. **Qualitative Localization Disclaimer:**
   Because the competition dataset provides only whole-image class labels and no ground-truth pixel masks, localization heatmaps and bounding boxes are unsupervised nearest-neighbor approximations. Intersection-over-Union (IoU) and mean Average Precision (mAP) cannot be quantitatively reported.
5. **Demo Data for Process & Economics:**
   The organizer dataset includes only visual part images. Production flow rates, station cycle times, sensor telemetry, and economic cost tables reflect the simulated 20-batch demo baseline. FactoryPulse displays an unmissable `DEMO` banner on these tabs and never invents artificial links between telemetry and images.
