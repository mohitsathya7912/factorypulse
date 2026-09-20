import sys, os, time
sys.path.insert(0, os.path.abspath("."))
from src.quality.robustness import train_and_evaluate_robust_pipeline

t0 = time.time()
print("Starting STAGE D Robustness and Augmentation Pipeline...")
results = train_and_evaluate_robust_pipeline(force_recompute=True)
print(f"PIPELINE COMPLETE in {time.time() - t0:.2f} s")
print("Worst-case perturbation for baseline model:", results["baseline_evaluation"]["worst_case"])
print("Cases improved by robust model:", results["cases_improved"])
print("Cases worse with robust model:", results["cases_worse"])
