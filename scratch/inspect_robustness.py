import json

with open("cache/robust_model_eval.json", "r", encoding="utf-8") as f:
    d = json.load(f)

wc = d["baseline_evaluation"]["worst_case"]
print(f"Baseline Worst Case: {wc['perturbation']} | drop_accuracy={wc['drop_accuracy_pct']}% | drop_f1={wc['drop_f1_macro']}")
print("\n--- COMPARISON TABLE ---")
for r in d["comparison_table"]:
    p = r["perturbation"]
    b_acc = r["baseline_accuracy_pct"]
    r_acc = r["robust_accuracy_pct"]
    d_acc = r["delta_accuracy_pct"]
    b_f1 = r["baseline_f1"]
    r_f1 = r["robust_f1"]
    d_f1 = r["delta_f1"]
    b_frr = r["baseline_frr_pct"]
    r_frr = r["robust_frr_pct"]
    b_far = r["baseline_far_pct"]
    r_far = r["robust_far_pct"]
    b_conf = r["baseline_conf"] * 100.0
    r_conf = r["robust_conf"] * 100.0
    stat = r["status"]
    print(f"{p:<15} | Base Acc: {b_acc:>6.2f}% -> Rob Acc: {r_acc:>6.2f}% ({d_acc:>+6.2f}%) | Base F1: {b_f1:.4f} -> Rob F1: {r_f1:.4f} ({d_f1:>+7.4f}) | Base FRR: {b_frr:>5.2f}% -> Rob FRR: {r_frr:>5.2f}% | Base FAR: {b_far:>5.2f}% -> Rob FAR: {r_far:>5.2f}% | Conf: {b_conf:>5.1f}% -> {r_conf:>5.1f}% | {stat}")

print("\nCases Improved:", d["cases_improved"])
print("Cases Worse:", d["cases_worse"])
print("Cases Unchanged:", d["cases_unchanged"])
