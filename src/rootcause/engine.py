"""
src/rootcause/engine.py
Statistical Root-Cause Engine and Multivariate Process Drift Attribution.
Focuses on simple, honest, non-parametric statistics (Spearman rank correlation,
Mann-Whitney U high-vs-low median split, and 2-sigma control limits).

Strictly adheres to:
- Output terminology: 'likely contributing factor' / 'statistical association' (never 'proven cause').
- No black-box SHAP/XGBoost claims in this phase.
- Demo mode notice: 'Relationships in synthetic data are built into the generator; this demonstrates the analysis pipeline only.'
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from config.schema import is_defective


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """
    Direct implementation of the Benjamini-Hochberg False Discovery Rate (FDR) procedure.
    
    Given m hypothesis tests with raw p-values:
    1. Sort p-values in ascending order: p_(1) <= p_(2) <= ... <= p_(m).
    2. Compute adjusted p-values: q_(k) = min(1.0, p_(k) * m / k).
    3. Enforce step-down monotonicity: q_(k) = min(q_(k), q_(k+1)) from k = m-1 down to 1.
    4. Restore original input ordering.
    
    Args:
        p_values: List or array of raw p-values
        
    Returns:
        List of adjusted p-values (q-values) in original order.
    """
    m = len(p_values)
    if m == 0:
        return []
    if m == 1:
        return [float(p_values[0])]

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    for rank_idx, (orig_idx, p_val) in enumerate(indexed, 1):
        q = min(1.0, float(p_val) * m / rank_idx)
        adjusted[rank_idx - 1] = q

    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    result = [0.0] * m
    for (orig_idx, _), q_val in zip(indexed, adjusted):
        result[orig_idx] = float(q_val)

    return result


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Compute a bootstrap confidence interval for the Spearman rank correlation coefficient.
    
    Args:
        x: First numeric array
        y: Second numeric array
        n_boot: Number of bootstrap iterations (default 1000)
        ci: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    n = len(x)
    if n < 3:
        return -1.0, 1.0

    rng = np.random.RandomState(seed)
    boot_rhos = []

    alpha = (1.0 - ci) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        x_sample = x[idx]
        y_sample = y[idx]

        if np.std(x_sample) > 0 and np.std(y_sample) > 0:
            res = stats.spearmanr(x_sample, y_sample)
            stat = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
            if not np.isnan(stat):
                boot_rhos.append(stat)

    if not boot_rhos:
        point = stats.spearmanr(x, y)[0]
        return float(point), float(point)

    ci_lower = float(np.percentile(boot_rhos, lower_pct))
    ci_upper = float(np.percentile(boot_rhos, upper_pct))

    ci_lower = max(-1.0, min(1.0, ci_lower))
    ci_upper = max(-1.0, min(1.0, ci_upper))

    return round(ci_lower, 3), round(ci_upper, 3)


def build_batch_table(
    df_insp: pd.DataFrame,
    df_proc: pd.DataFrame,
    df_prod: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Construct a consolidated batch-level dataset joining:
    1. Inspection defect rates (overall, per defect category, and variant distribution).
    2. Mean process telemetry parameters.
    3. Production station summaries (optional).
    
    Parameters:
        df_insp: Inspection DataFrame with unit_id, batch_id, label, defect_category.
        df_proc: Process telemetry DataFrame with batch_id, temperature, speed, pressure, etc.
        df_prod: Optional production flow DataFrame.
        
    Returns:
        Consolidated DataFrame indexed by batch_id.
    """
    if len(df_insp) == 0:
        return pd.DataFrame()

    batch_col = "batch_id" if ("batch_id" in df_insp.columns and df_insp["batch_id"].dropna().nunique() > 0) else None
    label_col = "label" if "label" in df_insp.columns else ("is_defective" if "is_defective" in df_insp.columns else None)
    cat_col = "defect_category" if "defect_category" in df_insp.columns else ("defect_family" if "defect_family" in df_insp.columns else None)

    if not batch_col or not label_col:
        return pd.DataFrame()

    # 1. Aggregate Inspection per batch
    records = []
    unique_categories = []
    if cat_col:
        raw_cats = df_insp[cat_col].dropna().astype(str).str.strip().unique()
        unique_categories = [c for c in raw_cats if c.lower() not in ["none", "good", "nan", ""]]

    for b_id, group in df_insp.groupby(batch_col):
        n_total = len(group)
        n_defects = int(is_defective(group[label_col]).sum())
        defect_rate = float(n_defects / max(1, n_total))

        row: Dict[str, Any] = {
            "batch_id": str(b_id),
            "total_inspected": n_total,
            "defect_count": n_defects,
            "defect_rate": defect_rate,
            "conforming_count": max(0, n_total - n_defects),
        }

        # Defect rates per category
        if cat_col and unique_categories:
            for cat in unique_categories:
                cat_cnt = int((group[cat_col].astype(str) == cat).sum())
                row[f"rate_{cat}"] = float(cat_cnt / max(1, n_total))
                row[f"count_{cat}"] = cat_cnt

        # Dominant variant in batch
        var_col = "variant" if "variant" in group.columns else ("variant_id" if "variant_id" in group.columns else None)
        if var_col:
            mode_var = group[var_col].mode()
            row["dominant_variant"] = str(mode_var.iloc[0]) if len(mode_var) > 0 else "Unknown"

        records.append(row)

    if not records:
        return pd.DataFrame()

    batch_df = pd.DataFrame(records)

    # 2. Join Process Telemetry
    if len(df_proc) > 0 and "batch_id" in df_proc.columns:
        num_cols = [c for c in df_proc.columns if c not in ["batch_id", "timestamp"] and pd.api.types.is_numeric_dtype(df_proc[c])]
        if num_cols:
            proc_means = df_proc.groupby("batch_id")[num_cols].mean().reset_index()
            proc_means["batch_id"] = proc_means["batch_id"].astype(str)
            batch_df = batch_df.merge(proc_means, on="batch_id", how="left")

    # 3. Join Production summaries if present
    if df_prod is not None and len(df_prod) > 0 and "batch_id" in df_prod.columns:
        prod_num_cols = [c for c in ["wip", "downtime_hours", "rework_units", "scrap_units"] if c in df_prod.columns]
        if prod_num_cols:
            prod_agg = df_prod.groupby("batch_id")[prod_num_cols].sum().reset_index()
            prod_agg["batch_id"] = prod_agg["batch_id"].astype(str)
            batch_df = batch_df.merge(prod_agg, on="batch_id", how="left")

    return batch_df


class RootCauseEngine:
    """
    Honest, non-parametric root-cause statistical analysis engine.
    Calculates Spearman rank correlations, high-vs-low median differences with Mann-Whitney U test,
    and 2-sigma outlier process drift.
    """

    def __init__(self, joined_batch_df: pd.DataFrame, is_synthetic: bool = True):
        self.df = joined_batch_df.copy()
        self.is_synthetic = is_synthetic
        
        # Identify numeric process parameters
        meta_cols = {
            "batch_id", "timestamp", "unit_id", "total_inspected", "defect_count",
            "defect_rate", "conforming_count", "dominant_variant", "variant", "variant_id",
            "wip", "downtime_hours", "rework_units", "scrap_units"
        }
        self.param_cols = [
            col for col in self.df.columns
            if col not in meta_cols
            and not col.startswith("rate_")
            and not col.startswith("count_")
            and pd.api.types.is_numeric_dtype(self.df[col])
        ]

    def analyze_factors(self) -> Dict[str, Any]:
        """
        Perform complete non-parametric statistical association and drift analysis.
        
        Returns:
            Dictionary containing ranked contributing factors, evidence levels,
            p-values, 2-sigma drift batches, and honest advisory disclaimers.
        """
        n_batches = len(self.df)
        if n_batches < 3 or len(self.param_cols) == 0:
            return {
                "status": "insufficient_batches",
                "sample_size": n_batches,
                "warning": "At least 3 batches required to compute statistical associations.",
                "ranked_factors": [],
                "disclaimer": "Root-cause outputs are statistical associations / likely contributing factors, not proven causes."
            }

        df = self.df.copy()
        defect_rates = df["defect_rate"].values

        ranked_factors = []
        
        # Small sample size warning
        sample_warning = None
        if n_batches < 10:
            sample_warning = f"⚠️ Small sample size (N = {n_batches} batches). Statistical associations are preliminary."
        if n_batches < 5:
            sample_warning = f"⚠️ Very small sample size (N = {n_batches} batches). Correlation tests are severely underpowered."

        # Detect category-specific rate columns
        cat_rate_cols = [c for c in df.columns if c.startswith("rate_")]

        for col in self.param_cols:
            vals = df[col].dropna().values
            if len(vals) < 3:
                continue
            
            val_std = float(np.std(vals))
            val_mean = float(np.mean(vals))
            
            if val_std == 0.0:
                # Constant parameter
                continue

            # 1. Spearman Rank Correlation (rho and p-value) + Bootstrap 95% CI
            res_spearman = stats.spearmanr(vals, defect_rates)
            rho = float(res_spearman.statistic) if not np.isnan(res_spearman.statistic) else 0.0
            p_spearman = float(res_spearman.pvalue) if not np.isnan(res_spearman.pvalue) else 1.0
            ci_low, ci_high = bootstrap_spearman_ci(vals, defect_rates, n_boot=1000, seed=42)

            # 2. High-vs-Low Median Split & Mann-Whitney U test
            median_param = float(np.median(vals))
            high_mask = vals > median_param
            low_mask = vals <= median_param

            high_rates = defect_rates[high_mask]
            low_rates = defect_rates[low_mask]

            high_mean_rate = float(np.mean(high_rates)) if len(high_rates) > 0 else float(np.mean(defect_rates))
            low_mean_rate = float(np.mean(low_rates)) if len(low_rates) > 0 else float(np.mean(defect_rates))
            defect_delta = high_mean_rate - low_mean_rate # percentage points delta

            # Mann-Whitney U test (non-parametric comparison of defect rates)
            p_mwu = 1.0
            if len(high_rates) >= 2 and len(low_rates) >= 2:
                try:
                    if np.all(high_rates == high_rates[0]) and np.all(low_rates == low_rates[0]) and high_rates[0] == low_rates[0]:
                        p_mwu = 1.0
                    else:
                        mwu_res = stats.mannwhitneyu(high_rates, low_rates, alternative="two-sided")
                        p_mwu = float(mwu_res.pvalue) if not np.isnan(mwu_res.pvalue) else 1.0
                except Exception:
                    p_mwu = 1.0

            # 3. 2-Sigma Affected Outlier Batches
            upper_2s = val_mean + 2.0 * val_std
            lower_2s = val_mean - 2.0 * val_std
            
            outlier_mask = (vals > upper_2s) | (vals < lower_2s)
            affected_batch_indices = np.where(outlier_mask)[0]
            affected_batches = df.iloc[affected_batch_indices]["batch_id"].astype(str).tolist()
            affected_defect_rate = float(np.mean(defect_rates[outlier_mask])) if np.sum(outlier_mask) > 0 else 0.0

            # 4. Evidence Labeling Rules (preliminary raw)
            # Strong: |rho| >= 0.60 and p_spearman < 0.05 (or |defect_delta| >= 0.08)
            # Moderate: |rho| >= 0.35 and p_spearman < 0.15 (or |defect_delta| >= 0.04)
            # Weak: otherwise
            abs_rho = abs(rho)
            abs_delta = abs(defect_delta)

            if (abs_rho >= 0.60 and p_spearman < 0.05) or (abs_delta >= 0.08 and p_mwu < 0.10):
                evidence_label = "Strong"
            elif (abs_rho >= 0.35 and p_spearman < 0.15) or (abs_delta >= 0.04):
                evidence_label = "Moderate"
            else:
                evidence_label = "Weak"

            # 5. Defect family associations
            cat_corrs: Dict[str, float] = {}
            for c_col in cat_rate_cols:
                cat_name = c_col.replace("rate_", "")
                c_rates = df[c_col].values
                if np.std(c_rates) > 0:
                    r_cat = stats.spearmanr(vals, c_rates)
                    cat_corrs[cat_name] = round(float(r_cat.statistic), 3) if not np.isnan(r_cat.statistic) else 0.0

            # Combined effect size metric for ranking: |rho| * 0.7 + (|defect_delta| * 3.0)
            effect_score = abs_rho * 0.7 + min(1.0, abs_delta * 3.0) * 0.3

            factor_record = {
                "parameter": col,
                "spearman_rho": round(rho, 3),
                "correlation": round(rho, 3), # Backward compatibility with test_pipeline.py
                "shap_importance": 0.0,        # Backward compatibility with test_pipeline.py
                "spearman_p_value": round(p_spearman, 4),
                "bootstrap_ci": [ci_low, ci_high],
                "ci_str": f"[{ci_low:+.3f}, {ci_high:+.3f}]",
                "high_defect_mean": round(high_mean_rate * 100.0, 2), # %
                "low_defect_mean": round(low_mean_rate * 100.0, 2),   # %
                "defect_rate_delta_pts": round(defect_delta * 100.0, 2), # percentage points
                "mwu_p_value": round(p_mwu, 4),
                "evidence_label": evidence_label,
                "raw_evidence_label": evidence_label,
                "effect_score": round(effect_score, 4),
                
                # Outlier & Drift Profile
                "mean": round(val_mean, 2),
                "std": round(val_std, 2),
                "upper_limit_2sigma": round(upper_2s, 2),
                "lower_limit_2sigma": round(lower_2s, 2),
                "affected_batches_count": len(affected_batches),
                "affected_batches": affected_batches,
                "affected_defect_rate": round(affected_defect_rate * 100.0, 2), # %
                
                # Defect category correlations
                "category_associations": cat_corrs
            }
            ranked_factors.append(factor_record)

        # Apply Benjamini-Hochberg FDR correction across parameters
        if ranked_factors:
            raw_p_list = [f["spearman_p_value"] for f in ranked_factors]
            adj_p_list = benjamini_hochberg(raw_p_list)
            for f, q_val in zip(ranked_factors, adj_p_list):
                f["spearman_p_adjusted"] = round(q_val, 4)
                f["is_significant"] = bool(q_val < 0.05)
                f["significance_status"] = "Significant" if q_val < 0.05 else "not significant"
                # Ground rule 1: Mark any association that does not survive correction as "not significant"
                if not f["is_significant"]:
                    f["evidence_label"] = "not significant"

        # Sort ranked factors by effect_score descending
        ranked_factors.sort(key=lambda x: x["effect_score"], reverse=True)

        disclaimer = "All findings are statistical associations / likely contributing factors, not proven causes."
        demo_note = (
            "Relationships in synthetic data are built into the generator; this demonstrates the analysis pipeline only."
            if self.is_synthetic else "Evaluated on active manufacturing dataset."
        )

        return {
            "status": "success",
            "sample_size": n_batches,
            "sample_warning": sample_warning,
            "ranked_factors": ranked_factors,
            "disclaimer": disclaimer,
            "demo_note": demo_note,
            "top_factor": ranked_factors[0] if ranked_factors else None,
            "batch_table": df
        }

    def analyze_defect_families(self) -> Dict[str, Any]:
        """
        Compute process parameter Spearman rank correlations per defect family/category.
        Applies Benjamini-Hochberg FDR correction across parameters for each family,
        computes bootstrap 95% CIs, and identifies the top association per family.
        
        Returns:
            Dictionary with 'family_associations' and 'top_associations'.
        """
        df = self.df.copy()
        n_batches = len(df)
        cat_rate_cols = [c for c in df.columns if c.startswith("rate_")]

        family_results: Dict[str, List[Dict[str, Any]]] = {}
        top_associations: List[Dict[str, Any]] = []

        for c_col in cat_rate_cols:
            fam_name = c_col.replace("rate_", "")
            rates = df[c_col].values

            if np.std(rates) == 0.0:
                continue

            raw_items = []
            for col in self.param_cols:
                vals = df[col].dropna().values
                if len(vals) < 3 or np.std(vals) == 0:
                    continue
                res = stats.spearmanr(vals, rates)
                rho = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
                pval = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
                if np.isnan(rho):
                    rho, pval = 0.0, 1.0
                ci_low, ci_high = bootstrap_spearman_ci(vals, rates, n_boot=1000, seed=42)
                raw_items.append({
                    "parameter": col,
                    "rho": round(rho, 3),
                    "raw_p": round(pval, 4),
                    "ci_95": [ci_low, ci_high],
                    "ci_str": f"[{ci_low:+.3f}, {ci_high:+.3f}]"
                })

            if not raw_items:
                continue

            # Apply Benjamini-Hochberg across parameters for this family
            raw_p_list = [item["raw_p"] for item in raw_items]
            adj_p_list = benjamini_hochberg(raw_p_list)

            for item, q_val in zip(raw_items, adj_p_list):
                item["adj_p"] = round(q_val, 4)
                item["is_significant"] = bool(q_val < 0.05)
                item["significance_status"] = "Significant" if q_val < 0.05 else "not significant"
                abs_rho = abs(item["rho"])
                if item["is_significant"] and abs_rho >= 0.60:
                    item["evidence_label"] = "Strong"
                elif item["is_significant"] and abs_rho >= 0.35:
                    item["evidence_label"] = "Moderate"
                elif abs_rho >= 0.35 and item["raw_p"] < 0.15:
                    item["evidence_label"] = "Moderate (Uncorrected only)"
                else:
                    item["evidence_label"] = "Weak / not significant"

            # Sort by |rho| descending
            raw_items.sort(key=lambda x: abs(x["rho"]), reverse=True)
            family_results[fam_name] = raw_items

            # Top association for this family
            top_param = raw_items[0]
            top_associations.append({
                "defect_family": fam_name,
                "top_parameter": top_param["parameter"],
                "spearman_rho": top_param["rho"],
                "bootstrap_ci": top_param["ci_95"],
                "ci_str": top_param["ci_str"],
                "raw_p_value": top_param["raw_p"],
                "adjusted_p_value": top_param["adj_p"],
                "is_significant": top_param["is_significant"],
                "significance_status": top_param["significance_status"],
                "evidence_label": top_param["evidence_label"]
            })

        return {
            "family_associations": family_results,
            "top_associations": top_associations,
            "sample_size": n_batches
        }
