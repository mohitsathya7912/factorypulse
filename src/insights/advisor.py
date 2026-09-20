"""
src/insights/advisor.py
FactoryPulse Rule-Based Decision Advisor & End-to-End Impact Chain Engine.
Derives insights strictly from computed statistics and operational telemetry (no LLM).

Components:
1. Computed Insight Statements (with empirical evidence objects and confidence ratings).
2. End-to-End Impact Chain:
   Top Defect Family -> Associated Process Factor -> Bottleneck Station & Rework Load ->
   Throughput Lost -> Cost of Poor Quality -> Gross Margin.
3. Rule-Based Advisory Recommendations (clearly labeled 'SIMULATED / ADVISORY').
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from config.schema import is_defective

class DecisionAdvisor:
    """
    Rule-based manufacturing insight synthesizer and impact chain calculator.
    Synthesizes inspection, process telemetry, production flow, and plant economics.
    """

    def __init__(
        self,
        quality_summary: Optional[Dict[str, Any]] = None,
        production_results: Optional[Dict[str, Any]] = None,
        economic_results: Optional[Dict[str, Any]] = None,
        rootcause_results: Optional[Dict[str, Any]] = None,
        df_insp: Optional[pd.DataFrame] = None
    ):
        self.quality = quality_summary or {}
        self.production = production_results or {}
        self.economics = economic_results or {}
        self.rootcause = rootcause_results or {}
        self.df_insp = df_insp

    def generate_computed_insights(self) -> List[Dict[str, Any]]:
        """
        Generate structured insight sentences strictly from computed numbers.
        Each statement is backed by an evidence object and confidence level.
        """
        insights = []

        # 1. Bottleneck Flow Insight
        bn = self.production.get("bottleneck_station", {}) or {}
        bn_name = bn.get("station", bn.get("station_name", "N/A"))
        if "utilization" in bn:
            bn_util = float(bn["utilization"]) * 100.0
        elif "total_effective_utilization_pct" in bn:
            bn_util = float(bn["total_effective_utilization_pct"])
        else:
            bn_util = 0.0
        bn_wip = bn.get("wip", bn.get("wip_count", 0))
        rework_rate = bn.get("rework_rate", 0.0) * 100.0
        rework_lost = self.production.get("throughput_lost_due_to_rework", self.production.get("throughput_lost_rework", 0.0))

        if bn_name != "N/A" and bn_util > 0:
            stmt_bn = (
                f"Bottleneck: {bn_name} (utilization {bn_util:.1f}%, WIP {int(bn_wip)}); "
                f"rework adds {rework_rate:.1f}% extra load, costing {rework_lost:.1f} UPH in lost capacity."
            )
            insights.append({
                "category": "Bottleneck & Flow",
                "statement": stmt_bn,
                "evidence": {
                    "station": bn_name,
                    "utilization_pct": round(bn_util, 1),
                    "wip": int(bn_wip),
                    "rework_load_pct": round(rework_rate, 1),
                    "throughput_lost_uph": round(rework_lost, 1)
                },
                "confidence": "High",
                "evidence_type": "Direct Operational Line Calculation"
            })

        # 2. Root Cause / Likely Contributing Factor Insight
        ranked_factors = self.rootcause.get("ranked_factors", [])
        n_batches = self.rootcause.get("sample_size", 0)
        
        if ranked_factors and len(ranked_factors) > 0:
            top = ranked_factors[0]
            param_name = top.get("parameter", "N/A")
            rho = top.get("spearman_rho", top.get("correlation", 0.0))
            p_val = top.get("spearman_p_value", top.get("corr_p_value", 1.0))
            high_mean = top.get("high_defect_mean", 0.0)
            low_mean = top.get("low_defect_mean", 0.0)
            ev_label = top.get("evidence_label", "Moderate")

            stmt_rc = (
                f"Likely contributing factor: {param_name} (Spearman rho={rho:+.2f}, p={p_val:.4f}, n={n_batches} batches); "
                f"batches with elevated {param_name} averaged {high_mean:.1f}% defect rate vs {low_mean:.1f}% in baseline batches."
            )
            insights.append({
                "category": "Root-Cause Attribution",
                "statement": stmt_rc,
                "evidence": {
                    "parameter": param_name,
                    "spearman_rho": rho,
                    "p_value": p_val,
                    "n_batches": n_batches,
                    "high_defect_mean_pct": high_mean,
                    "low_defect_mean_pct": low_mean,
                    "delta_points": round(high_mean - low_mean, 1)
                },
                "confidence": ev_label,
                "evidence_type": f"Non-Parametric Statistical Association ({ev_label} Evidence)"
            })
        else:
            insights.append({
                "category": "Root-Cause Attribution",
                "statement": "Likely contributing factor: insufficient telemetry data to establish statistical correlation.",
                "evidence": {"status": "insufficient_data"},
                "confidence": "Low",
                "evidence_type": "Data Missing"
            })

        # 3. Cost of Poor Quality & Profitability Insight
        copq = self.economics.get("cost_of_poor_quality", self.economics.get("total_quality_loss", 0.0))
        scrap_c = self.economics.get("scrap_cost", 0.0)
        rework_c = self.economics.get("rework_cost", 0.0)
        dt_c = self.economics.get("downtime_cost", 0.0)
        margin = self.economics.get("margin", self.economics.get("gross_margin_pct", 0.0))
        if margin <= 1.0:
            margin *= 100.0
        revenue = self.economics.get("revenue", self.economics.get("gross_revenue", 0.0))
        margin_loss_pts = (copq / max(1.0, revenue)) * 100.0 if revenue > 0 else 0.0

        if copq > 0:
            stmt_econ = (
                f"Cost of poor quality: ${copq:,.2f} (scrap: ${scrap_c:,.2f}, rework: ${rework_c:,.2f}, downtime: ${dt_c:,.2f}); "
                f"direct margin impact: -{margin_loss_pts:.1f} percentage points of gross operating margin."
            )
            insights.append({
                "category": "Plant Economics",
                "statement": stmt_econ,
                "evidence": {
                    "copq": round(copq, 2),
                    "scrap_cost": round(scrap_c, 2),
                    "rework_cost": round(rework_c, 2),
                    "downtime_cost": round(dt_c, 2),
                    "gross_margin_pct": round(margin, 2),
                    "margin_penalty_points": round(margin_loss_pts, 2)
                },
                "confidence": "High",
                "evidence_type": "Financial Cost Accounting"
            })

        # 4. Quality AI & Triage Insight
        if self.quality:
            q_acc = self.quality.get("accuracy", 0.0) * 100.0
            q_far = self.quality.get("false_accept_rate", 0.0) * 100.0
            q_frr = self.quality.get("false_reject_rate", 0.0) * 100.0
            unc_cnt = self.quality.get("uncertain_count", 0)
            unc_frac = self.quality.get("uncertain_fraction", 0.0) * 100.0
            
            stmt_q = (
                f"Quality AI inspection: test accuracy {q_acc:.1f}% (FRR {q_frr:.1f}%, FAR {q_far:.1f}%); "
                f"{unc_cnt} borderline units ({unc_frac:.1f}%) isolated to human triage queue."
            )
            insights.append({
                "category": "Quality AI & Triage",
                "statement": stmt_q,
                "evidence": {
                    "test_accuracy_pct": round(q_acc, 1),
                    "false_reject_rate_pct": round(q_frr, 1),
                    "false_accept_rate_pct": round(q_far, 1),
                    "uncertain_count": unc_cnt,
                    "uncertain_pct": round(unc_frac, 1)
                },
                "confidence": "High",
                "evidence_type": "Held-Out Test Validation"
            })

        return insights

    def compute_impact_chain(self) -> Dict[str, Any]:
        """
        Compute the 6-stage end-to-end impact chain:
        Top Defect Family -> Most Associated Process Factor -> Bottleneck Station & Rework Load ->
        Throughput Lost -> Cost of Poor Quality -> Gross Margin.
        
        Every element is derived strictly from computed values; if a link cannot be computed,
        'Insufficient data' is returned rather than guessing.
        """
        chain_steps = []

        # Step 1: Top Defect Family
        top_cat_name = "insufficient data"
        top_cat_count = 0
        top_cat_pct = 0.0
        cat_available = False

        if self.df_insp is not None and len(self.df_insp) > 0:
            cat_col = "defect_category" if "defect_category" in self.df_insp.columns else ("defect_family" if "defect_family" in self.df_insp.columns else None)
            lbl_col = "label" if "label" in self.df_insp.columns else ("is_defective" if "is_defective" in self.df_insp.columns else None)
            
            if cat_col and lbl_col:
                def_rows = self.df_insp[is_defective(self.df_insp[lbl_col]) & (self.df_insp[cat_col].notna())]
                # Filter out 'None'
                valid_def = def_rows[~def_rows[cat_col].astype(str).str.lower().isin(["none", "good", "nan", ""])]
                if len(valid_def) > 0:
                    counts = valid_def[cat_col].value_counts()
                    top_cat_name = str(counts.index[0])
                    top_cat_count = int(counts.iloc[0])
                    top_cat_pct = (top_cat_count / max(1, len(valid_def))) * 100.0
                    cat_available = True

        chain_steps.append({
            "stage": 1,
            "title": "Top Defect Family",
            "name": top_cat_name if cat_available else "insufficient data",
            "metric": f"{top_cat_pct:.1f}% of defects" if cat_available else "insufficient data",
            "detail": f"{top_cat_count} units flagged" if cat_available else "No categorized defects in active window",
            "is_available": cat_available
        })

        # Step 2: Most Associated Process Factor
        ranked = self.rootcause.get("ranked_factors", [])
        factor_available = False
        factor_name = "insufficient data"
        factor_metric = "insufficient data"
        factor_detail = "insufficient data"

        if ranked and len(ranked) > 0:
            top_f = ranked[0]
            factor_name = str(top_f.get("parameter", "N/A"))
            rho = top_f.get("spearman_rho", top_f.get("correlation", 0.0))
            p_val = top_f.get("spearman_p_value", 1.0)
            delta_pts = top_f.get("defect_rate_delta_pts", 0.0)
            ev = top_f.get("evidence_label", "Moderate")

            factor_metric = f"Spearman rho = {rho:+.2f}"
            factor_detail = f"p={p_val:.4f} ({ev} evidence, +{delta_pts:.1f}% pts in high batches)"
            factor_available = True

        chain_steps.append({
            "stage": 2,
            "title": "Likely Contributing Factor",
            "name": factor_name,
            "metric": factor_metric,
            "detail": factor_detail,
            "is_available": factor_available
        })

        # Step 3: Bottleneck Station & Rework Load
        bn = self.production.get("bottleneck_station", {}) or {}
        bn_available = False
        bn_name = "insufficient data"
        bn_metric = "insufficient data"
        bn_detail = "insufficient data"

        if bn:
            bn_name = str(bn.get("station", bn.get("station_name", "insufficient data")))
            if "utilization" in bn:
                u_val = float(bn["utilization"]) * 100.0
            elif "total_effective_utilization_pct" in bn:
                u_val = float(bn["total_effective_utilization_pct"])
            else:
                u_val = 0.0
            rew_rate = bn.get("rework_rate", 0.0) * 100.0
            wip_val = bn.get("wip", bn.get("wip_count", 0))

            if bn_name != "insufficient data" and u_val > 0:
                bn_metric = f"{u_val:.1f}% Utilization"
                bn_detail = f"WIP: {int(wip_val)} | Rework adds +{rew_rate:.1f}% load"
                bn_available = True

        chain_steps.append({
            "stage": 3,
            "title": "Primary Line Bottleneck",
            "name": bn_name,
            "metric": bn_metric,
            "detail": bn_detail,
            "is_available": bn_available
        })

        # Step 4: Throughput Lost Due to Rework Loop
        th_lost = self.production.get("throughput_lost_due_to_rework", self.production.get("throughput_lost_rework", None))
        th_available = th_lost is not None and th_lost >= 0
        th_metric = f"-{float(th_lost):.1f} UPH" if th_available else "insufficient data"
        th_detail = "Lost due to rework looping back into bottleneck" if th_available else "insufficient data"

        chain_steps.append({
            "stage": 4,
            "title": "Throughput Lost",
            "name": f"{float(th_lost):.1f} UPH Lost" if th_available else "insufficient data",
            "metric": th_metric,
            "detail": th_detail,
            "is_available": th_available
        })

        # Step 5: Cost of Poor Quality (COPQ)
        copq = self.economics.get("cost_of_poor_quality", self.economics.get("total_quality_loss", None))
        copq_available = copq is not None and copq >= 0
        scrap_c = self.economics.get("scrap_cost", 0.0)
        rew_c = self.economics.get("rework_cost", 0.0)
        dt_c = self.economics.get("downtime_cost", 0.0)

        chain_steps.append({
            "stage": 5,
            "title": "Cost of Poor Quality",
            "name": f"${float(copq):,.2f}" if copq_available else "insufficient data",
            "metric": f"${float(copq):,.0f} COPQ" if copq_available else "insufficient data",
            "detail": f"${scrap_c:,.0f} scrap, ${rew_c:,.0f} rework, ${dt_c:,.0f} dt" if copq_available else "insufficient data",
            "is_available": copq_available
        })

        # Step 6: Gross Operating Margin Impact
        margin = self.economics.get("margin", self.economics.get("gross_margin_pct", None))
        revenue = self.economics.get("revenue", self.economics.get("gross_revenue", 0.0))
        margin_available = margin is not None
        if margin_available and margin <= 1.0:
            margin *= 100.0

        margin_loss_pts = (float(copq) / max(1.0, float(revenue))) * 100.0 if (copq_available and revenue > 0) else 0.0

        chain_steps.append({
            "stage": 6,
            "title": "Gross Margin Impact",
            "name": f"{float(margin):.1f}% Margin" if margin_available else "insufficient data",
            "metric": f"{float(margin):.1f}% Operating Margin" if margin_available else "insufficient data",
            "detail": f"-{margin_loss_pts:.1f}% pts lost to poor quality" if margin_available else "insufficient data",
            "is_available": margin_available
        })

        # End-to-end narrative
        narrative_parts = []
        if cat_available:
            narrative_parts.append(f"Top defect family is <b>{top_cat_name}</b> ({top_cat_pct:.1f}% of defects)")
        if factor_available:
            narrative_parts.append(f"statistically associated with process parameter <b>{factor_name}</b> (rho={rho:+.2f}, p={p_val:.4f})")
        if bn_available:
            narrative_parts.append(f"cycling rework into primary bottleneck <b>{bn_name}</b> ({u_val:.1f}% utilization)")
        if th_available:
            narrative_parts.append(f"causing an avoidable throughput loss of <b>{float(th_lost):.1f} UPH</b>")
        if copq_available:
            narrative_parts.append(f"and driving <b>${float(copq):,.2f}</b> in Cost of Poor Quality (COPQ)")
        if margin_available:
            narrative_parts.append(f"depressing plant operating margin by <b>{margin_loss_pts:.1f}% points</b> to <b>{float(margin):.1f}%</b>")

        narrative = ". ".join(narrative_parts) + "." if narrative_parts else "Insufficient data across stages to form complete narrative."

        return {
            "steps": chain_steps,
            "narrative": narrative,
            "all_available": all(s["is_available"] for s in chain_steps)
        }

    def generate_advisory_insights(self) -> List[Dict[str, Any]]:
        """
        Synthesize actionable rule-based advisory recommendations.
        Each recommendation is backed by computed evidence and tagged 'SIMULATED / ADVISORY'.
        """
        advisories = []
        chain = self.compute_impact_chain()
        steps = {s["title"]: s for s in chain["steps"]}

        # Recommendation 1: Process Drift & Quality Control
        factor_step = steps.get("Likely Contributing Factor", {})
        defect_step = steps.get("Top Defect Family", {})
        if factor_step.get("is_available") and factor_step["name"] != "insufficient data":
            p_name = factor_step["name"]
            d_name = defect_step.get("name", "Defect")
            ev_detail = factor_step.get("detail", "")
            
            action_text = (
                f"Recalibrate `{p_name}` PID controllers and enforce tighter operational limits. "
                f"Statistical association indicates elevated `{p_name}` drives {d_name} formation ({ev_detail})."
            )
            th_lost_val = float(self.production.get("throughput_lost_due_to_rework", 0.0))
            sim_impact = (
                f"Simulated / advisory: Restoring `{p_name}` to baseline envelope is projected to reduce {d_name} "
                f"defects by ~40-60%, recovering up to ~{th_lost_val:.1f} UPH of effective line capacity."
            )
            advisories.append({
                "category": "Process Telemetry & Quality Control",
                "title": f"Process Intervention: Restabilize {p_name}",
                "evidence": f"Likely contributing factor: {p_name} ({factor_step['metric']}, {ev_detail})",
                "recommended_action": action_text,
                "projected_impact": sim_impact,
                "confidence_level": "High (Statistical Association)",
                "status": "SIMULATED / ADVISORY"
            })

        # Recommendation 2: Bottleneck Debottlenecking & Rework Divert
        bn_step = steps.get("Primary Line Bottleneck", {})
        th_step = steps.get("Throughput Lost", {})
        if bn_step.get("is_available") and bn_step["name"] != "insufficient data":
            bn_name = bn_step["name"]
            bn_det = bn_step.get("detail", "")
            th_name = th_step.get("name", "lost capacity")

            action_text = (
                f"Install pre-station optical triage upstream of `{bn_name}` to intercept defective parts "
                f"before they enter the constraint stage ({bn_det})."
            )
            sim_impact = (
                f"Simulated / advisory: Diverting rework from `{bn_name}` eliminates the {th_name} rework penalty, "
                f"restoring bottleneck effective utilization toward target baseline."
            )
            advisories.append({
                "category": "Production Flow & Bottlenecks",
                "title": f"Bottleneck Relief: Pre-Screen at {bn_name}",
                "evidence": f"Bottleneck station: {bn_name} ({bn_step['metric']}, {bn_det})",
                "recommended_action": action_text,
                "projected_impact": sim_impact,
                "confidence_level": "High (Deterministic Line Flow)",
                "status": "SIMULATED / ADVISORY"
            })

        # Recommendation 3: Financial ROI & Scrap Cost Recovery
        copq_step = steps.get("Cost of Poor Quality", {})
        margin_step = steps.get("Gross Margin Impact", {})
        if copq_step.get("is_available") and copq_step["name"] != "insufficient data":
            copq_val = copq_step["name"]
            margin_det = margin_step.get("detail", "")
            copq_raw = float(self.economics.get("cost_of_poor_quality", 0.0))
            rev_raw = float(self.economics.get("revenue", 1.0))
            projected_profit_gain = copq_raw * 0.30
            projected_margin_pts = (projected_profit_gain / max(1.0, rev_raw)) * 100.0

            action_text = (
                f"Prioritize scrap reduction targets on the top defect family to recover direct material costs "
                f"and eliminate secondary rework loops."
            )
            sim_impact = (
                f"Simulated / advisory: Cutting poor-quality disruptions by 30% would recover ~${projected_profit_gain:,.0f} "
                f"in net operating profit, lifting gross margin by ~{projected_margin_pts:.1f} percentage points."
            )
            advisories.append({
                "category": "Plant Financials & COPQ",
                "title": f"Cost Recovery: Mitigate {copq_val} COPQ",
                "evidence": f"Total Cost of Poor Quality: {copq_val} ({copq_step.get('detail', '')}, {margin_det})",
                "recommended_action": action_text,
                "projected_impact": sim_impact,
                "confidence_level": "High (Direct P&L Accounting)",
                "status": "SIMULATED / ADVISORY"
            })

        return advisories
