"""
11_shap_explainability.py — SHAP Explainability on XGBoost

Pattern: Explainable AI (XAI) / Model Interpretability
Key Concepts:
  - SHAP (SHapley Additive exPlanations) — global + local explanations
  - Summary plots: bar (global importance) + beeswarm (feature distribution)
  - Waterfall plot: explain a single prediction step-by-step
  - Force plot: visualise positive/negative feature contributions
  - Dependence plot: how one feature affects predictions across all samples
  - Feature importance drift: compare SHAP values across two time windows
  - Enterprise context: regulatory compliance, audit trails, stakeholder trust

Why XAI matters in production:
  - Regulators (GDPR Article 22, EU AI Act) require explainability for
    automated decisions affecting individuals
  - Business owners accept AI decisions more readily when they see *why*
  - SHAP surfaces data quality issues and feature leakage that accuracy
    metrics alone won't catch
  - Post-deployment SHAP drift signals model behaviour changes before
    accuracy metrics degrade
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from dataclasses import dataclass, field
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

RANDOM_SEED      = 42
OUTPUT_DIR       = "shap_outputs"
SHAP_DRIFT_ALERT = 0.15   # flag feature if mean |SHAP| changes > 15%

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_churn_dataset(n_samples: int = 4000) -> tuple[pd.DataFrame, list[str]]:
    """
    Synthetic churn-prediction dataset with named features.
    Mirrors the kind of tabular data common in telecom/fintech churn models.
    """
    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=8,
        n_redundant=3,
        n_classes=2,
        weights=[0.72, 0.28],
        random_state=RANDOM_SEED,
    )

    feature_names = [
        "days_since_login",       # recency signal
        "monthly_spend",          # value signal
        "support_tickets_90d",    # friction signal
        "plan_tier",              # product signal (0=basic, 1=pro, 2=enterprise)
        "contract_months_left",   # commitment signal
        "nps_score",              # satisfaction signal
        "feature_adoption_pct",   # engagement signal
        "billing_failures_6m",    # risk signal
        "referrals_made",         # loyalty signal
        "mobile_app_sessions",    # activity signal
        "email_open_rate",        # marketing engagement
        "api_calls_monthly",      # technical usage
        "login_failures_30d",     # friction signal
        "upsell_offers_declined", # intent signal
        "tenure_months",          # loyalty signal
    ]

    df = pd.DataFrame(X_raw, columns=feature_names)
    df["churn"] = y
    return df, feature_names


# ── Model Training ────────────────────────────────────────────────────────────

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


# ── SHAP Analysis ─────────────────────────────────────────────────────────────

@dataclass
class SHAPReport:
    feature_names:      list[str]
    mean_abs_shap:      dict[str, float]   # global importance: feature → mean |SHAP|
    top_features:       list[str]          # top 10 by global importance
    sample_explanation: dict               # local explanation for one prediction
    plots_saved:        list[str] = field(default_factory=list)


def compute_shap_values(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[shap.Explainer, np.ndarray]:
    """
    Compute SHAP values using TreeExplainer — exact, fast for tree models.
    Returns the explainer and the raw SHAP value matrix (n_samples × n_features).
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, XGBoost TreeExplainer returns a 2D array
    # of shape (n_samples, n_features) — values for the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # positive class

    return explainer, shap_values


# ── Plot: Global Feature Importance (Bar) ────────────────────────────────────

def plot_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    title: str = "Global Feature Importance — Mean |SHAP|",
) -> str:
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if i < 3 else "#1f77b4" for i in range(len(sorted_idx))]
    ax.barh(
        [feature_names[i] for i in reversed(sorted_idx)],
        mean_abs[sorted_idx[::-1]],
        color=colors[::-1],
    )
    ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)")
    ax.set_title(title)
    ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "shap_global_importance.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot: Beeswarm Summary ────────────────────────────────────────────────────

def plot_beeswarm_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
) -> str:
    """
    Beeswarm plot shows:
      - Y-axis: features ranked by global importance
      - X-axis: SHAP value (positive = pushes toward churn)
      - Colour: actual feature value (red = high, blue = low)
    This is the richest single SHAP visualisation.
    """
    X_df = pd.DataFrame(X, columns=feature_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_df,
        plot_type="dot",
        show=False,
        max_display=12,
    )
    plt.title("SHAP Beeswarm — Feature Impact Distribution", pad=12)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "shap_beeswarm.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {path}")
    return path


# ── Plot: Waterfall — Single Prediction Explanation ──────────────────────────

def plot_waterfall_single(
    explainer: shap.Explainer,
    X_sample: np.ndarray,
    feature_names: list[str],
    sample_idx: int = 0,
) -> str:
    """
    Waterfall plot for one prediction:
      - Starts from E[f(x)] (base value = average model output)
      - Each bar shows how one feature pushed the prediction up or down
      - Ends at f(x) (final prediction for this sample)

    This is what you show a regulator or business stakeholder to explain
    why a specific customer was flagged for churn.
    """
    shap_expl = explainer(X_sample)

    # Handle list output from some XGBoost + SHAP versions
    if isinstance(shap_expl, list):
        shap_expl = shap_expl[1]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_expl[sample_idx], max_display=12, show=False)
    plt.title(f"SHAP Waterfall — Sample {sample_idx} (Individual Prediction Explanation)", pad=12)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"shap_waterfall_sample{sample_idx}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {path}")
    return path


# ── Plot: Dependence — One Feature vs Model Output ────────────────────────────

def plot_dependence(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    feature: str,
) -> str:
    """
    Dependence plot for one feature:
      - X-axis: actual feature value
      - Y-axis: SHAP value (impact on churn prediction)
      - Colour: automatic interaction feature (most correlated)
    Shows non-linear relationships and interactions.
    """
    X_df     = pd.DataFrame(X, columns=feature_names)
    feat_idx = feature_names.index(feature)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X_df,
        ax=ax,
        show=False,
    )
    ax.set_title(f"SHAP Dependence — {feature}", pad=10)
    plt.tight_layout()

    safe_name = feature.replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"shap_dependence_{safe_name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── SHAP Drift Detection ──────────────────────────────────────────────────────

def detect_shap_drift(
    shap_t0: np.ndarray,   # SHAP values from earlier window (e.g. last month)
    shap_t1: np.ndarray,   # SHAP values from current window
    feature_names: list[str],
    threshold: float = SHAP_DRIFT_ALERT,
) -> dict:
    """
    Compare mean |SHAP| across two time windows.
    A large change in a feature's SHAP importance signals:
      - Data distribution shift (input drift)
      - Concept drift (relationship between feature and target changed)
      - Data quality issue (sudden change in feature values)

    Returns a drift report with flagged features.
    """
    mean_t0 = np.abs(shap_t0).mean(axis=0)
    mean_t1 = np.abs(shap_t1).mean(axis=0)

    report = {"flagged": [], "stable": [], "threshold": threshold}

    for i, fname in enumerate(feature_names):
        base = mean_t0[i]
        curr = mean_t1[i]
        change = abs(curr - base) / (base + 1e-9)

        entry = {
            "feature":    fname,
            "shap_t0":    round(float(base), 4),
            "shap_t1":    round(float(curr), 4),
            "pct_change": round(change * 100, 1),
            "drifted":    change > threshold,
        }

        if entry["drifted"]:
            report["flagged"].append(entry)
        else:
            report["stable"].append(entry)

    report["flagged"].sort(key=lambda x: x["pct_change"], reverse=True)
    return report


def plot_shap_drift(drift_report: dict, feature_names: list[str]) -> str:
    flagged = drift_report["flagged"]
    stable  = drift_report["stable"]
    all_features = flagged + stable

    names   = [e["feature"]    for e in all_features]
    t0_vals = [e["shap_t0"]    for e in all_features]
    t1_vals = [e["shap_t1"]    for e in all_features]
    colors  = ["#d62728" if e["drifted"] else "#1f77b4" for e in all_features]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, t0_vals, width, label="Window T0 (baseline)", color="#aec7e8", alpha=0.9)
    ax.bar(x + width/2, t1_vals, width, label="Window T1 (current)",  color=colors,   alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Mean |SHAP Value|")
    ax.set_title(f"SHAP Feature Importance Drift  (threshold={drift_report['threshold']*100:.0f}%)\n"
                 f"Red = drifted ({len(flagged)} features flagged)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "shap_drift_comparison.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Build SHAP Report ─────────────────────────────────────────────────────────

def build_shap_report(
    explainer:     shap.Explainer,
    shap_values:   np.ndarray,
    X_test:        np.ndarray,
    feature_names: list[str],
) -> SHAPReport:
    mean_abs = {
        fname: round(float(np.abs(shap_values[:, i]).mean()), 4)
        for i, fname in enumerate(feature_names)
    }
    top_features = sorted(mean_abs, key=mean_abs.get, reverse=True)[:10]

    # Local explanation for sample 0
    shap_row   = shap_values[0]
    base_value = float(explainer.expected_value
                       if not isinstance(explainer.expected_value, np.ndarray)
                       else explainer.expected_value[1])

    sample_explanation = {
        "sample_index":  0,
        "base_value":    round(base_value, 4),
        "prediction":    round(float(base_value + shap_row.sum()), 4),
        "top_drivers":   sorted(
            [{"feature": feature_names[i], "shap": round(float(shap_row[i]), 4)}
             for i in range(len(feature_names))],
            key=lambda x: abs(x["shap"]),
            reverse=True,
        )[:5],
    }

    return SHAPReport(
        feature_names=feature_names,
        mean_abs_shap=mean_abs,
        top_features=top_features,
        sample_explanation=sample_explanation,
    )


# ── Report Printer ────────────────────────────────────────────────────────────

def print_report(report: SHAPReport, drift_report: dict, metrics: dict):
    print("\n" + "=" * 70)
    print("  SHAP EXPLAINABILITY REPORT")
    print("=" * 70)
    print(f"  Model AUC : {metrics['auc']:.4f}   F1 : {metrics['f1']:.4f}")
    print()
    print("  Global Feature Importance (Top 10 by Mean |SHAP|):")
    for i, fname in enumerate(report.top_features, 1):
        bar = "█" * int(report.mean_abs_shap[fname] * 200)
        print(f"    {i:>2}. {fname:<28} {report.mean_abs_shap[fname]:.4f}  {bar}")

    ex = report.sample_explanation
    print(f"\n  Local Explanation — Sample 0")
    print(f"    Base value (avg prediction) : {ex['base_value']}")
    print(f"    Final prediction            : {ex['prediction']}")
    print(f"    Top 5 drivers:")
    for d in ex["top_drivers"]:
        direction = "▲ churn" if d["shap"] > 0 else "▼ retain"
        print(f"      {d['feature']:<28} SHAP={d['shap']:+.4f}  {direction}")

    print(f"\n  SHAP Drift Report  (threshold={drift_report['threshold']*100:.0f}%)")
    if drift_report["flagged"]:
        print(f"    ⚠ {len(drift_report['flagged'])} feature(s) drifted:")
        for f in drift_report["flagged"][:5]:
            print(f"      {f['feature']:<28} T0={f['shap_t0']:.4f} → T1={f['shap_t1']:.4f}  "
                  f"Δ={f['pct_change']}%")
    else:
        print("    ✓ No drift detected — feature importance stable")

    print()
    print(f"  Plots saved to: {OUTPUT_DIR}/")
    for p in report.plots_saved:
        print(f"    • {os.path.basename(p)}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SHAP Explainability — XGBoost Churn Prediction")
    print("  Pattern : Global + Local XAI for regulatory compliance")
    print("=" * 70)

    # ── Data & Model ──────────────────────────────────────────────────────────
    print("\n  Building dataset and training model...")
    df, feature_names = build_churn_dataset(n_samples=4000)
    X = df[feature_names].values
    y = df["churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "f1":  f1_score(y_test, y_pred),
    }
    print(f"  Model trained — AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}")

    # ── SHAP Values ───────────────────────────────────────────────────────────
    print("\n  Computing SHAP values...")
    explainer, shap_values = compute_shap_values(model, X_test, feature_names)
    print(f"  SHAP matrix shape: {shap_values.shape}  "
          f"({shap_values.shape[0]} samples × {shap_values.shape[1]} features)")

    # ── Build Report ──────────────────────────────────────────────────────────
    report = build_shap_report(explainer, shap_values, X_test, feature_names)

    # ── Generate Plots ────────────────────────────────────────────────────────
    print("\n  Generating SHAP plots...")
    paths = []
    paths.append(plot_global_importance(shap_values, feature_names))
    paths.append(plot_beeswarm_summary(shap_values, X_test, feature_names))
    paths.append(plot_waterfall_single(explainer, X_test[:20], feature_names, sample_idx=0))
    paths.append(plot_dependence(shap_values, X_test, feature_names, "days_since_login"))
    paths.append(plot_dependence(shap_values, X_test, feature_names, "monthly_spend"))
    report.plots_saved = paths

    # ── SHAP Drift Simulation ─────────────────────────────────────────────────
    # Simulate T1 window: inject artificial drift into 3 features
    print("\n  Simulating SHAP drift between two time windows...")
    np.random.seed(99)
    X_t1 = X_test.copy()

    # days_since_login: double the values (users logging in less — drift signal)
    X_t1[:, feature_names.index("days_since_login")] *= np.random.uniform(1.6, 2.2,
                                                                            size=X_t1.shape[0])
    # billing_failures_6m: sharp increase (payment issues emerging)
    X_t1[:, feature_names.index("billing_failures_6m")] *= np.random.uniform(1.8, 2.5,
                                                                               size=X_t1.shape[0])

    _, shap_t1 = compute_shap_values(model, X_t1, feature_names)
    drift_report = detect_shap_drift(shap_values, shap_t1, feature_names)
    paths.append(plot_shap_drift(drift_report, feature_names))

    # ── Save JSON audit trail ─────────────────────────────────────────────────
    audit = {
        "model_metrics":      metrics,
        "global_importance":  report.mean_abs_shap,
        "top_10_features":    report.top_features,
        "sample_0_explanation": report.sample_explanation,
        "drift_flagged":      drift_report["flagged"],
    }
    audit_path = os.path.join(OUTPUT_DIR, "shap_audit_trail.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  Saved: {audit_path}")

    # ── Print Report ──────────────────────────────────────────────────────────
    print_report(report, drift_report, metrics)

    print("\n  SHAP concepts demonstrated:")
    print("    TreeExplainer      — exact SHAP for tree-based models")
    print("    Global importance  — mean |SHAP| across all samples")
    print("    Beeswarm           — feature impact distribution (richest view)")
    print("    Waterfall          — single prediction step-by-step breakdown")
    print("    Dependence plot    — feature value vs SHAP value (non-linearity)")
    print("    SHAP drift         — importance change across time windows")
    print("    JSON audit trail   — regulatory-grade explanation record\n")


if __name__ == "__main__":
    main()
