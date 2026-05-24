"""
12_evidently_drift.py — Evidently AI Model & Data Drift Monitoring

Pattern: MLOps / Production Monitoring / Data Quality
Key Concepts:
  - Evidently AI: open-source ML observability framework
  - Data drift detection: statistical tests on feature distributions
  - Target drift: output distribution changes over time
  - Data quality report: missing values, outliers, schema violations
  - Model performance monitoring: track accuracy/AUC over time windows
  - HTML report generation: stakeholder-ready drift reports
  - CI/CD integration: fail pipeline if drift thresholds exceeded

Production context:
  Models degrade silently. By the time accuracy drops noticeably,
  weeks of bad predictions may have already impacted users.
  Evidently catches distribution shifts BEFORE they become accuracy issues —
  giving you early warning to retrain before the model fails in production.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from dataclasses import dataclass, field
from typing import Optional

# Evidently imports — graceful fallback if not installed
try:
    from evidently.report import Report
    from evidently.metric_preset import (
        DataDriftPreset,
        DataQualityPreset,
        TargetDriftPreset,
        ClassificationPreset,
    )
    from evidently.metrics import (
        DatasetDriftMetric,
        DatasetMissingValuesSummaryMetric,
        ColumnDriftMetric,
    )
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
    from evidently.tests import TestNumberOfDriftedColumns
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────

RANDOM_SEED         = 42
OUTPUT_DIR          = "evidently_outputs"
DRIFT_THRESHOLD     = 0.3     # fraction of features drifted that triggers alert
PERF_DROP_THRESHOLD = 0.05    # AUC drop that triggers retraining alert

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Feature Schema ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "days_since_login",       # recency
    "monthly_spend",          # value
    "support_tickets_90d",    # friction
    "plan_tier",              # categorical (0/1/2)
    "contract_months_left",   # commitment
    "nps_score",              # satisfaction
    "feature_adoption_pct",   # engagement
    "billing_failures_6m",    # risk
    "referrals_made",         # loyalty
    "mobile_app_sessions",    # activity
    "email_open_rate",        # marketing
    "api_calls_monthly",      # technical
    "login_failures_30d",     # friction
    "upsell_offers_declined", # intent
    "tenure_months",          # loyalty
]

CATEGORICAL_FEATURES = ["plan_tier"]
NUMERICAL_FEATURES   = [f for f in FEATURE_NAMES if f not in CATEGORICAL_FEATURES]


# ── Dataset Builder ────────────────────────────────────────────────────────────

def build_dataset(n_samples: int, drift_factor: float = 0.0,
                  noise_features: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Build a churn dataset. drift_factor > 0 injects distributional shift
    into specified features to simulate production drift scenarios.
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

    df = pd.DataFrame(X_raw, columns=FEATURE_NAMES)

    # Discretise plan_tier to 0/1/2
    df["plan_tier"] = pd.cut(df["plan_tier"], bins=3, labels=[0, 1, 2]).astype(int)

    # Inject drift into specified features
    if drift_factor > 0 and noise_features:
        np.random.seed(99)
        for fname in noise_features:
            if fname in df.columns:
                shift = df[fname].std() * drift_factor
                df[fname] = df[fname] + np.random.normal(shift, shift * 0.3, size=len(df))

    df["churn"]         = y
    df["churn_proba"]   = 0.0   # filled after model prediction
    df["prediction"]    = 0     # filled after model prediction

    return df


# ── Model ──────────────────────────────────────────────────────────────────────

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def add_predictions(df: pd.DataFrame, model, feature_names: list[str]) -> pd.DataFrame:
    df = df.copy()
    X = df[feature_names].values
    df["churn_proba"] = model.predict_proba(X)[:, 1]
    df["prediction"]  = model.predict(X)
    return df


# ── Custom Drift Analysis (no Evidently needed) ────────────────────────────────

@dataclass
class DriftResult:
    feature:     str
    stat_method: str
    statistic:   float
    p_value:     float
    drifted:     bool
    severity:    str    # low / medium / high


@dataclass
class DriftReport:
    window:            str
    n_reference:       int
    n_current:         int
    results:           list[DriftResult]
    drifted_features:  list[str]
    drift_fraction:    float
    alert_triggered:   bool
    perf_reference:    dict
    perf_current:      dict
    perf_drop:         dict


def kolmogorov_smirnov_test(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test — detects any shape difference in distributions."""
    from scipy import stats
    stat, p = stats.ks_2samp(ref, cur)
    return float(stat), float(p)


def population_stability_index(ref: np.ndarray, cur: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    PSI — Population Stability Index:
      PSI < 0.1   : no drift
      PSI 0.1–0.2 : slight drift (monitor)
      PSI > 0.2   : significant drift (investigate)
    """
    ref_counts, bin_edges = np.histogram(ref, bins=n_bins)
    cur_counts, _         = np.histogram(cur, bins=bin_edges)

    # Avoid zero division
    ref_pct = (ref_counts + 1e-6) / len(ref)
    cur_pct = (cur_counts + 1e-6) / len(cur)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def detect_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    features: list[str],
    p_value_threshold: float = 0.05,
    psi_threshold: float = 0.2,
) -> list[DriftResult]:
    results = []

    for fname in features:
        ref_vals = ref_df[fname].dropna().values.astype(float)
        cur_vals = cur_df[fname].dropna().values.astype(float)

        ks_stat, p_val = kolmogorov_smirnov_test(ref_vals, cur_vals)
        psi            = population_stability_index(ref_vals, cur_vals)
        drifted        = p_val < p_value_threshold

        if psi > 0.4 or ks_stat > 0.4:
            severity = "high"
        elif psi > 0.2 or ks_stat > 0.2:
            severity = "medium"
        else:
            severity = "low"

        results.append(DriftResult(
            feature=fname, stat_method="KS+PSI",
            statistic=round(ks_stat, 4), p_value=round(p_val, 4),
            drifted=drifted, severity=severity,
        ))

    return results


def compare_performance(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
) -> tuple[dict, dict, dict]:
    def metrics(df):
        return {
            "auc":      round(roc_auc_score(df["churn"],    df["churn_proba"]), 4),
            "f1":       round(f1_score(      df["churn"],    df["prediction"]),  4),
            "accuracy": round(accuracy_score(df["churn"],    df["prediction"]),  4),
        }

    ref_m = metrics(ref_df)
    cur_m = metrics(cur_df)
    drop  = {k: round(ref_m[k] - cur_m[k], 4) for k in ref_m}
    return ref_m, cur_m, drop


def build_drift_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    features: list[str],
    window_label: str = "T0 vs T1",
) -> DriftReport:
    results         = detect_drift(ref_df, cur_df, features)
    drifted         = [r.feature for r in results if r.drifted]
    drift_fraction  = len(drifted) / len(features)
    alert_triggered = drift_fraction >= DRIFT_THRESHOLD

    ref_perf, cur_perf, perf_drop = compare_performance(ref_df, cur_df)

    return DriftReport(
        window=window_label,
        n_reference=len(ref_df),
        n_current=len(cur_df),
        results=results,
        drifted_features=drifted,
        drift_fraction=round(drift_fraction, 3),
        alert_triggered=alert_triggered,
        perf_reference=ref_perf,
        perf_current=cur_perf,
        perf_drop=perf_drop,
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_feature_distributions(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    features: list[str],
    drift_results: list[DriftResult],
    n_cols: int = 3,
) -> str:
    drifted_set = {r.feature for r in drift_results if r.drifted}
    n_features  = min(len(features), 9)   # cap at 9 for readability
    n_rows      = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()

    for i, fname in enumerate(features[:n_features]):
        ax     = axes[i]
        color  = "#d62728" if fname in drifted_set else "#1f77b4"
        title  = f"{fname}  ⚠ DRIFT" if fname in drifted_set else fname

        ax.hist(ref_df[fname].values, bins=25, alpha=0.6, color="#aec7e8",
                label="Reference", density=True)
        ax.hist(cur_df[fname].values, bins=25, alpha=0.6, color=color,
                label="Current",   density=True)
        ax.set_title(title, fontsize=9, color=color if fname in drifted_set else "black")
        ax.legend(fontsize=7)
        ax.set_xlabel(fname, fontsize=8)

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distribution Comparison — Reference vs Current Window",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "feature_distributions.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_drift_heatmap(drift_results: list[DriftResult]) -> str:
    features   = [r.feature    for r in drift_results]
    ks_stats   = [r.statistic  for r in drift_results]
    p_values   = [r.p_value    for r in drift_results]
    drifted    = [r.drifted    for r in drift_results]

    sorted_idx = np.argsort(ks_stats)[::-1]
    fig, axes  = plt.subplots(1, 2, figsize=(14, 6))

    # KS statistic bar
    ax = axes[0]
    colors = ["#d62728" if drifted[i] else "#1f77b4" for i in sorted_idx]
    ax.barh([features[i] for i in reversed(sorted_idx)],
            [ks_stats[i] for i in reversed(sorted_idx)],
            color=colors[::-1])
    ax.axvline(x=0.2, color="orange", linestyle="--", linewidth=1, label="Medium threshold")
    ax.axvline(x=0.4, color="red",    linestyle="--", linewidth=1, label="High threshold")
    ax.set_xlabel("KS Statistic")
    ax.set_title("Drift Severity by Feature (KS Statistic)")
    ax.legend(fontsize=8)

    # P-value bar
    ax = axes[1]
    ax.barh([features[i] for i in reversed(sorted_idx)],
            [-np.log10(max(p_values[i], 1e-10)) for i in reversed(sorted_idx)],
            color=colors[::-1])
    ax.axvline(x=-np.log10(0.05), color="red", linestyle="--", linewidth=1,
               label="p=0.05 threshold")
    ax.set_xlabel("-log10(p-value)  [higher = more significant drift]")
    ax.set_title("Statistical Significance of Drift (KS p-value)")
    ax.legend(fontsize=8)

    plt.suptitle("Drift Detection Heatmap — All Features", fontsize=12)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "drift_heatmap.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_performance_timeline(windows: list[dict]) -> str:
    """
    Simulate performance over multiple monitoring windows — shows gradual
    degradation pattern that Evidently/custom monitoring catches.
    """
    labels  = [w["label"] for w in windows]
    aucs    = [w["auc"]   for w in windows]
    f1s     = [w["f1"]    for w in windows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(labels, aucs, marker="o", linewidth=2, color="#1f77b4", label="ROC-AUC")
    ax.plot(labels, f1s,  marker="s", linewidth=2, color="#ff7f0e", label="F1 Score")
    ax.axhline(y=aucs[0] - PERF_DROP_THRESHOLD, color="red", linestyle="--",
               linewidth=1, label=f"Alert threshold (AUC drop >{PERF_DROP_THRESHOLD})")
    ax.fill_between(labels,
                    [v - PERF_DROP_THRESHOLD for v in [aucs[0]] * len(labels)],
                    [min(aucs + f1s) - 0.02] * len(labels),
                    alpha=0.08, color="red")
    ax.set_xlabel("Monitoring Window")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Timeline — Degradation Detection")
    ax.legend()
    ax.set_ylim(min(f1s) - 0.05, max(aucs) + 0.05)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "performance_timeline.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Evidently HTML Report ─────────────────────────────────────────────────────

def generate_evidently_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    feature_names: list[str],
) -> Optional[str]:
    """
    Generate a full Evidently HTML report with:
      - Data Drift (KS test per feature, drift share)
      - Data Quality (missing values, outliers, schema)
      - Target Drift (output distribution shift)
    Returns path to saved HTML, or None if Evidently not installed.
    """
    if not EVIDENTLY_AVAILABLE:
        return None

    ref = ref_df[feature_names + ["churn", "churn_proba", "prediction"]].copy()
    cur = cur_df[feature_names + ["churn", "churn_proba", "prediction"]].copy()
    ref.rename(columns={"churn": "target", "churn_proba": "prediction_proba",
                        "prediction": "prediction"}, inplace=True)
    cur.rename(columns={"churn": "target", "churn_proba": "prediction_proba",
                        "prediction": "prediction"}, inplace=True)

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        ClassificationPreset(),
    ])
    report.run(reference_data=ref, current_data=cur)

    path = os.path.join(OUTPUT_DIR, "evidently_drift_report.html")
    report.save_html(path)
    print(f"  Saved: {path}  (open in browser for interactive dashboard)")
    return path


def generate_evidently_test_suite(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    feature_names: list[str],
) -> Optional[dict]:
    """
    Run Evidently TestSuite — returns pass/fail for CI/CD gates.
    Fails if: >30% features drifted OR data quality issues detected.
    """
    if not EVIDENTLY_AVAILABLE:
        return None

    ref = ref_df[feature_names + ["churn"]].copy()
    cur = cur_df[feature_names + ["churn"]].copy()

    suite = TestSuite(tests=[
        DataDriftTestPreset(),
        DataQualityTestPreset(),
        TestNumberOfDriftedColumns(lt=int(len(feature_names) * DRIFT_THRESHOLD)),
    ])
    suite.run(reference_data=ref, current_data=cur)

    path = os.path.join(OUTPUT_DIR, "evidently_test_suite.html")
    suite.save_html(path)

    results = suite.as_dict()
    passed  = results["summary"]["all_passed"]
    return {"passed": passed, "summary": results["summary"], "html_path": path}


# ── Print Report ──────────────────────────────────────────────────────────────

def print_drift_report(report: DriftReport):
    print("\n" + "=" * 70)
    print(f"  DRIFT MONITORING REPORT — {report.window}")
    print("=" * 70)
    print(f"  Reference : {report.n_reference:,} samples")
    print(f"  Current   : {report.n_current:,} samples")
    print()

    # Performance
    print("  Model Performance:")
    for metric in ["auc", "f1", "accuracy"]:
        ref_v = report.perf_reference[metric]
        cur_v = report.perf_current[metric]
        drop  = report.perf_drop[metric]
        flag  = " ⚠ ALERT" if metric == "auc" and drop > PERF_DROP_THRESHOLD else ""
        print(f"    {metric.upper():<10}  reference={ref_v:.4f}  current={cur_v:.4f}  "
              f"Δ={drop:+.4f}{flag}")

    # Feature drift
    print(f"\n  Feature Drift ({len(report.drifted_features)}/{len(report.results)} features drifted, "
          f"{report.drift_fraction*100:.1f}%):")

    for r in sorted(report.results, key=lambda x: x.statistic, reverse=True)[:10]:
        flag = "⚠" if r.drifted else "✓"
        print(f"    {flag} {r.feature:<28} KS={r.statistic:.4f}  p={r.p_value:.4f}  "
              f"[{r.severity}]")

    alert_msg = "⚠ DRIFT ALERT — investigate and consider retraining" \
        if report.alert_triggered else "✓ Within acceptable drift bounds"
    print(f"\n  Overall: {alert_msg}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Evidently AI — Data & Model Drift Monitoring")
    print("  Pattern : Production ML observability + CI/CD quality gates")
    print("=" * 70)

    # ── Build reference dataset (training time / last month) ─────────────────
    print("\n  Building reference dataset (training window)...")
    ref_df = build_dataset(n_samples=3000)

    # ── Train model on reference ──────────────────────────────────────────────
    X_ref = ref_df[FEATURE_NAMES].values
    y_ref = ref_df["churn"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X_ref, y_ref, test_size=0.2, random_state=RANDOM_SEED, stratify=y_ref
    )
    model = train_model(X_train, y_train)
    ref_df = add_predictions(ref_df, model, FEATURE_NAMES)
    print(f"  Model trained — reference AUC: {roc_auc_score(y_ref, ref_df['churn_proba'].values):.4f}")

    # ── Build current window with drift ──────────────────────────────────────
    print("\n  Building current window (simulated production drift)...")
    DRIFTED_FEATURES = [
        "days_since_login",       # users logging in less — early churn signal
        "support_tickets_90d",    # support load increasing
        "billing_failures_6m",    # payment issues rising
        "email_open_rate",        # marketing disengagement
    ]
    cur_df = build_dataset(
        n_samples=1500,
        drift_factor=1.8,
        noise_features=DRIFTED_FEATURES,
    )
    cur_df = add_predictions(cur_df, model, FEATURE_NAMES)

    # ── Run drift analysis ────────────────────────────────────────────────────
    print("\n  Running drift detection...")
    drift_report = build_drift_report(ref_df, cur_df, FEATURE_NAMES, "Training vs Production")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\n  Generating monitoring plots...")
    plot_feature_distributions(ref_df, cur_df, FEATURE_NAMES, drift_report.results)
    plot_drift_heatmap(drift_report.results)

    # Performance timeline (simulate 5 monitoring windows)
    perf_windows = []
    for i, factor in enumerate([0.0, 0.3, 0.6, 1.0, 1.8]):
        win_df = build_dataset(n_samples=800, drift_factor=factor,
                               noise_features=DRIFTED_FEATURES[:2])
        win_df = add_predictions(win_df, model, FEATURE_NAMES)
        auc = roc_auc_score(win_df["churn"], win_df["churn_proba"])
        f1  = f1_score(win_df["churn"], win_df["prediction"])
        perf_windows.append({"label": f"W{i+1}", "auc": round(auc, 4), "f1": round(f1, 4)})
    plot_performance_timeline(perf_windows)

    # ── Evidently HTML (if installed) ────────────────────────────────────────
    if EVIDENTLY_AVAILABLE:
        print("\n  Generating Evidently HTML reports...")
        generate_evidently_report(ref_df, cur_df, FEATURE_NAMES)
        test_result = generate_evidently_test_suite(ref_df, cur_df, FEATURE_NAMES)
        if test_result:
            status = "PASS" if test_result["passed"] else "FAIL"
            print(f"  Evidently TestSuite CI/CD gate: {status}")
    else:
        print("\n  Evidently not installed — running custom drift engine only.")
        print("  Install: pip install evidently")
        print("  Then run again to generate interactive HTML reports.")

    # ── Save JSON audit trail ─────────────────────────────────────────────────
    audit = {
        "window":           drift_report.window,
        "drift_fraction":   drift_report.drift_fraction,
        "alert_triggered":  drift_report.alert_triggered,
        "drifted_features": drift_report.drifted_features,
        "perf_reference":   drift_report.perf_reference,
        "perf_current":     drift_report.perf_current,
        "perf_drop":        drift_report.perf_drop,
        "feature_results":  [
            {"feature": r.feature, "ks_stat": r.statistic, "p_value": r.p_value,
             "drifted": r.drifted, "severity": r.severity}
            for r in drift_report.results
        ],
    }
    audit_path = os.path.join(OUTPUT_DIR, "drift_audit_trail.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  Saved: {audit_path}")

    # ── Print report ──────────────────────────────────────────────────────────
    print_drift_report(drift_report)

    print()
    print(f"  Outputs saved to: {OUTPUT_DIR}/")
    print()
    print("  Monitoring patterns demonstrated:")
    print("    KS test          — detects any shape difference in distributions")
    print("    PSI              — quantifies magnitude of distribution shift")
    print("    Performance gap  — tracks AUC/F1 degradation across windows")
    print("    Drift heatmap    — visual severity ranking across all features")
    print("    Timeline plot    — shows gradual degradation pattern")
    print("    JSON audit       — regulatory-grade monitoring record")
    print("    Evidently HTML   — interactive stakeholder dashboard (if installed)")
    print()
    print("  CI/CD integration:")
    print("    if drift_report.alert_triggered:")
    print("        trigger_retraining_pipeline()")
    print("        notify_ml_team('Drift alert: ' + str(drift_report.drifted_features))")
    print()


if __name__ == "__main__":
    main()
