"""
7_mlflow_experiments.py — MLflow Experiment Tracking + XGBoost Model Registry

Pattern: MLOps / Experiment Management
Key Concepts:
  - MLflow experiment tracking (params, metrics, artifacts)
  - XGBoost hyperparameter tuning across multiple runs
  - Model registry with staging → production promotion
  - Run comparison and best-model selection
  - Feature importance logging as artifact

Portfolio context: Production ML platforms require reproducible experiments,
auditable model lineage, and governed promotion workflows. This file
demonstrates the MLOps patterns used in enterprise AI deployments.
"""

import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works in CI/servers)
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
from mlflow.models import infer_signature
import os
import json
import tempfile

# ── Configuration ─────────────────────────────────────────────────────────────

EXPERIMENT_NAME  = "xgboost-classifier-tuning"
MODEL_NAME       = "xgboost-churn-predictor"      # name in MLflow Model Registry
MLFLOW_URI       = "sqlite:///mlflow_portfolio.db" # local SQLite tracking store
RANDOM_SEED      = 42

# Hyperparameter grid — each dict is one MLflow run
PARAM_GRID = [
    {"n_estimators": 50,  "max_depth": 3, "learning_rate": 0.1,  "subsample": 0.8},
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.01, "subsample": 1.0},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.2,  "subsample": 0.7},
]

# ── Data Generation ────────────────────────────────────────────────────────────

def generate_dataset():
    """
    Synthetic binary classification dataset simulating a churn-prediction task.
    In production this would be replaced by a feature store query or data pipeline.
    """
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        weights=[0.7, 0.3],   # class imbalance — realistic churn ratio
        random_state=RANDOM_SEED,
    )

    feature_names = (
        [f"usage_{i}"    for i in range(6)]  +
        [f"billing_{i}"  for i in range(4)]  +
        [f"support_{i}"  for i in range(4)]  +
        [f"product_{i}"  for i in range(4)]  +
        [f"tenure_{i}"   for i in range(2)]
    )

    df = pd.DataFrame(X, columns=feature_names)
    df["churn"] = y
    return df, feature_names


# ── Metric Helpers ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),           4),
        "f1":        round(f1_score(y_true, y_pred),                  4),
        "precision": round(precision_score(y_true, y_pred),           4),
        "recall":    round(recall_score(y_true, y_pred),              4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob[:, 1]),       4),
    }


# ── Feature Importance Plot ────────────────────────────────────────────────────

def save_feature_importance_plot(model, feature_names, run_id: str) -> str:
    """Save feature importance bar chart and return the file path."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]   # top-15

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(indices)), importances[indices], color="#1f77b4")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Feature Importance — Run {run_id[:8]}")
    ax.set_ylabel("Importance Score")
    ax.set_xlabel("Feature")
    plt.tight_layout()

    path = os.path.join(tempfile.gettempdir(), f"feature_importance_{run_id[:8]}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ── Single Training Run ────────────────────────────────────────────────────────

def train_and_log(params: dict, X_train, X_test, y_train, y_test,
                  feature_names: list, experiment_id: str) -> dict:
    """
    Train one XGBoost model with given params, log everything to MLflow,
    and return a summary dict for run-comparison.
    """
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id

        # ── Tags ──────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "model_type":    "XGBoostClassifier",
            "task":          "binary_classification",
            "dataset":       "synthetic_churn",
            "engineer":      "Mathanraja Ramasamy",
            "portfolio_file": "7_mlflow_experiments.py",
        })

        # ── Parameters ───────────────────────────────────────────────────────
        mlflow.log_params({
            **params,
            "eval_metric":      "logloss",
            "use_label_encoder": False,
            "random_state":     RANDOM_SEED,
            "train_samples":    len(X_train),
            "test_samples":     len(X_test),
            "n_features":       len(feature_names),
        })

        # ── Train ─────────────────────────────────────────────────────────────
        model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Metrics ──────────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # ── Artifacts: feature importance plot ───────────────────────────────
        plot_path = save_feature_importance_plot(model, feature_names, run_id)
        mlflow.log_artifact(plot_path, artifact_path="plots")
        os.remove(plot_path)

        # ── Artifacts: params + metrics as JSON (audit trail) ────────────────
        summary = {"run_id": run_id, "params": params, "metrics": metrics}
        summary_path = os.path.join(tempfile.gettempdir(), f"run_summary_{run_id[:8]}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path, artifact_path="metadata")
        os.remove(summary_path)

        # ── Log model with signature ──────────────────────────────────────────
        signature = infer_signature(X_train, y_pred)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,   # auto-registers in Model Registry
        )

        print(f"  run_id={run_id[:8]}  "
              f"n_est={params['n_estimators']:3d}  "
              f"depth={params['max_depth']}  "
              f"lr={params['learning_rate']:.2f}  │  "
              f"acc={metrics['accuracy']:.4f}  "
              f"auc={metrics['roc_auc']:.4f}  "
              f"f1={metrics['f1']:.4f}")

        return {**summary, "run_name": run.info.run_name}


# ── Best-Run Selection ────────────────────────────────────────────────────────

def select_best_run(runs: list[dict], metric: str = "roc_auc") -> dict:
    """Return the run with the highest value for the given metric."""
    return max(runs, key=lambda r: r["metrics"][metric])


# ── Model Registry: promote best run to Production ───────────────────────────

def promote_to_production(best_run: dict):
    """
    Transition the latest registered model version to 'Production' stage.
    In a real pipeline this would also run integration tests and require approval.
    """
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("  No registered versions found — skipping promotion.")
        return

    # find version whose source run_id matches our best run
    best_version = None
    for v in versions:
        if v.run_id == best_run["run_id"]:
            best_version = v
            break

    if best_version is None:
        # fallback: take the latest version
        best_version = sorted(versions, key=lambda v: int(v.version))[-1]

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=best_version.version,
        stage="Production",
        archive_existing_versions=True,   # demote previous Production versions
    )

    print(f"\n  Promoted version {best_version.version} → Production")
    print(f"  run_id : {best_run['run_id'][:8]}")
    print(f"  auc    : {best_run['metrics']['roc_auc']}")
    print(f"  f1     : {best_run['metrics']['f1']}")


# ── Run Comparison Report ─────────────────────────────────────────────────────

def print_comparison_table(runs: list[dict]):
    print("\n" + "─" * 80)
    print(f"  {'run_id':<10} {'n_est':>5} {'depth':>5} {'lr':>5} │ "
          f"{'acc':>6} {'auc':>6} {'f1':>6} {'prec':>6} {'rec':>6}")
    print("─" * 80)
    for r in sorted(runs, key=lambda x: x["metrics"]["roc_auc"], reverse=True):
        p = r["params"]
        m = r["metrics"]
        print(f"  {r['run_id'][:8]:<10} {p['n_estimators']:>5} {p['max_depth']:>5} "
              f"{p['learning_rate']:>5.2f} │ "
              f"{m['accuracy']:>6.4f} {m['roc_auc']:>6.4f} {m['f1']:>6.4f} "
              f"{m['precision']:>6.4f} {m['recall']:>6.4f}")
    print("─" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  MLflow Experiment Tracking + XGBoost Model Registry")
    print("  Pattern: MLOps / Reproducible Experiments / Governed Promotion")
    print("=" * 80)

    # ── Setup tracking store ──────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n  Experiment : {EXPERIMENT_NAME}")
    print(f"  Tracking   : {MLFLOW_URI}")
    print(f"  Experiment ID: {experiment.experiment_id}\n")

    # ── Prepare data ──────────────────────────────────────────────────────────
    df, feature_names = generate_dataset()
    X = df[feature_names].values
    y = df["churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Dataset: {len(df):,} samples | "
          f"train={len(X_train):,} test={len(X_test):,} | "
          f"churn rate={y.mean():.1%}\n")

    # ── Run hyperparameter sweep ──────────────────────────────────────────────
    print(f"  Running {len(PARAM_GRID)} experiments...\n")
    print(f"  {'run_id':<10} {'n_est':>5} {'depth':>5} {'lr':>5} │ "
          f"{'acc':>6} {'auc':>6} {'f1':>6}")
    print("  " + "─" * 65)

    all_runs = []
    for params in PARAM_GRID:
        result = train_and_log(
            params, X_train, X_test, y_train, y_test,
            feature_names, experiment.experiment_id
        )
        all_runs.append(result)

    # ── Compare all runs ──────────────────────────────────────────────────────
    print("\n  All Runs — Sorted by ROC-AUC")
    print_comparison_table(all_runs)

    # ── Select and promote best model ─────────────────────────────────────────
    best = select_best_run(all_runs, metric="roc_auc")
    print(f"\n  Best Run (by ROC-AUC): {best['run_id'][:8]}")
    print(f"  Params: n_estimators={best['params']['n_estimators']}, "
          f"max_depth={best['params']['max_depth']}, "
          f"learning_rate={best['params']['learning_rate']}")

    promote_to_production(best)

    # ── How to view results ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  To explore results in the MLflow UI:")
    print("    mlflow ui --backend-store-uri sqlite:///mlflow_portfolio.db")
    print("    → open http://localhost:5000")
    print()
    print("  Key MLflow concepts demonstrated:")
    print("    mlflow.start_run()          — creates an auditable run record")
    print("    mlflow.log_params()         — tracks hyperparameters")
    print("    mlflow.log_metrics()        — tracks evaluation scores")
    print("    mlflow.log_artifact()       — stores plots, JSON, model files")
    print("    mlflow.xgboost.log_model()  — saves model + auto-registers")
    print("    client.transition_model_version_stage() — governs promotion")
    print("=" * 80)


if __name__ == "__main__":
    main()
