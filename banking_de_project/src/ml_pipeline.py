"""
ml_pipeline.py
--------------
Fraud detection model built on the star-schema warehouse.

- Feature engineering (customer + account + behavioral aggregates joined onto txns)
- Train/test split with stratification (fraud is rare -> class imbalance)
- Pipeline: preprocessing (impute/scale/encode) + RandomForestClassifier
- Handles class imbalance via class_weight
- Evaluation: precision/recall/F1/ROC-AUC/PR-AUC (accuracy is misleading here)
- Feature importance + confusion matrix + ROC/PR curve plots
- Model persisted with pickle
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import sqlite3
import pickle
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score
)

WAREHOUSE = Path(__file__).resolve().parent.parent / "data" / "warehouse" / "banking_dw.sqlite"
FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
MODEL_DIR = Path(__file__).resolve().parent.parent / "outputs" / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "amount", "hour", "is_weekend", "customer_age", "monthly_income", "credit_score",
    "account_tenure_days", "customer_avg_amount", "customer_txn_count",
]
CATEGORICAL_FEATURES = ["txn_type", "category", "channel", "account_type", "day_of_week"]
TARGET = "is_fraud"


def build_feature_table() -> pd.DataFrame:
    conn = sqlite3.connect(WAREHOUSE)
    df = pd.read_sql("""
        SELECT f.transaction_id, f.amount, f.txn_type, f.category, f.channel, f.hour,
               f.date_key, f.is_fraud,
               a.account_id, a.account_type, a.open_date, a.customer_id,
               c.age AS customer_age, c.monthly_income, c.credit_score, c.signup_date
        FROM fact_transactions f
        JOIN dim_account a ON f.account_id = a.account_id
        JOIN dim_customer c ON a.customer_id = c.customer_id
    """, conn)
    conn.close()

    df["date_key"] = pd.to_datetime(df["date_key"])
    df["open_date"] = pd.to_datetime(df["open_date"], errors="coerce")
    df["day_of_week"] = df["date_key"].dt.day_name()
    df["is_weekend"] = df["date_key"].dt.dayofweek.isin([5, 6]).astype(int)
    df["account_tenure_days"] = (df["date_key"] - df["open_date"]).dt.days.clip(lower=0)

    # behavioral aggregates per customer (computed BEFORE split is fine here since
    # these are historical profile stats, not label-derived / leakage-free)
    cust_stats = df.groupby("customer_id")["amount"].agg(
        customer_avg_amount="mean", customer_txn_count="count").reset_index()
    df = df.merge(cust_stats, on="customer_id", how="left")

    return df


def train_and_evaluate(df: pd.DataFrame, model_name="random_forest"):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])

    if model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"=== {model_name} evaluation ===")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}  |  F1 (fraud): {f1:.4f}")

    metrics = {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_fraud": f1,
        "precision_fraud": report["Fraud"]["precision"],
        "recall_fraud": report["Fraud"]["recall"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "fraud_rate_test": float(y_test.mean()),
    }

    return pipeline, X_test, y_test, y_pred, y_proba, metrics


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set_title("Confusion Matrix — Fraud Detection")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_confusion_matrix.png")
    plt.close(fig)


def plot_roc_pr(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].plot(rec, prec, color="#dc2626", linewidth=2, label=f"AP = {pr_auc:.3f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_roc_pr_curves.png")
    plt.close(fig)


def plot_feature_importance(pipeline):
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    cat_names = prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names = NUMERIC_FEATURES + list(cat_names)
    importances = clf.feature_importances_

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x="importance", y="feature", data=imp_df, ax=ax, color="#16a34a")
    ax.set_title("Top 15 Feature Importances — Fraud Model (Random Forest)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_feature_importance.png")
    plt.close(fig)
    return imp_df


def run_all():
    print("Building feature table from warehouse...")
    df = build_feature_table()
    print(f"Feature table: {len(df):,} rows, fraud rate {df[TARGET].mean()*100:.3f}%")

    pipeline, X_test, y_test, y_pred, y_proba, metrics = train_and_evaluate(df, "random_forest")

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_pr(y_test, y_proba)
    imp_df = plot_feature_importance(pipeline)

    with open(MODEL_DIR / "fraud_model_random_forest.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    with open(REPORT_DIR / "ml_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    imp_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    print(f"\nModel saved to {MODEL_DIR/'fraud_model_random_forest.pkl'}")
    print(f"Metrics saved to {REPORT_DIR/'ml_metrics.json'}")
    print("\nTop 5 features:")
    print(imp_df.head(5).to_string(index=False))

    return pipeline, metrics


if __name__ == "__main__":
    run_all()
