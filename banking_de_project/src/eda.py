"""
eda.py
------
Exploratory Data Analysis over the star-schema warehouse.
Produces a set of publication-quality figures + a text summary report.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

WAREHOUSE = Path(__file__).resolve().parent.parent / "data" / "warehouse" / "banking_dw.sqlite"
FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
REPORT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"


def load():
    conn = sqlite3.connect(WAREHOUSE)
    customers = pd.read_sql("SELECT * FROM dim_customer", conn)
    accounts = pd.read_sql("SELECT * FROM dim_account", conn)
    txns = pd.read_sql("""
        SELECT f.*, a.customer_id, a.account_type
        FROM fact_transactions f JOIN dim_account a ON f.account_id = a.account_id
    """, conn)
    loans = pd.read_sql("SELECT * FROM fact_loans", conn)
    conn.close()
    txns["date_key"] = pd.to_datetime(txns["date_key"])
    return customers, accounts, txns, loans


def fig_income_distribution(customers):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(customers["monthly_income"], bins=50, kde=True, ax=axes[0], color="#2563eb")
    axes[0].set_title("Customer Monthly Income Distribution")
    axes[0].set_xlabel("Monthly Income (ZAR)")

    sns.histplot(customers["credit_score"], bins=40, kde=True, ax=axes[1], color="#16a34a")
    axes[1].set_title("Customer Credit Score Distribution")
    axes[1].set_xlabel("Credit Score")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_income_credit_distribution.png")
    plt.close(fig)


def fig_txn_by_category(txns):
    top = txns.groupby("category")["amount"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(x=top["sum"] / 1e6, y=top.index, ax=ax, color="#7c3aed")
    ax.set_title("Total Transaction Volume by Category")
    ax.set_xlabel("Total Amount (ZAR millions)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_txn_volume_by_category.png")
    plt.close(fig)


def fig_daily_trend(txns):
    daily = txns.groupby("date_key")["amount"].sum().reset_index()
    daily["date_key"] = pd.to_datetime(daily["date_key"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily["date_key"], daily["amount"] / 1e3, color="#0891b2", linewidth=1)
    daily["roll"] = daily["amount"].rolling(14).mean()
    ax.plot(daily["date_key"], daily["roll"] / 1e3, color="#dc2626", linewidth=2, label="14-day avg")
    ax.set_title("Daily Transaction Volume Over Time")
    ax.set_ylabel("Amount (ZAR thousands)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_daily_transaction_trend.png")
    plt.close(fig)


def fig_channel_hour_heatmap(txns):
    pivot = txns.pivot_table(index="channel", columns="hour", values="amount",
                              aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(pivot, cmap="rocket_r", ax=ax, cbar_kws={"label": "Txn Count"})
    ax.set_title("Transaction Frequency: Channel x Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_channel_hour_heatmap.png")
    plt.close(fig)


def fig_account_balance_by_type(accounts):
    conn = sqlite3.connect(WAREHOUSE)
    bal = pd.read_sql("SELECT account_type, customer_id, account_id FROM dim_account", conn)
    conn.close()
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = accounts["account_type"].value_counts()
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#f59e0b")
    ax.set_title("Account Count by Type")
    ax.set_xlabel("Number of Accounts")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_account_type_distribution.png")
    plt.close(fig)


def fig_fraud_overview(txns):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fraud_rate = txns["is_fraud"].mean() * 100
    sns.countplot(x="is_fraud", hue="is_fraud", data=txns, ax=axes[0], palette=["#2563eb", "#dc2626"], legend=False)
    axes[0].set_title(f"Fraud vs Legit Transactions ({fraud_rate:.2f}% fraud)")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Legit", "Fraud"])

    sns.boxplot(x="is_fraud", y="amount", hue="is_fraud", data=txns[txns["amount"] < txns["amount"].quantile(0.99)],
                ax=axes[1], palette=["#2563eb", "#dc2626"], legend=False)
    axes[1].set_title("Transaction Amount: Fraud vs Legit")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Legit", "Fraud"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_fraud_overview.png")
    plt.close(fig)


def fig_correlation_heatmap(customers, loans):
    merged = customers.merge(
        loans.groupby("customer_id")["principal"].sum().reset_index().rename(
            columns={"principal": "total_loan_principal"}),
        on="customer_id", how="left")
    merged["total_loan_principal"] = merged["total_loan_principal"].fillna(0)
    num_cols = ["age", "monthly_income", "credit_score", "total_loan_principal"]
    corr = merged[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation: Customer Financial Attributes")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_correlation_heatmap.png")
    plt.close(fig)


def fig_loan_status(loans):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    status_counts = loans["status"].value_counts()
    axes[0].pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%",
                colors=sns.color_palette("deep"))
    axes[0].set_title("Loan Portfolio Status")

    sns.boxplot(x="loan_type", y="interest_rate", data=loans, ax=axes[1], color="#10b981")
    axes[1].set_title("Interest Rate by Loan Type")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_loan_portfolio.png")
    plt.close(fig)


def write_summary_report(customers, accounts, txns, loans):
    lines = []
    lines.append("BANKING DATA WAREHOUSE — EDA SUMMARY REPORT")
    lines.append("=" * 50)
    lines.append(f"\nCustomers: {len(customers):,}")
    lines.append(f"  Avg age: {customers['age'].mean():.1f}  |  Avg income: R{customers['monthly_income'].mean():,.0f}")
    lines.append(f"  Avg credit score: {customers['credit_score'].mean():.0f}")
    lines.append(f"\nAccounts: {len(accounts):,}")
    lines.append(f"  Active: {accounts['is_active'].sum():,} ({accounts['is_active'].mean()*100:.1f}%)")
    lines.append(f"\nTransactions: {len(txns):,}")
    lines.append(f"  Total volume: R{txns['amount'].sum():,.0f}")
    lines.append(f"  Fraud rate: {txns['is_fraud'].mean()*100:.3f}%  ({txns['is_fraud'].sum():,} flagged)")
    lines.append(f"  Avg transaction: R{txns['amount'].mean():,.2f}  |  Median: R{txns['amount'].median():,.2f}")
    top_cat = txns.groupby("category")["amount"].sum().idxmax()
    lines.append(f"  Top category by volume: {top_cat}")
    lines.append(f"\nLoans: {len(loans):,}")
    lines.append(f"  Total principal outstanding: R{loans['principal'].sum():,.0f}")
    lines.append(f"  Default rate: {(loans['status']=='Defaulted').mean()*100:.1f}%")
    lines.append(f"  Avg interest rate: {loans['interest_rate'].mean():.2f}%")

    report = "\n".join(lines)
    (REPORT_DIR / "eda_summary.txt").write_text(report)
    print(report)


def run_all():
    customers, accounts, txns, loans = load()
    fig_income_distribution(customers)
    fig_txn_by_category(txns)
    fig_daily_trend(txns)
    fig_channel_hour_heatmap(txns)
    fig_account_balance_by_type(accounts)
    fig_fraud_overview(txns)
    fig_correlation_heatmap(customers, loans)
    fig_loan_status(loans)
    write_summary_report(customers, accounts, txns, loans)
    print(f"\n8 figures saved to {FIG_DIR}")


if __name__ == "__main__":
    run_all()
