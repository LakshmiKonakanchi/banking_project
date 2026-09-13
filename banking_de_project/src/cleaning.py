"""
cleaning.py
-----------
Deterministic, logged cleaning transforms for each raw dataset.
Every function returns (clean_df, log: list[str]) so the pipeline can
produce an auditable "what changed and why" trail — critical in
banking where you must be able to explain every transformation.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log = []
    df = df.copy()
    before = len(df)

    # drop exact duplicate rows
    df = df.drop_duplicates()
    log.append(f"Dropped {before - len(df):,} exact duplicate rows")

    # drop duplicate customer_id keeping first occurrence (data re-submission)
    before = len(df)
    df = df.drop_duplicates(subset="customer_id", keep="first")
    log.append(f"Dropped {before - len(df):,} duplicate customer_id rows (kept first)")

    # standardize text fields
    df["first_name"] = df["first_name"].str.strip().str.title()
    df["last_name"] = df["last_name"].str.strip().str.title()
    df["city"] = df["city"].str.strip().str.title()
    log.append("Standardized casing/whitespace on first_name, last_name, city")

    # parse mixed-format dates
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce", format="mixed")
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce", format="mixed")
    log.append("Parsed date_of_birth and signup_date from mixed string formats")

    # fix invalid credit scores -> clip to valid FICO-like range, flag out-of-range as NaN
    invalid_mask = ~df["credit_score"].between(300, 850)
    n_invalid = invalid_mask.sum()
    df.loc[invalid_mask, "credit_score"] = np.nan
    log.append(f"Nulled {n_invalid:,} credit_score values outside valid [300, 850] range")

    # impute credit_score median, monthly_income median (robust to outliers)
    median_score = df["credit_score"].median()
    df["credit_score"] = df["credit_score"].fillna(median_score)
    log.append(f"Imputed missing credit_score with median ({median_score:.0f})")

    # cap extreme income outliers at 99.5th percentile (winsorize) rather than drop
    cap = df["monthly_income"].quantile(0.995)
    n_capped = (df["monthly_income"] > cap).sum()
    df["monthly_income"] = df["monthly_income"].clip(upper=cap)
    log.append(f"Winsorized {n_capped:,} monthly_income outliers at 99.5th pct ({cap:,.0f})")

    median_income = df["monthly_income"].median()
    df["monthly_income"] = df["monthly_income"].fillna(median_income)
    log.append(f"Imputed missing monthly_income with median ({median_income:,.0f})")

    df["occupation"] = df["occupation"].fillna("Unknown")
    log.append("Filled missing occupation with 'Unknown'")

    # derive age
    ref_date = pd.Timestamp("2026-08-15")
    df["age"] = ((ref_date - df["date_of_birth"]).dt.days / 365.25).round(1)

    df["email"] = df["email"].str.lower().str.strip()

    n_before = len(df)
    df = df.dropna(subset=["customer_id", "date_of_birth"])
    log.append(f"Dropped {n_before - len(df):,} rows with missing critical keys (customer_id/dob)")

    return df.reset_index(drop=True), log


def clean_accounts(df: pd.DataFrame, valid_customer_ids: set) -> tuple[pd.DataFrame, list[str]]:
    log = []
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset="account_id", keep="first")
    log.append(f"Dropped {before - len(df):,} duplicate account_id rows")

    before = len(df)
    df = df[df["customer_id"].isin(valid_customer_ids)]
    log.append(f"Removed {before - len(df):,} orphan accounts (customer_id FK violation)")

    df["currency"] = df["currency"].astype(str).str.strip().str.upper()
    log.append("Normalized currency codes (strip + upper)")

    # negative balances beyond a plausible overdraft limit -> treat as data error, null out
    bad_balance = df["balance"] < -100_000
    n_bad = bad_balance.sum()
    df.loc[bad_balance, "balance"] = np.nan
    log.append(f"Nulled {n_bad:,} balance values below plausible overdraft floor (-100,000)")

    median_balance = df["balance"].median()
    df["balance"] = df["balance"].fillna(median_balance)
    log.append(f"Imputed missing balance with median ({median_balance:,.0f})")

    df["open_date"] = pd.to_datetime(df["open_date"], errors="coerce", format="mixed")
    df["is_active"] = df["is_active"].astype(int)

    return df.reset_index(drop=True), log


def clean_transactions(df: pd.DataFrame, valid_account_ids: set) -> tuple[pd.DataFrame, list[str]]:
    log = []
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset="transaction_id", keep="first")
    log.append(f"Dropped {before - len(df):,} duplicate transaction_id rows")

    before = len(df)
    df = df[df["account_id"].isin(valid_account_ids)]
    log.append(f"Removed {before - len(df):,} transactions with orphan account_id")

    # fix sign errors: amount should always be stored positive; direction is in txn_type
    n_neg = (df["amount"] < 0).sum()
    df["amount"] = df["amount"].abs()
    log.append(f"Corrected {n_neg:,} negative amount sign errors (took absolute value)")

    df["category"] = df["category"].fillna("Uncategorized")
    log.append("Filled missing category with 'Uncategorized'")

    before_na = df["amount"].isna().sum()
    median_amt_by_cat = df.groupby("category")["amount"].transform("median")
    df["amount"] = df["amount"].fillna(median_amt_by_cat)
    df["amount"] = df["amount"].fillna(df["amount"].median())
    log.append(f"Imputed {before_na:,} missing amounts with category-level median")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    before = len(df)
    df = df.dropna(subset=["timestamp"])
    log.append(f"Dropped {before - len(df):,} rows with unparseable timestamps")

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    return df.reset_index(drop=True), log


def clean_loans(df: pd.DataFrame, valid_customer_ids: set) -> tuple[pd.DataFrame, list[str]]:
    log = []
    df = df.copy()

    before = len(df)
    df = df[df["customer_id"].isin(valid_customer_ids)]
    log.append(f"Removed {before - len(df):,} loans with orphan customer_id")

    median_rate = df["interest_rate"].median()
    n_missing = df["interest_rate"].isna().sum()
    df["interest_rate"] = df["interest_rate"].fillna(median_rate)
    log.append(f"Imputed {n_missing:,} missing interest_rate with median ({median_rate:.2f}%)")

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", format="mixed")
    df["monthly_payment"] = (
        df["principal"] * (df["interest_rate"] / 100 / 12) /
        (1 - (1 + df["interest_rate"] / 100 / 12) ** (-df["term_months"]))
    ).round(2)

    return df.reset_index(drop=True), log


def run_all():
    raw = Path(__file__).resolve().parent.parent / "data" / "raw"
    customers = pd.read_csv(raw / "customers.csv")
    accounts = pd.read_csv(raw / "accounts.csv")
    transactions = pd.read_csv(raw / "transactions.csv")
    loans = pd.read_csv(raw / "loans.csv")

    full_log = {}

    customers_c, log = clean_customers(customers)
    full_log["customers"] = log
    accounts_c, log = clean_accounts(accounts, set(customers_c["customer_id"]))
    full_log["accounts"] = log
    transactions_c, log = clean_transactions(transactions, set(accounts_c["account_id"]))
    full_log["transactions"] = log
    loans_c, log = clean_loans(loans, set(customers_c["customer_id"]))
    full_log["loans"] = log

    customers_c.to_csv(PROCESSED_DIR / "customers_clean.csv", index=False)
    accounts_c.to_csv(PROCESSED_DIR / "accounts_clean.csv", index=False)
    transactions_c.to_csv(PROCESSED_DIR / "transactions_clean.csv", index=False)
    loans_c.to_csv(PROCESSED_DIR / "loans_clean.csv", index=False)

    print("=== CLEANING LOG ===")
    for dataset, entries in full_log.items():
        print(f"\n[{dataset}]")
        for e in entries:
            print(f"  - {e}")

    print(f"\nFinal row counts: customers={len(customers_c):,} accounts={len(accounts_c):,} "
          f"transactions={len(transactions_c):,} loans={len(loans_c):,}")

    return customers_c, accounts_c, transactions_c, loans_c, full_log


if __name__ == "__main__":
    run_all()
