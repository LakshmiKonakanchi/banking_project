"""
generate_data.py
-----------------
Generates realistic, MESSY synthetic banking datasets:
  - customers.csv
  - accounts.csv
  - transactions.csv
  - loans.csv

Intentionally injects the kinds of problems a senior data engineer
has to deal with in production: missing values, duplicates, mixed
date formats, inconsistent casing/whitespace, outliers, orphan
foreign keys, and a small % of fraudulent transactions (for the
downstream ML task).

No external dependencies beyond numpy/pandas -> works offline.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

N_CUSTOMERS = 5000
N_ACCOUNTS = 7000
N_TRANSACTIONS = 120_000
N_LOANS = 2500

FIRST_NAMES = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
               "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
               "Thabo", "Naledi", "Sipho", "Zanele", "Kagiso", "Lerato", "Amara", "Chidi",
               "Wei", "Mei", "Raj", "Priya", "Ahmed", "Fatima", "Carlos", "Sofia"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Ndlovu", "Dlamini", "Khumalo", "Mokoena", "Chen", "Patel",
              "Khan", "Silva", "Van der Merwe", "Botha", "Pretorius", "Nkosi"]
CITIES = ["Johannesburg", "Cape Town", "Durban", "Pretoria", "Port Elizabeth",
          "Bloemfontein", "East London", "Polokwane", "Nelspruit", "Kimberley"]
OCCUPATIONS = ["Engineer", "Teacher", "Nurse", "Accountant", "Driver", "Manager",
               "Sales Rep", "Technician", "Consultant", "Artisan", "Student", "Retired",
               "Entrepreneur", "Civil Servant", None]
ACCOUNT_TYPES = ["Savings", "Current", "Credit Card", "Fixed Deposit", "Business"]
CHANNELS = ["ATM", "Online", "Branch", "Mobile App", "POS"]
TXN_CATEGORIES = ["Groceries", "Utilities", "Salary", "Transfer", "Entertainment",
                   "Fuel", "Rent", "Insurance", "Withdrawal", "Deposit", "Online Shopping",
                   "Restaurant", "Medical", "Education", "Loan Repayment"]
LOAN_TYPES = ["Personal", "Home", "Vehicle", "Student", "Business"]
LOAN_STATUS = ["Active", "Closed", "Defaulted", "Restructured"]


def _rand_dates(start, end, n):
    start_u, end_u = start.timestamp(), end.timestamp()
    return pd.to_datetime(RNG.uniform(start_u, end_u, n), unit="s")


def gen_customers():
    n = N_CUSTOMERS
    ids = [f"CUST{100000+i}" for i in range(n)]
    first = RNG.choice(FIRST_NAMES, n)
    last = RNG.choice(LAST_NAMES, n)
    dob = _rand_dates(datetime(1945, 1, 1), datetime(2005, 12, 31), n)
    signup = _rand_dates(datetime(2015, 1, 1), datetime(2026, 8, 1), n)
    city = RNG.choice(CITIES, n)
    occupation = RNG.choice(OCCUPATIONS, n)
    income = RNG.lognormal(mean=10.2, sigma=0.6, size=n).round(2)  # monthly income, ZAR-ish
    credit_score = RNG.normal(650, 90, n).round(0)
    email = [f"{f.lower()}.{l.lower()}{RNG.integers(1,999)}@{RNG.choice(['gmail.com','yahoo.com','webmail.co.za','outlook.com'])}"
             for f, l in zip(first, last)]

    df = pd.DataFrame({
        "customer_id": ids,
        "first_name": first,
        "last_name": last,
        "date_of_birth": dob,
        "signup_date": signup,
        "city": city,
        "occupation": occupation,
        "monthly_income": income,
        "credit_score": credit_score,
        "email": email,
    })

    # --- inject messiness ---
    # inconsistent casing / whitespace on names & city
    idx = RNG.choice(n, int(n * 0.15), replace=False)
    df.loc[idx, "first_name"] = df.loc[idx, "first_name"].str.upper()
    idx = RNG.choice(n, int(n * 0.1), replace=False)
    df.loc[idx, "city"] = df.loc[idx, "city"] + "  "
    idx = RNG.choice(n, int(n * 0.08), replace=False)
    df.loc[idx, "city"] = df.loc[idx, "city"].str.lower()

    # missing values
    for col, frac in [("occupation", 0.06), ("monthly_income", 0.03),
                       ("credit_score", 0.04), ("email", 0.02), ("city", 0.02)]:
        idx = RNG.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # credit score out-of-range outliers / bad entries
    idx = RNG.choice(n, 15, replace=False)
    df.loc[idx, "credit_score"] = RNG.choice([-1, 0, 999, 1200], 15)

    # income outliers
    idx = RNG.choice(n, 10, replace=False)
    df.loc[idx, "monthly_income"] = df.loc[idx, "monthly_income"] * 200

    # exact duplicate customers (data entry re-submission)
    dup = df.sample(int(n * 0.02), random_state=1).copy()
    df = pd.concat([df, dup], ignore_index=True)

    # mixed date formats when exported (simulate by writing some as strings later)
    return df


def gen_accounts(customer_ids):
    n = N_ACCOUNTS
    ids = [f"ACC{200000+i}" for i in range(n)]
    cust = RNG.choice(customer_ids, n)
    # a few orphan accounts (FK integrity issue) referencing non-existent customers
    orphan_idx = RNG.choice(n, 12, replace=False)
    cust = cust.astype(object)
    for i in orphan_idx:
        cust[i] = f"CUST{999000 + RNG.integers(0, 999)}"

    acc_type = RNG.choice(ACCOUNT_TYPES, n, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    open_date = _rand_dates(datetime(2015, 1, 1), datetime(2026, 8, 1), n)
    balance = RNG.normal(25000, 40000, n).round(2)
    currency = RNG.choice(["ZAR", "zar", "ZAR "], n, p=[0.9, 0.05, 0.05])
    is_active = RNG.choice([1, 0], n, p=[0.92, 0.08])

    df = pd.DataFrame({
        "account_id": ids,
        "customer_id": cust,
        "account_type": acc_type,
        "open_date": open_date,
        "balance": balance,
        "currency": currency,
        "is_active": is_active,
    })

    # negative balances beyond realistic overdraft (bad data)
    idx = RNG.choice(n, 8, replace=False)
    df.loc[idx, "balance"] = -RNG.uniform(500000, 900000, 8)

    # missing balance
    idx = RNG.choice(n, int(n * 0.015), replace=False)
    df.loc[idx, "balance"] = np.nan

    return df


def gen_transactions(account_ids):
    n = N_TRANSACTIONS
    ids = [f"TXN{1000000+i}" for i in range(n)]
    acc = RNG.choice(account_ids, n)
    ts = _rand_dates(datetime(2024, 1, 1), datetime(2026, 8, 14), n)
    category = RNG.choice(TXN_CATEGORIES, n)
    channel = RNG.choice(CHANNELS, n)
    txn_type = RNG.choice(["Debit", "Credit"], n, p=[0.65, 0.35])

    # amount: log-normal, category-dependent scale
    base = RNG.lognormal(mean=5.5, sigma=1.1, size=n).round(2)

    # fraud label: rare, tends to be high amount + ATM/Online + odd hour
    fraud_prob = 0.006
    is_fraud = RNG.random(n) < fraud_prob
    amount = base.copy()
    amount[is_fraud] = amount[is_fraud] * RNG.uniform(8, 25, is_fraud.sum())

    df = pd.DataFrame({
        "transaction_id": ids,
        "account_id": acc,
        "timestamp": ts,
        "amount": amount,
        "txn_type": txn_type,
        "category": category,
        "channel": channel,
        "is_fraud": is_fraud.astype(int),
    })

    # duplicate transactions (double-submission at POS/ATM)
    dup = df.sample(int(n * 0.008), random_state=2).copy()
    df = pd.concat([df, dup], ignore_index=True)

    # missing amount / category
    idx = RNG.choice(len(df), int(len(df) * 0.01), replace=False)
    df.loc[idx, "amount"] = np.nan
    idx = RNG.choice(len(df), int(len(df) * 0.02), replace=False)
    df.loc[idx, "category"] = np.nan

    # negative amounts that shouldn't be negative (sign errors)
    idx = RNG.choice(len(df), int(len(df) * 0.005), replace=False)
    df.loc[idx, "amount"] = -df.loc[idx, "amount"].abs()

    # some timestamps exported as plain date strings only (mixed granularity) -
    # handled at ingestion; here we just shuffle row order to mimic raw export
    df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    return df


def gen_loans(customer_ids):
    n = N_LOANS
    ids = [f"LOAN{300000+i}" for i in range(n)]
    cust = RNG.choice(customer_ids, n)
    loan_type = RNG.choice(LOAN_TYPES, n)
    principal = RNG.lognormal(mean=10.5, sigma=0.9, size=n).round(2)
    interest_rate = RNG.uniform(7.5, 24.9, n).round(2)
    term_months = RNG.choice([12, 24, 36, 48, 60, 84, 120, 240], n)
    start_date = _rand_dates(datetime(2016, 1, 1), datetime(2026, 6, 1), n)
    status = RNG.choice(LOAN_STATUS, n, p=[0.55, 0.3, 0.1, 0.05])

    df = pd.DataFrame({
        "loan_id": ids,
        "customer_id": cust,
        "loan_type": loan_type,
        "principal": principal,
        "interest_rate": interest_rate,
        "term_months": term_months,
        "start_date": start_date,
        "status": status,
    })

    idx = RNG.choice(n, int(n * 0.02), replace=False)
    df.loc[idx, "interest_rate"] = np.nan
    return df


def main():
    print("Generating customers...")
    customers = gen_customers()
    print("Generating accounts...")
    accounts = gen_accounts(customers["customer_id"].unique())
    print("Generating transactions...")
    transactions = gen_transactions(accounts["account_id"].unique())
    print("Generating loans...")
    loans = gen_loans(customers["customer_id"].unique())

    # Write with a mix of date formats in a couple of files to simulate real
    # multi-source ingestion (e.g. an export from a legacy mainframe vs a
    # modern core banking system).
    customers_out = customers.copy()
    customers_out["date_of_birth"] = customers_out["date_of_birth"].dt.strftime("%Y-%m-%d")
    half = len(customers_out) // 2
    customers_out.loc[:half, "signup_date"] = pd.to_datetime(
        customers_out.loc[:half, "signup_date"]).dt.strftime("%d/%m/%Y")
    customers_out.loc[half:, "signup_date"] = pd.to_datetime(
        customers_out.loc[half:, "signup_date"]).dt.strftime("%Y-%m-%d")

    customers_out.to_csv(RAW_DIR / "customers.csv", index=False)
    accounts.to_csv(RAW_DIR / "accounts.csv", index=False)
    transactions.to_csv(RAW_DIR / "transactions.csv", index=False)
    loans.to_csv(RAW_DIR / "loans.csv", index=False)

    print(f"customers:    {len(customers):,} rows -> {RAW_DIR/'customers.csv'}")
    print(f"accounts:     {len(accounts):,} rows -> {RAW_DIR/'accounts.csv'}")
    print(f"transactions: {len(transactions):,} rows -> {RAW_DIR/'transactions.csv'}")
    print(f"loans:        {len(loans):,} rows -> {RAW_DIR/'loans.csv'}")


if __name__ == "__main__":
    main()
