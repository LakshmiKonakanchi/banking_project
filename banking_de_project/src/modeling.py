"""
modeling.py
-----------
Builds a dimensional (star schema) data warehouse from the cleaned data:

    dim_customer  --\
    dim_account    --> fact_transactions
    dim_date      --/
    fact_loans -- dim_customer

Loaded into SQLite for portability (no external DB engine required).
This is the "serving layer" the EDA/ML/BI steps read from, decoupled
from raw ingestion.
"""
from __future__ import annotations
import pandas as pd
import sqlite3
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
WAREHOUSE_DIR = Path(__file__).resolve().parent.parent / "data" / "warehouse"
WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = WAREHOUSE_DIR / "banking_dw.sqlite"

DDL = """
DROP TABLE IF EXISTS dim_customer;
CREATE TABLE dim_customer (
    customer_key    INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     TEXT UNIQUE NOT NULL,
    first_name      TEXT,
    last_name       TEXT,
    age             REAL,
    city            TEXT,
    occupation      TEXT,
    monthly_income  REAL,
    credit_score    REAL,
    signup_date     TEXT
);

DROP TABLE IF EXISTS dim_account;
CREATE TABLE dim_account (
    account_key     INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id      TEXT UNIQUE NOT NULL,
    customer_id     TEXT NOT NULL,
    account_type    TEXT,
    currency        TEXT,
    open_date       TEXT,
    is_active       INTEGER,
    FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id)
);

DROP TABLE IF EXISTS dim_date;
CREATE TABLE dim_date (
    date_key    TEXT PRIMARY KEY,   -- YYYY-MM-DD
    year        INTEGER,
    month       INTEGER,
    day         INTEGER,
    day_of_week TEXT,
    is_weekend  INTEGER,
    quarter     INTEGER
);

DROP TABLE IF EXISTS fact_transactions;
CREATE TABLE fact_transactions (
    transaction_key INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  TEXT UNIQUE NOT NULL,
    account_id      TEXT NOT NULL,
    date_key        TEXT NOT NULL,
    amount          REAL,
    txn_type        TEXT,
    category        TEXT,
    channel         TEXT,
    hour            INTEGER,
    is_fraud        INTEGER,
    FOREIGN KEY (account_id) REFERENCES dim_account(account_id),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

DROP TABLE IF EXISTS fact_loans;
CREATE TABLE fact_loans (
    loan_key        INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_id         TEXT UNIQUE NOT NULL,
    customer_id     TEXT NOT NULL,
    loan_type       TEXT,
    principal       REAL,
    interest_rate   REAL,
    term_months     INTEGER,
    monthly_payment REAL,
    status          TEXT,
    start_date      TEXT,
    FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id)
);

CREATE INDEX idx_fact_txn_account ON fact_transactions(account_id);
CREATE INDEX idx_fact_txn_date ON fact_transactions(date_key);
CREATE INDEX idx_fact_txn_fraud ON fact_transactions(is_fraud);
CREATE INDEX idx_dim_account_customer ON dim_account(customer_id);
CREATE INDEX idx_fact_loans_customer ON fact_loans(customer_id);
"""


def build_dim_date(min_date, max_date) -> pd.DataFrame:
    dates = pd.date_range(min_date, max_date, freq="D")
    return pd.DataFrame({
        "date_key": dates.strftime("%Y-%m-%d"),
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "day_of_week": dates.day_name(),
        "is_weekend": dates.dayofweek.isin([5, 6]).astype(int),
        "quarter": dates.quarter,
    })


def load_warehouse():
    customers = pd.read_csv(PROCESSED_DIR / "customers_clean.csv")
    accounts = pd.read_csv(PROCESSED_DIR / "accounts_clean.csv")
    transactions = pd.read_csv(PROCESSED_DIR / "transactions_clean.csv", parse_dates=["timestamp"])
    loans = pd.read_csv(PROCESSED_DIR / "loans_clean.csv")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(DDL)

    dim_customer = customers[["customer_id", "first_name", "last_name", "age", "city",
                               "occupation", "monthly_income", "credit_score", "signup_date"]]
    dim_customer.to_sql("dim_customer", conn, if_exists="append", index=False)

    dim_account = accounts[["account_id", "customer_id", "account_type", "currency",
                             "open_date", "is_active"]]
    dim_account.to_sql("dim_account", conn, if_exists="append", index=False)

    dim_date = build_dim_date(transactions["timestamp"].min(), transactions["timestamp"].max())
    dim_date.to_sql("dim_date", conn, if_exists="append", index=False)

    fact_txn = transactions.copy()
    fact_txn["date_key"] = fact_txn["timestamp"].dt.strftime("%Y-%m-%d")
    fact_txn = fact_txn[["transaction_id", "account_id", "date_key", "amount", "txn_type",
                          "category", "channel", "hour", "is_fraud"]]
    fact_txn.to_sql("fact_transactions", conn, if_exists="append", index=False)

    fact_loans = loans[["loan_id", "customer_id", "loan_type", "principal", "interest_rate",
                         "term_months", "monthly_payment", "status", "start_date"]]
    fact_loans.to_sql("fact_loans", conn, if_exists="append", index=False)

    conn.commit()

    # sanity: row counts + a sample analytical join
    print("=== WAREHOUSE LOAD SUMMARY ===")
    for tbl in ["dim_customer", "dim_account", "dim_date", "fact_transactions", "fact_loans"]:
        n = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:<20} {n:,} rows")

    print("\nSample star-schema join (top 5 customers by total transaction volume):")
    q = """
    SELECT c.customer_id, c.first_name, c.last_name, c.city,
           COUNT(*) AS n_txns, ROUND(SUM(f.amount), 2) AS total_amount
    FROM fact_transactions f
    JOIN dim_account a ON f.account_id = a.account_id
    JOIN dim_customer c ON a.customer_id = c.customer_id
    GROUP BY c.customer_id
    ORDER BY total_amount DESC
    LIMIT 5;
    """
    print(pd.read_sql(q, conn).to_string(index=False))

    conn.close()
    print(f"\nWarehouse written to {DB_PATH}")


if __name__ == "__main__":
    load_warehouse()
