"""
validation.py
--------------
A lightweight, dependency-free data validation framework in the spirit of
Great Expectations. Each check returns a structured result so we can build
a data quality report per dataset, and fail the pipeline loudly on
critical violations.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Any
import json
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CheckResult:
    check_name: str
    column: str | None
    passed: bool
    severity: str  # "critical" | "warning"
    detail: str
    n_failed: int = 0


@dataclass
class ValidationSuite:
    dataset_name: str
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult):
        self.results.append(result)

    @property
    def critical_failures(self):
        return [r for r in self.results if not r.passed and r.severity == "critical"]

    @property
    def warnings(self):
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    def summary(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "total_checks": len(self.results),
            "passed": sum(r.passed for r in self.results),
            "failed": sum(not r.passed for r in self.results),
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.results])

    def save(self):
        path = REPORT_DIR / f"validation_{self.dataset_name}.json"
        payload = {
            "summary": self.summary(),
            "checks": [r.__dict__ for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return path


# ---------------------------------------------------------------- checks --
def expect_column_exists(df, col) -> CheckResult:
    ok = col in df.columns
    return CheckResult("column_exists", col, ok, "critical",
                        f"'{col}' {'present' if ok else 'MISSING from dataset'}")


def expect_not_null(df, col, max_null_frac=0.0) -> CheckResult:
    n_null = df[col].isna().sum()
    frac = n_null / len(df) if len(df) else 0
    ok = frac <= max_null_frac
    sev = "critical" if max_null_frac == 0 else "warning"
    return CheckResult("not_null", col, ok, sev,
                        f"{n_null:,} nulls ({frac:.2%}), threshold {max_null_frac:.2%}", n_null)


def expect_unique(df, col) -> CheckResult:
    n_dupe = df[col].duplicated().sum()
    ok = n_dupe == 0
    return CheckResult("unique", col, ok, "critical",
                        f"{n_dupe:,} duplicate values", n_dupe)


def expect_no_full_row_duplicates(df) -> CheckResult:
    n_dupe = df.duplicated().sum()
    ok = n_dupe == 0
    return CheckResult("no_full_row_duplicates", None, ok, "warning",
                        f"{n_dupe:,} exact duplicate rows", n_dupe)


def expect_in_range(df, col, lo, hi) -> CheckResult:
    s = pd.to_numeric(df[col], errors="coerce")
    bad = ((s < lo) | (s > hi)).sum()
    ok = bad == 0
    return CheckResult("in_range", col, ok, "warning",
                        f"{bad:,} values outside [{lo}, {hi}]", bad)


def expect_in_set(df, col, allowed) -> CheckResult:
    bad = (~df[col].dropna().isin(allowed)).sum()
    ok = bad == 0
    return CheckResult("in_set", col, ok, "warning",
                        f"{bad:,} values outside allowed set {sorted(allowed)}", bad)


def expect_foreign_key(df, col, ref_values, ref_name) -> CheckResult:
    bad = (~df[col].isin(ref_values)).sum()
    ok = bad == 0
    return CheckResult("foreign_key", col, ok, "critical",
                        f"{bad:,} rows reference non-existent {ref_name}", bad)


def expect_positive(df, col) -> CheckResult:
    s = pd.to_numeric(df[col], errors="coerce")
    bad = (s <= 0).sum()
    ok = bad == 0
    return CheckResult("positive", col, ok, "warning", f"{bad:,} non-positive values", bad)


def expect_dtype_parseable_datetime(df, col) -> CheckResult:
    parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
    bad = parsed.isna().sum() - df[col].isna().sum()
    ok = bad == 0
    return CheckResult("parseable_datetime", col, ok, "warning",
                        f"{bad:,} values could not be parsed as dates", int(max(bad, 0)))


# --------------------------------------------------------------- runners --
def validate_customers(df: pd.DataFrame) -> ValidationSuite:
    suite = ValidationSuite("customers")
    for col in ["customer_id", "first_name", "last_name", "date_of_birth", "monthly_income"]:
        suite.add(expect_column_exists(df, col))
    suite.add(expect_unique(df, "customer_id"))
    suite.add(expect_not_null(df, "customer_id"))
    suite.add(expect_not_null(df, "monthly_income", max_null_frac=0.05))
    suite.add(expect_not_null(df, "credit_score", max_null_frac=0.06))
    suite.add(expect_in_range(df, "credit_score", 300, 850))
    suite.add(expect_positive(df, "monthly_income"))
    suite.add(expect_dtype_parseable_datetime(df, "date_of_birth"))
    suite.add(expect_dtype_parseable_datetime(df, "signup_date"))
    suite.add(expect_no_full_row_duplicates(df))
    return suite


def validate_accounts(df: pd.DataFrame, valid_customer_ids) -> ValidationSuite:
    suite = ValidationSuite("accounts")
    for col in ["account_id", "customer_id", "account_type", "balance"]:
        suite.add(expect_column_exists(df, col))
    suite.add(expect_unique(df, "account_id"))
    suite.add(expect_foreign_key(df, "customer_id", valid_customer_ids, "customers.customer_id"))
    suite.add(expect_in_set(df, "account_type",
                             {"Savings", "Current", "Credit Card", "Fixed Deposit", "Business"}))
    suite.add(expect_not_null(df, "balance", max_null_frac=0.02))
    suite.add(expect_no_full_row_duplicates(df))
    return suite


def validate_transactions(df: pd.DataFrame, valid_account_ids) -> ValidationSuite:
    suite = ValidationSuite("transactions")
    for col in ["transaction_id", "account_id", "timestamp", "amount"]:
        suite.add(expect_column_exists(df, col))
    suite.add(expect_unique(df, "transaction_id"))
    suite.add(expect_foreign_key(df, "account_id", valid_account_ids, "accounts.account_id"))
    suite.add(expect_not_null(df, "amount", max_null_frac=0.02))
    suite.add(expect_in_set(df, "txn_type", {"Debit", "Credit"}))
    suite.add(expect_dtype_parseable_datetime(df, "timestamp"))
    return suite


def validate_loans(df: pd.DataFrame, valid_customer_ids) -> ValidationSuite:
    suite = ValidationSuite("loans")
    for col in ["loan_id", "customer_id", "principal", "interest_rate"]:
        suite.add(expect_column_exists(df, col))
    suite.add(expect_unique(df, "loan_id"))
    suite.add(expect_foreign_key(df, "customer_id", valid_customer_ids, "customers.customer_id"))
    suite.add(expect_positive(df, "principal"))
    suite.add(expect_in_range(df, "interest_rate", 0, 40))
    suite.add(expect_in_set(df, "status", {"Active", "Closed", "Defaulted", "Restructured"}))
    return suite


def run_all(customers, accounts, transactions, loans) -> dict:
    suites = {
        "customers": validate_customers(customers),
        "accounts": validate_accounts(accounts, set(customers["customer_id"])),
        "transactions": validate_transactions(transactions, set(accounts["account_id"])),
        "loans": validate_loans(loans, set(customers["customer_id"])),
    }
    for suite in suites.values():
        path = suite.save()
        s = suite.summary()
        status = "FAIL" if s["critical_failures"] else "OK"
        print(f"[{status}] {s['dataset']:<14} passed {s['passed']}/{s['total_checks']} "
              f"checks | critical={s['critical_failures']} warnings={s['warnings']} -> {path.name}")
    return suites


if __name__ == "__main__":
    raw = Path(__file__).resolve().parent.parent / "data" / "raw"
    customers = pd.read_csv(raw / "customers.csv")
    accounts = pd.read_csv(raw / "accounts.csv")
    transactions = pd.read_csv(raw / "transactions.csv")
    loans = pd.read_csv(raw / "loans.csv")
    run_all(customers, accounts, transactions, loans)
