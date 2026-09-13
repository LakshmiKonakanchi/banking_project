"""
test_pipeline.py
-----------------
Unit tests using stdlib unittest (no network access available for pytest).
Run with:  python3 -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cleaning
import validation


class TestCleaningCustomers(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "customer_id": ["C1", "C2", "C2", "C3"],
            "first_name": ["  bob", "ALICE", "ALICE", "Zoe"],
            "last_name": ["smith", "JONES", "JONES", "Lee"],
            "date_of_birth": ["1990-01-01", "1985-05-05", "1985-05-05", "2000-12-31"],
            "signup_date": ["2020-01-01", "2021-01-01", "2021-01-01", "2022-01-01"],
            "city": ["cape town ", "Durban", "Durban", None],
            "occupation": ["Engineer", None, None, "Teacher"],
            "monthly_income": [30000.0, np.nan, np.nan, 25000.0],
            "credit_score": [650, 900, 900, 300],  # 900 invalid
            "email": ["Bob@X.com", "alice@x.com", "alice@x.com", "zoe@x.com"],
        })

    def test_deduplicates_customer_id(self):
        clean, log = cleaning.clean_customers(self.df)
        self.assertEqual(clean["customer_id"].duplicated().sum(), 0)

    def test_invalid_credit_score_corrected(self):
        clean, log = cleaning.clean_customers(self.df)
        self.assertTrue((clean["credit_score"] <= 850).all())
        self.assertTrue((clean["credit_score"] >= 300).all())

    def test_income_imputed_no_nulls(self):
        clean, log = cleaning.clean_customers(self.df)
        self.assertEqual(clean["monthly_income"].isna().sum(), 0)

    def test_name_casing_standardized(self):
        clean, log = cleaning.clean_customers(self.df)
        self.assertEqual(clean.loc[clean["customer_id"] == "C1", "first_name"].iloc[0], "Bob")


class TestCleaningTransactions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "transaction_id": ["T1", "T2", "T2", "T3"],
            "account_id": ["A1", "A1", "A1", "A_ORPHAN"],
            "timestamp": ["2024-01-01 10:00:00"] * 4,
            "amount": [100.0, -50.0, -50.0, 200.0],
            "txn_type": ["Debit", "Debit", "Debit", "Credit"],
            "category": ["Groceries", None, None, "Salary"],
            "channel": ["ATM", "Online", "Online", "Branch"],
            "is_fraud": [0, 0, 0, 0],
        })

    def test_removes_orphan_accounts(self):
        clean, log = cleaning.clean_transactions(self.df, valid_account_ids={"A1"})
        self.assertNotIn("A_ORPHAN", clean["account_id"].values)

    def test_fixes_negative_amounts(self):
        clean, log = cleaning.clean_transactions(self.df, valid_account_ids={"A1"})
        self.assertTrue((clean["amount"] >= 0).all())

    def test_deduplicates_transaction_id(self):
        clean, log = cleaning.clean_transactions(self.df, valid_account_ids={"A1"})
        self.assertEqual(clean["transaction_id"].duplicated().sum(), 0)


class TestValidationChecks(unittest.TestCase):
    def test_expect_unique_detects_dupes(self):
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        result = validation.expect_unique(df, "id")
        self.assertFalse(result.passed)
        self.assertEqual(result.n_failed, 1)

    def test_expect_not_null_passes_clean_column(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = validation.expect_not_null(df, "id")
        self.assertTrue(result.passed)

    def test_expect_in_range_flags_outliers(self):
        df = pd.DataFrame({"score": [300, 900, 500]})
        result = validation.expect_in_range(df, "score", 300, 850)
        self.assertFalse(result.passed)
        self.assertEqual(result.n_failed, 1)

    def test_expect_foreign_key_flags_orphans(self):
        df = pd.DataFrame({"cust_id": ["A", "B", "Z"]})
        result = validation.expect_foreign_key(df, "cust_id", {"A", "B"}, "customers")
        self.assertFalse(result.passed)
        self.assertEqual(result.n_failed, 1)


if __name__ == "__main__":
    unittest.main()
