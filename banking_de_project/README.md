# Banking Data Engineering Platform

A senior-level, end-to-end data engineering project on synthetic banking data:
raw ingestion → **data validation** → **data cleaning** → **dimensional modeling**
(star schema) → **EDA** → **ML** (fraud detection). Pure Python + pandas/numpy/
scikit-learn/matplotlib/seaborn/sqlite3 — no external services required.

## Architecture

```
generate_data.py  ──▶  data/raw/*.csv  (messy, multi-format, "as if from prod")
        │
        ▼
validation.py     ──▶  outputs/reports/validation_*.json   (data quality gate)
        │
        ▼
cleaning.py       ──▶  data/processed/*_clean.csv          (audited transforms)
        │
        ▼
modeling.py       ──▶  data/warehouse/banking_dw.sqlite    (star schema)
        │
        ├──▶ eda.py         ──▶ outputs/figures/*.png, outputs/reports/eda_summary.txt
        └──▶ ml_pipeline.py ──▶ outputs/models/*.pkl, outputs/figures/09-11*.png
```

Orchestrated end-to-end by `src/pipeline.py`.

## Datasets (synthetic, generated on each run)

| Table | Rows | Description |
|---|---|---|
| `customers` | ~5,000 | demographics, income, credit score |
| `accounts` | ~7,000 | savings/current/credit card/fixed deposit/business |
| `transactions` | ~120,000 | 2024–2026, category/channel/hour, `is_fraud` label |
| `loans` | ~2,500 | personal/home/vehicle/student/business, status, rate |

Data is generated **intentionally messy** — duplicate keys, orphan foreign keys,
mixed date formats, inconsistent casing, sign errors, out-of-range values,
outliers — so the validation/cleaning stages have real work to do, the same
way you'd encounter a legacy core-banking export in practice.

## Star schema (data modeling)

```
        dim_customer ──┐
                        ├─▶ dim_account ──▶ fact_transactions ◀── dim_date
        dim_customer ──────────────────▶ fact_loans
```

Loaded into SQLite (`data/warehouse/banking_dw.sqlite`) with indexes on the
common join/filter columns. Query it directly:

```bash
sqlite3 data/warehouse/banking_dw.sqlite \
  "SELECT category, COUNT(*), ROUND(SUM(amount),2) FROM fact_transactions GROUP BY category ORDER BY 3 DESC;"
```

## Data validation

`src/validation.py` implements a small Great-Expectations-style framework
(`expect_not_null`, `expect_unique`, `expect_foreign_key`, `expect_in_range`,
`expect_in_set`, `expect_dtype_parseable_datetime`, ...). Every check produces
a `CheckResult` (severity: critical/warning) rolled up into a JSON report per
dataset in `outputs/reports/`. Critical failures (broken keys, missing IDs)
are designed to be pipeline-blocking in a real production setup; here they're
logged and then resolved explicitly in the cleaning stage so you can see the
before/after.

## Data cleaning

`src/cleaning.py` produces an **audit log** for every transform (exact counts
of what was dropped/imputed/corrected and why) — e.g.:

```
[transactions]
  - Dropped 960 duplicate transaction_id rows
  - Removed 201 transactions with orphan account_id
  - Corrected 590 negative amount sign errors (took absolute value)
  - Imputed 1,198 missing amounts with category-level median
```

Techniques used: dedup on business key, FK-based orphan removal, winsorizing
(not deleting) income outliers at the 99.5th percentile, median imputation
(overall and group-wise), mixed-format date parsing, categorical
standardization.

## EDA

`src/eda.py` generates 8 figures into `outputs/figures/`: income/credit
distributions, transaction volume by category, daily volume trend with a
14-day rolling average, channel×hour heatmap, account type mix, fraud
overview, a correlation heatmap of financial attributes, and loan portfolio
status/rates. A text summary lands in `outputs/reports/eda_summary.txt`.

## ML — fraud detection

`src/ml_pipeline.py` builds a feature table by joining the fact/dim tables
(transaction + account + customer + behavioral aggregates), then trains a
`RandomForestClassifier` inside an sklearn `Pipeline` (median/most-frequent
imputation → scaling/one-hot → model), with `class_weight="balanced_subsample"`
to handle the realistic ~0.5% fraud rate.

Evaluated with ROC-AUC, PR-AUC, and per-class precision/recall/F1 (accuracy
is meaningless on this imbalance). Typical run:

```
ROC-AUC: 0.92   |   PR-AUC: 0.18   |   F1 (fraud class): 0.22
```

These are realistic, non-inflated numbers for a hard, imbalanced problem with
tabular features only — not a cherry-picked demo. Outputs: confusion matrix,
ROC/PR curves, feature importance plot, and the serialized pipeline
(`outputs/models/fraud_model_random_forest.pkl`).

## Running it

```bash
pip install -r requirements.txt

# full pipeline, fresh synthetic data
python3 src/pipeline.py

# reuse existing data/raw/*.csv, skip regeneration
python3 src/pipeline.py --no-generate

# run any single stage directly
python3 src/generate_data.py
python3 src/validation.py
python3 src/cleaning.py
python3 src/modeling.py
python3 src/eda.py
python3 src/ml_pipeline.py

# unit tests (stdlib unittest — no pytest dependency)
python3 -m unittest discover -s tests -v
```

## Project layout

```
src/
  generate_data.py   synthetic raw data with injected quality issues
  validation.py       lightweight data-quality check framework
  cleaning.py          logged, auditable cleaning transforms
  modeling.py           star-schema DDL + warehouse load (SQLite)
  eda.py                 exploratory analysis + figures
  ml_pipeline.py          feature engineering + fraud model + evaluation
  pipeline.py               orchestrator (stage timing, logging, fail-fast hook)
tests/
  test_pipeline.py    unit tests for cleaning + validation logic
data/{raw,processed,warehouse}
outputs/{figures,reports,models}
logs/pipeline.log
```

## Design notes / trade-offs

- **SQLite over Postgres/Snowflake**: keeps the project runnable anywhere
  with zero infra, while still exercising real dimensional modeling (star
  schema, surrogate keys, indexes, FK-shaped joins). Swapping to another
  RDBMS is a matter of changing the connection string — the DDL and load
  logic are standard SQL.
- **CSV over Parquet** for `data/processed/`: this environment has no
  `pyarrow`; swap `to_csv`/`read_csv` for `to_parquet`/`read_parquet` in
  `cleaning.py`/`modeling.py` if you have it available.
- **Winsorize, don't drop** income outliers — preserves sample size and
  avoids survivorship bias, standard practice for financial data.
- **class_weight over SMOTE** for fraud imbalance — no `imbalanced-learn`
  dependency, and `class_weight="balanced_subsample"` is a solid baseline for
  tree ensembles.
- **Behavioral features computed pre-split** (`customer_avg_amount`,
  `customer_txn_count`) are historical profile aggregates, not
  label-derived — no leakage from the train/test split.
