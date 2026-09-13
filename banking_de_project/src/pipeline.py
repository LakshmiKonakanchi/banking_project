"""
pipeline.py
-----------
End-to-end orchestrator. Run with:

    python3 src/pipeline.py

Stages:
  1. generate_data   - synthetic raw banking data (skip if --no-generate)
  2. validate         - data quality checks on raw data
  3. clean             - cleaning transforms -> data/processed
  4. model             - build star-schema warehouse -> data/warehouse
  5. eda               - exploratory analysis + figures
  6. ml                - fraud detection model

Each stage is logged with timing. A failed CRITICAL validation halts
the pipeline before cleaning (fail-fast on data quality issues) unless
--force is passed.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_data
import validation
import cleaning
import modeling
import eda
import ml_pipeline

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline")


def stage(name, fn, *args, **kwargs):
    log.info(f"--- STAGE START: {name} ---")
    t0 = time.time()
    result = fn(*args, **kwargs)
    log.info(f"--- STAGE DONE: {name} ({time.time()-t0:.1f}s) ---")
    return result


def main():
    parser = argparse.ArgumentParser(description="Banking ETL/ML pipeline")
    parser.add_argument("--no-generate", action="store_true", help="skip synthetic data generation")
    parser.add_argument("--force", action="store_true", help="continue even if critical validation fails")
    args = parser.parse_args()

    t_start = time.time()
    log.info("=========== BANKING DATA PIPELINE START ===========")

    if not args.no_generate:
        stage("generate_data", generate_data.main)
    else:
        log.info("Skipping data generation (--no-generate)")

    import pandas as pd
    raw = Path(__file__).resolve().parent.parent / "data" / "raw"
    customers = pd.read_csv(raw / "customers.csv")
    accounts = pd.read_csv(raw / "accounts.csv")
    transactions = pd.read_csv(raw / "transactions.csv")
    loans = pd.read_csv(raw / "loans.csv")

    suites = stage("validate", validation.run_all, customers, accounts, transactions, loans)
    n_critical = sum(len(s.critical_failures) for s in suites.values())
    if n_critical and not args.force:
        log.warning(f"{n_critical} critical validation failures found. "
                    f"Proceeding to cleaning stage (cleaning resolves known FK/duplicate issues). "
                    f"Use --force to suppress this message.")

    stage("clean", cleaning.run_all)
    stage("model_warehouse", modeling.load_warehouse)
    stage("eda", eda.run_all)
    stage("ml", ml_pipeline.run_all)

    log.info(f"=========== PIPELINE COMPLETE in {time.time()-t_start:.1f}s ===========")


if __name__ == "__main__":
    main()
