"""
pipeline/predict.py

Batch prediction using BigQuery ML (no Vertex AI endpoint needed).
Useful for generating attrition scores for all employees at once —
e.g., for a monthly HR report.

Output: BigQuery table + optional GCS CSV export

Usage:
    python pipeline/predict.py
    python pipeline/predict.py --export-gcs
"""

import argparse
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
DATASET_ID = os.environ["BQ_DATASET"]
BUCKET_NAME = os.environ["GCS_BUCKET"]
REGION = os.environ.get("GCP_REGION", "us-central1")
THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.45"))

CLIENT = bigquery.Client(project=PROJECT_ID)
RUN_DATE = datetime.utcnow().strftime("%Y%m%d")
OUTPUT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.batch_predictions_{RUN_DATE}"


def run_batch_prediction() -> int:
    """Run ML.PREDICT on the full clean table and save results."""
    sql = f"""
    CREATE OR REPLACE TABLE `{OUTPUT_TABLE}` AS

    WITH raw_predictions AS (
      SELECT
        predicted_label,
        predicted_label_probs,
        -- Reconstruct key employee fields for context
        Age,
        MonthlyIncome,
        YearsAtCompany,
        overtime,
        dept_enc,
        JobSatisfaction,
        WorkLifeBalance
      FROM ML.PREDICT(
        MODEL `{PROJECT_ID}.{DATASET_ID}.model_boosted_v1`,
        (
          SELECT * EXCEPT(data_split, label)
          FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean`
        ),
        STRUCT({THRESHOLD} AS threshold)
      )
    )

    SELECT
      predicted_label AS predicted_attrition,
      -- Extract probability for class 1 (attrition = Yes)
      (SELECT prob FROM UNNEST(predicted_label_probs) WHERE label = 1) AS attrition_probability,
      CASE
        WHEN (SELECT prob FROM UNNEST(predicted_label_probs) WHERE label = 1) >= 0.7
          THEN 'Critical'
        WHEN (SELECT prob FROM UNNEST(predicted_label_probs) WHERE label = 1) >= {THRESHOLD}
          THEN 'High'
        WHEN (SELECT prob FROM UNNEST(predicted_label_probs) WHERE label = 1) >= 0.25
          THEN 'Medium'
        ELSE 'Low'
      END AS risk_tier,
      Age,
      MonthlyIncome,
      YearsAtCompany,
      overtime,
      dept_enc,
      JobSatisfaction,
      WorkLifeBalance,
      CURRENT_TIMESTAMP() AS prediction_timestamp,
      '{RUN_DATE}' AS run_date
    FROM raw_predictions
    ORDER BY attrition_probability DESC
    """

    log.info("Running batch prediction → %s", OUTPUT_TABLE)
    job = CLIENT.query(sql)
    job.result()

    # Count results by risk tier
    summary_sql = f"""
    SELECT risk_tier, COUNT(*) AS n, ROUND(AVG(attrition_probability), 3) AS avg_prob
    FROM `{OUTPUT_TABLE}`
    GROUP BY risk_tier
    ORDER BY avg_prob DESC
    """
    rows = list(CLIENT.query(summary_sql).result())
    total = sum(r.n for r in rows)
    log.info("Batch prediction complete — %d employees scored:", total)
    for row in rows:
        pct = row.n / total * 100
        log.info("  %-10s %4d employees (%4.1f%%)  avg_prob=%.3f", row.risk_tier, row.n, pct, row.avg_prob)

    return total


def export_to_gcs() -> str:
    """Export predictions to GCS as CSV for downstream HR tools."""
    gcs_uri = f"gs://{BUCKET_NAME}/batch-predictions/{RUN_DATE}/predictions_*.csv"

    extract_config = bigquery.ExtractJobConfig(
        destination_format=bigquery.DestinationFormat.CSV,
        print_header=True,
    )

    log.info("Exporting predictions to %s", gcs_uri)
    job = CLIENT.extract_table(OUTPUT_TABLE, gcs_uri, job_config=extract_config)
    job.result()
    log.info("✅ Export complete: %s", gcs_uri)
    return gcs_uri


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch attrition predictions")
    parser.add_argument("--export-gcs", action="store_true", help="Export results to GCS CSV")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("Batch Prediction — %s", RUN_DATE)
    log.info("Threshold: %.2f  |  Model: model_boosted_v1", THRESHOLD)
    log.info("=" * 55)

    total = run_batch_prediction()

    if args.export_gcs:
        export_to_gcs()

    log.info("Results available at: %s.%s.batch_predictions_%s", PROJECT_ID, DATASET_ID, RUN_DATE)
    log.info("Query in BigQuery: SELECT * FROM `%s` WHERE risk_tier IN ('Critical','High') ORDER BY attrition_probability DESC", OUTPUT_TABLE)


if __name__ == "__main__":
    main()
