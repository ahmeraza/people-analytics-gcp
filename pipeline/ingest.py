"""
pipeline/ingest.py

Phase 2 — Data Ingestion
Uploads the IBM Watson Attrition CSV to Cloud Storage and loads it into BigQuery.
Then runs SQL transforms to produce a clean training table.

Usage:
    python pipeline/ingest.py
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery, storage

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
PROJECT_ID = os.environ["GCP_PROJECT_ID"]
BUCKET_NAME = os.environ["GCS_BUCKET"]
DATASET_ID = os.environ["BQ_DATASET"]
REGION = os.environ.get("GCP_REGION", "us-central1")

RAW_CSV_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
GCS_BLOB_NAME = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
RAW_TABLE = f"{PROJECT_ID}.{DATASET_ID}.attrition_raw"
CLEAN_TABLE = f"{PROJECT_ID}.{DATASET_ID}.attrition_clean"


# ── Step 1: Upload CSV to Cloud Storage ─────────────────────────────────────
def upload_to_gcs() -> str:
    """Upload the raw CSV to GCS. Returns the GCS URI."""
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_CSV_PATH} not found. "
            "Download via: kaggle datasets download "
            "-d pavansubhasht/ibm-hr-analytics-attrition-dataset -p data/ --unzip"
        )

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(GCS_BLOB_NAME)

    log.info("Uploading %s → gs://%s/%s", RAW_CSV_PATH, BUCKET_NAME, GCS_BLOB_NAME)
    blob.upload_from_filename(str(RAW_CSV_PATH))

    uri = f"gs://{BUCKET_NAME}/{GCS_BLOB_NAME}"
    log.info("Upload complete: %s", uri)
    return uri


# ── Step 2: Load raw CSV into BigQuery ──────────────────────────────────────
def load_to_bigquery(gcs_uri: str) -> None:
    """Create the raw BigQuery table from the GCS CSV."""
    client = bigquery.Client(project=PROJECT_ID)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    log.info("Loading %s → %s", gcs_uri, RAW_TABLE)
    load_job = client.load_table_from_uri(
        gcs_uri, RAW_TABLE, job_config=job_config
    )
    load_job.result()  # Wait for completion

    table = client.get_table(RAW_TABLE)
    log.info(
        "Loaded %d rows, %d columns into %s",
        table.num_rows, len(table.schema), RAW_TABLE,
    )


# ── Step 3: SQL transforms → clean training table ───────────────────────────
TRANSFORM_SQL = f"""
CREATE OR REPLACE TABLE `{CLEAN_TABLE}` AS

WITH base AS (
  SELECT
    Age,
    MonthlyIncome,
    YearsAtCompany,
    YearsSinceLastPromotion,
    YearsWithCurrManager,
    NumCompaniesWorked,
    DistanceFromHome,
    PercentSalaryHike,
    TrainingTimesLastYear,
    TotalWorkingYears,

    -- Numeric encodings for categorical features
    CASE CAST(OverTime AS STRING) WHEN 'Yes' THEN 1 ELSE 0 END AS overtime,

    CASE Department
      WHEN 'Sales'                THEN 1
      WHEN 'Research & Development' THEN 2
      WHEN 'Human Resources'      THEN 3
      ELSE 0
    END AS dept_enc,

    CASE JobRole
      WHEN 'Sales Executive'         THEN 1
      WHEN 'Research Scientist'      THEN 2
      WHEN 'Laboratory Technician'   THEN 3
      WHEN 'Manufacturing Director'  THEN 4
      WHEN 'Healthcare Representative' THEN 5
      WHEN 'Manager'                 THEN 6
      WHEN 'Sales Representative'    THEN 7
      WHEN 'Research Director'       THEN 8
      WHEN 'Human Resources'         THEN 9
      ELSE 0
    END AS job_role_enc,

    CASE MaritalStatus
      WHEN 'Single'   THEN 1
      WHEN 'Married'  THEN 2
      WHEN 'Divorced' THEN 3
      ELSE 0
    END AS marital_enc,

    CASE EducationField
      WHEN 'Life Sciences'     THEN 1
      WHEN 'Medical'           THEN 2
      WHEN 'Marketing'         THEN 3
      WHEN 'Technical Degree'  THEN 4
      WHEN 'Human Resources'   THEN 5
      WHEN 'Other'             THEN 6
      ELSE 0
    END AS edu_field_enc,

    JobSatisfaction,
    WorkLifeBalance,
    EnvironmentSatisfaction,
    RelationshipSatisfaction,
    JobInvolvement,
    PerformanceRating,
    Education,
    JobLevel,
    StockOptionLevel,

    -- FAANG-grade engineered features
    (MonthlyIncome / AVG(MonthlyIncome) OVER(
      PARTITION BY Department, JobRole)) AS internal_equity_ratio,

    (YearsAtCompany / (YearsSinceLastPromotion + 1)) AS promotion_stagnation_index,

    (YearsWithCurrManager / (YearsAtCompany + 1)) AS manager_dependency_score,

    -- Target label: 1 = attrition (positive class)
    IF(CAST(Attrition AS BOOL) = TRUE, 1, 0) AS label,

    -- Deterministic split: 80% train, 20% eval
    -- FARM_FINGERPRINT gives a consistent, reproducible split
    IF(
      MOD(ABS(FARM_FINGERPRINT(CAST(EmployeeNumber AS STRING))), 10) < 8,
      'TRAIN',
      'EVAL'
    ) AS data_split

  FROM `{RAW_TABLE}`
  -- Exclude rows where derived constraints don't hold
  WHERE EmployeeCount = 1          -- sanity check (always 1 in this dataset)
    AND StandardHours = 80         -- sanity check (always 80)
)

SELECT * FROM base
"""


def run_transforms() -> None:
    """Execute the SQL transforms to produce the clean training table."""
    client = bigquery.Client(project=PROJECT_ID)

    log.info("Running SQL transforms → %s", CLEAN_TABLE)
    query_job = client.query(TRANSFORM_SQL)
    query_job.result()

    table = client.get_table(CLEAN_TABLE)
    log.info(
        "Clean table created: %d rows, %d columns",
        table.num_rows, len(table.schema),
    )

    # Log split distribution for validation
    split_sql = f"""
        SELECT data_split, label, COUNT(*) AS n
        FROM `{CLEAN_TABLE}`
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    rows = list(client.query(split_sql).result())
    log.info("Split distribution:")
    for row in rows:
        log.info("  %s | label=%s | n=%d", row.data_split, row.label, row.n)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 55)
    log.info("Phase 2: Data Ingestion")
    log.info("Project: %s | Bucket: %s | Dataset: %s", PROJECT_ID, BUCKET_NAME, DATASET_ID)
    log.info("=" * 55)

    gcs_uri = upload_to_gcs()
    load_to_bigquery(gcs_uri)
    run_transforms()

    log.info("✅ Ingestion complete. Next: python pipeline/train_runner.py")


if __name__ == "__main__":
    main()
