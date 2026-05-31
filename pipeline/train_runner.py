"""
pipeline/train_runner.py

Phase 3 — Model Training
Executes BQML training statements, logs evaluation metrics, and saves results.

Usage:
    python pipeline/train_runner.py
"""

import json
import logging
import os
from pathlib import Path

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
REGION = os.environ.get("GCP_REGION", "us-central1")

CLIENT = bigquery.Client(project=PROJECT_ID)
RESULTS_DIR = Path("pipeline/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_sql(sql: str, description: str) -> list[bigquery.Row]:
    """Execute a BigQuery SQL statement and return rows."""
    log.info("Running: %s", description)
    job = CLIENT.query(sql)
    rows = list(job.result())
    log.info("  → %d rows returned", len(rows))
    return rows


def train_model(model_type: str) -> None:
    """Train a single BQML model."""
    sql_path = Path("pipeline/train.sql")
    raw_sql = sql_path.read_text()

    # Split on the separator comments to get individual statements
    # Each statement ends before the next `-- ──` separator
    statements = [s.strip() for s in raw_sql.split("\n\n\n") if s.strip()]

    # Map model_type to statement index
    model_statements = {
        "logistic": 0,
        "boosted": 1,
    }

    idx = model_statements.get(model_type)
    if idx is None:
        raise ValueError(f"Unknown model type: {model_type}")

    sql = statements[idx].format(PROJECT_ID=PROJECT_ID, DATASET_ID=DATASET_ID)
    log.info("Training %s model — this may take a few minutes...", model_type)
    job = CLIENT.query(sql)
    job.result()
    log.info("✅ %s model training complete.", model_type)


def evaluate_all_models() -> dict:
    """Evaluate both models and return comparison metrics."""
    sql = f"""
    SELECT 'Logistic Regression' AS model, roc_auc, precision, recall, f1_score, log_loss, accuracy
    FROM ML.EVALUATE(
      MODEL `{PROJECT_ID}.{DATASET_ID}.model_logistic_v1`,
      (SELECT * EXCEPT(data_split) FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean` WHERE data_split = 'EVAL')
    )
    UNION ALL
    SELECT 'Boosted Trees' AS model, roc_auc, precision, recall, f1_score, log_loss, accuracy
    FROM ML.EVALUATE(
      MODEL `{PROJECT_ID}.{DATASET_ID}.model_boosted_v1`,
      (SELECT * EXCEPT(data_split) FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean` WHERE data_split = 'EVAL')
    )
    ORDER BY roc_auc DESC
    """

    rows = run_sql(sql, "Evaluating both models")
    metrics = {}
    for row in rows:
        model_name = row["model"]
        metrics[model_name] = {
            "roc_auc": round(row["roc_auc"], 4),
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "f1_score": round(row["f1_score"], 4),
            "log_loss": round(row["log_loss"], 4),
            "accuracy": round(row["accuracy"], 4),
        }
        log.info(
            "  %s: AUC=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f",
            model_name,
            metrics[model_name]["roc_auc"],
            metrics[model_name]["f1_score"],
            metrics[model_name]["precision"],
            metrics[model_name]["recall"],
        )
    return metrics


def get_feature_importance() -> list[dict]:
    """Retrieve and log feature importance from the boosted model."""
    sql = f"""
    SELECT feature, importance_gain, importance_weight, importance_cover
    FROM ML.FEATURE_IMPORTANCE(MODEL `{PROJECT_ID}.{DATASET_ID}.model_boosted_v1`)
    ORDER BY importance_gain DESC
    LIMIT 15
    """

    rows = run_sql(sql, "Fetching feature importance")
    features = []
    log.info("Top 15 features by importance_gain:")
    for i, row in enumerate(rows, 1):
        log.info(
            "  %2d. %-30s gain=%.4f  weight=%.4f",
            i, row["feature"], row["importance_gain"], row["importance_weight"],
        )
        features.append({
            "rank": i,
            "feature": row["feature"],
            "importance_gain": round(row["importance_gain"], 6),
            "importance_weight": round(row["importance_weight"], 6),
        })
    return features


def run_batch_predictions() -> None:
    """Generate batch predictions on the eval set and save to BigQuery."""
    sql = f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.predictions_eval` AS
    SELECT
      predicted_label,
      predicted_label_probs[OFFSET(0)].prob AS prob_no_attrition,
      predicted_label_probs[OFFSET(1)].prob AS prob_attrition,
      label AS actual_label
    FROM ML.PREDICT(
      MODEL `{PROJECT_ID}.{DATASET_ID}.model_boosted_v1`,
      (SELECT * EXCEPT(data_split) FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean`
       WHERE data_split = 'EVAL'),
      STRUCT(0.45 AS threshold)
    )
    """
    run_sql(sql, "Running batch predictions on eval set")
    log.info("Batch predictions saved to %s.%s.predictions_eval", PROJECT_ID, DATASET_ID)


def main() -> None:
    log.info("=" * 55)
    log.info("Phase 3: Model Training")
    log.info("=" * 55)
    
    # Train both models
    
    train_model("logistic")
    train_model("boosted") 

    # Evaluate
    metrics = evaluate_all_models()

    # Feature importance
    features = get_feature_importance()

    # Batch predictions
    run_batch_predictions()

    # Save results JSON (for CI artefact and notebook consumption)
    results = {"metrics": metrics, "feature_importance": features}
    out_path = RESULTS_DIR / "eval_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Results saved to %s", out_path)

    # Select best model
    best = max(metrics, key=lambda m: metrics[m]["roc_auc"])
    log.info("")
    log.info("🏆 Best model: %s (AUC=%.4f)", best, metrics[best]["roc_auc"])
    log.info("✅ Training complete. Next: python pipeline/deploy_vertex.py")


if __name__ == "__main__":
    main()
