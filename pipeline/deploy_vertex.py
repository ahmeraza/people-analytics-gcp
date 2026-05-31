"""
pipeline/deploy_vertex.py

Phase 4 — Vertex AI Deployment
Imports the BQML-trained model into Vertex AI Model Registry and deploys
it to an online prediction endpoint.

Usage:
    python pipeline/deploy_vertex.py [--delete-endpoint]

IMPORTANT: Delete the endpoint after testing to avoid ongoing charges (~$0.35/hr).
Run: python pipeline/deploy_vertex.py --delete-endpoint
"""

import argparse
import logging
import os
import time

from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
REGION = os.environ.get("GCP_REGION", "us-central1")
DATASET_ID = os.environ["BQ_DATASET"]

MODEL_DISPLAY_NAME = "attrition-boosted-bqml"
ENDPOINT_DISPLAY_NAME = "attrition-endpoint"

# The BQML model in Vertex AI Registry (set by model_registry='vertex_ai' in SQL)
VERTEX_MODEL_ID = os.environ.get("VERTEX_MODEL_ID", "attrition-boosted")


def init_vertex() -> None:
    aiplatform.init(project=PROJECT_ID, location=REGION)
    log.info("Vertex AI initialised: project=%s  region=%s", PROJECT_ID, REGION)


def get_or_create_endpoint() -> aiplatform.Endpoint:
    """Return existing endpoint or create a new one."""
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"',
        order_by="create_time desc",
    )

    if endpoints:
        log.info("Found existing endpoint: %s", endpoints[0].resource_name)
        return endpoints[0]

    log.info("Creating new endpoint: %s", ENDPOINT_DISPLAY_NAME)
    endpoint = aiplatform.Endpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        labels={"project": "attrition-pipeline", "env": "dev"},
    )
    log.info("Endpoint created: %s", endpoint.resource_name)
    return endpoint


def get_latest_model() -> aiplatform.Model:
    """Get the latest version of the attrition model from Vertex AI Registry."""
    models = aiplatform.Model.list(
        filter=f'display_name="{VERTEX_MODEL_ID}"',
        order_by="create_time desc",
    )

    if not models:
        raise RuntimeError(
            f"Model '{VERTEX_MODEL_ID}' not found in Vertex AI Registry. "
            "Ensure BQML training completed with model_registry='vertex_ai'."
        )

    model = models[0]
    log.info(
        "Using model: %s (version=%s, created=%s)",
        model.display_name, model.version_id, model.create_time,
    )
    return model


def deploy_model(model: aiplatform.Model, endpoint: aiplatform.Endpoint) -> None:
    """Deploy model to endpoint with a minimal, cost-effective configuration."""
    log.info("Deploying model to endpoint — this takes ~5-10 minutes...")

    deployed = model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",   # Minimal VM — sufficient for testing
        min_replica_count=1,
        max_replica_count=1,            # No auto-scaling for dev (saves cost)
        traffic_percentage=100,
        deployed_model_display_name=f"{VERTEX_MODEL_ID}-v1",
        sync=True,                      # Block until deployment completes
    )

    log.info("✅ Model deployed successfully.")
    log.info("Endpoint resource name: %s", endpoint.resource_name)
    log.info("")
    log.info("⚠️  REMINDER: Delete this endpoint after testing to stop billing.")
    log.info("    Run: python pipeline/deploy_vertex.py --delete-endpoint")


def test_endpoint(endpoint: aiplatform.Endpoint) -> None:
    """Send a test prediction to verify the endpoint is working."""
    log.info("Sending test prediction...")

    # Sample employee with high attrition risk factors
    test_instance = {
        "Age": 28,
        "MonthlyIncome": 2500,
        "YearsAtCompany": 1,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 0,
        "NumCompaniesWorked": 5,
        "DistanceFromHome": 25,
        "PercentSalaryHike": 11,
        "TrainingTimesLastYear": 1,
        "TotalWorkingYears": 4,
        "overtime": 1,
        "dept_enc": 1,       # Sales
        "job_role_enc": 7,   # Sales Representative
        "marital_enc": 1,    # Single
        "edu_field_enc": 3,  # Marketing
        "JobSatisfaction": 1,
        "WorkLifeBalance": 1,
        "EnvironmentSatisfaction": 2,
        "RelationshipSatisfaction": 2,
        "JobInvolvement": 2,
        "PerformanceRating": 3,
        "Education": 2,
        "JobLevel": 1,
        "StockOptionLevel": 0,
        "internal_equity_ratio": 1.0,
        "promotion_stagnation_index": 2.0,
        "manager_dependency_score": 0.5,
    }

    prediction = endpoint.predict(instances=[test_instance])

    log.info("Test prediction result:")
    log.info("  Raw response: %s", prediction.predictions)

    for pred in prediction.predictions:
        if isinstance(pred, dict):
            prob = pred.get("scores", [0, 0])[1]  # prob of class 1 (attrition)
            log.info("  Attrition probability: %.4f", prob)
            log.info("  Risk level: %s", "HIGH" if prob >= 0.45 else "LOW")


def delete_endpoint(endpoint: aiplatform.Endpoint) -> None:
    """Undeploy all models and delete the endpoint to stop billing."""
    log.info("Undeploying all models from endpoint...")
    endpoint.undeploy_all()

    log.info("Deleting endpoint: %s", endpoint.resource_name)
    endpoint.delete()
    log.info("✅ Endpoint deleted. No further charges for this endpoint.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy or delete Vertex AI endpoint")
    parser.add_argument(
        "--delete-endpoint",
        action="store_true",
        help="Delete the endpoint instead of deploying (stops billing)",
    )
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("Phase 4: Vertex AI Deployment")
    log.info("=" * 55)

    init_vertex()
    endpoint = get_or_create_endpoint()

    if args.delete_endpoint:
        delete_endpoint(endpoint)
        return

    model = get_latest_model()
    deploy_model(model, endpoint)

    log.info("Waiting 30s for endpoint to warm up...")
    time.sleep(30)

    test_endpoint(endpoint)

    log.info("")
    log.info("Next: Deploy the Cloud Run API")
    log.info("  cd api && gcloud run deploy attrition-api --source . --region %s", REGION)


if __name__ == "__main__":
    main()
