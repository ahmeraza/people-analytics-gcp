#!/usr/bin/env bash
# infra/setup.sh — One-command GCP project bootstrap
# Usage: ./infra/setup.sh
# Requires: gcloud CLI authenticated, PROJECT_ID set in .env or exported

set -euo pipefail

# ── Load environment ────────────────────────────────────────────────────────
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

: "${GCP_PROJECT_ID:?GCP_PROJECT_ID must be set in .env}"
: "${GCS_BUCKET:?GCS_BUCKET must be set in .env}"
: "${BQ_DATASET:?BQ_DATASET must be set in .env}"
: "${GCP_REGION:=us-central1}"

SA_NAME="attrition-pipeline-sa"
SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

echo "🔧 Bootstrapping project: ${GCP_PROJECT_ID}"

# ── Set active project ───────────────────────────────────────────────────────
gcloud config set project "${GCP_PROJECT_ID}"

# ── Enable required APIs ─────────────────────────────────────────────────────
echo "📡 Enabling GCP APIs..."
gcloud services enable \
  bigquery.googleapis.com \
  bigquerymigration.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  cloudbuild.googleapis.com \
  --quiet

echo "✅ APIs enabled."

# ── Create GCS bucket ────────────────────────────────────────────────────────
echo "🪣 Creating Cloud Storage bucket: gs://${GCS_BUCKET}"
if ! gsutil ls "gs://${GCS_BUCKET}" &>/dev/null; then
  gsutil mb -p "${GCP_PROJECT_ID}" -l "${GCP_REGION}" "gs://${GCS_BUCKET}"
  echo "✅ Bucket created."
else
  echo "ℹ️  Bucket already exists, skipping."
fi

# Sub-folders (GCS doesn't have real folders, but placeholder objects help organise)
gsutil cp /dev/null "gs://${GCS_BUCKET}/raw-data/.keep"
gsutil cp /dev/null "gs://${GCS_BUCKET}/models/.keep"
gsutil cp /dev/null "gs://${GCS_BUCKET}/batch-predictions/.keep"

# ── Create BigQuery dataset ──────────────────────────────────────────────────
echo "📊 Creating BigQuery dataset: ${BQ_DATASET}"
if ! bq ls --project_id="${GCP_PROJECT_ID}" "${BQ_DATASET}" &>/dev/null; then
  bq mk \
    --project_id="${GCP_PROJECT_ID}" \
    --dataset \
    --location="${GCP_REGION}" \
    --description="Employee attrition ML pipeline" \
    "${BQ_DATASET}"
  echo "✅ Dataset created."
else
  echo "ℹ️  Dataset already exists, skipping."
fi

# ── Create service account ───────────────────────────────────────────────────
echo "🔑 Creating service account: ${SA_NAME}"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="Attrition Pipeline SA" \
    --description="Least-privilege SA for employee attrition ML pipeline"
  echo "✅ Service account created."
else
  echo "ℹ️  Service account already exists, skipping."
fi

# ── Grant IAM roles (least privilege) ───────────────────────────────────────
echo "🔐 Granting IAM roles..."
ROLES=(
  "roles/bigquery.dataEditor"
  "roles/bigquery.jobUser"
  "roles/storage.objectAdmin"
  "roles/aiplatform.user"
  "roles/logging.logWriter"
  "roles/monitoring.metricWriter"
  "roles/run.invoker"
)

for ROLE in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet
  echo "  ✓ ${ROLE}"
done

# ── Download service account key (for local dev only) ───────────────────────
KEY_PATH="infra/service-account-key.json"
if [ ! -f "${KEY_PATH}" ]; then
  echo "🗝  Downloading SA key to ${KEY_PATH} (local dev only — never commit this)"
  gcloud iam service-accounts keys create "${KEY_PATH}" \
    --iam-account="${SA_EMAIL}"
  echo "  ⚠️  Add ${KEY_PATH} to .gitignore if not already present."
fi

# ── Create Artifact Registry for Docker images ───────────────────────────────
REGISTRY_NAME="attrition-repo"
echo "🐳 Creating Artifact Registry: ${REGISTRY_NAME}"
if ! gcloud artifacts repositories describe "${REGISTRY_NAME}" \
    --location="${GCP_REGION}" &>/dev/null; then
  gcloud artifacts repositories create "${REGISTRY_NAME}" \
    --repository-format=docker \
    --location="${GCP_REGION}" \
    --description="Docker images for attrition pipeline"
  echo "✅ Registry created."
else
  echo "ℹ️  Registry already exists, skipping."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅  Bootstrap complete!"
echo ""
echo "Next steps:"
echo "  1. Download dataset: kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset -p data/ --unzip"
echo "  2. Run pipeline:     python pipeline/ingest.py"
echo "  3. Train model:      python pipeline/train_runner.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
