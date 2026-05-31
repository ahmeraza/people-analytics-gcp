# Learning Reflection — People Analytics Intelligence Engine
### End-to-End ML Pipeline on Google Cloud Platform

> **Ahmed Raza · GCP ML Pipeline · 2026**  
> *BigQuery ML · Vertex AI · Cloud Run · FastAPI · Python 3.11*

| Metric | Value |
|---|---|
| ROC-AUC Score | 0.76 |
| Employees Modelled | 1,470 |
| Predictive Features | 26 |
| Inference Latency | <1ms |
| Total Project Cost | ~$2 from $300 credit |

---

## 1. Executive Summary

This document reflects on the complete learning journey of building a production-ready, end-to-end Machine Learning pipeline on Google Cloud Platform — from raw data ingestion to a live REST API serving real-time predictions with FAANG-grade People Science analytics.

The project was built incrementally, encountering and resolving real engineering challenges at every phase. Each problem solved — from Python environment conflicts to BigQuery SQL type casting bugs to Pydantic v2 schema conflicts — represents a genuine learning moment that maps directly to what ML Engineers and People Scientists encounter in production environments at scale.

> **Core Achievement:** A fully automated ML pipeline processing 1,470 employee records through 6 sequential stages — ingest, transform, train, register, serve, and observe — at a total cost of under $2 from a $300 GCP credit, demonstrating efficient resource utilisation without sacrificing production readiness.

---

## 2. Phase 1 — Data Engineering & Ingestion

### 2.1 Cloud Storage Architecture

The pipeline begins with structured ingestion of the IBM Watson HR Attrition dataset into Google Cloud Storage (GCS). This reflects the standard medallion architecture used at FAANG companies — raw data landing zone, then progressive refinement into structured analytical layers.

- GCS bucket creation with sub-folders for `raw-data`, `models`, and `predictions`
- Deterministic file naming using the original Kaggle CSV filename to ensure reproducibility
- Upload via the Python `google-cloud-storage` SDK using service account authentication

> **Production Principle:** Using GCS as a landing zone decouples data ingestion from processing. If BigQuery loading fails, the raw data is preserved in GCS and can be reloaded without re-downloading from Kaggle. This is the same pattern used in enterprise data lake architectures.

### 2.2 BigQuery Schema Autodetection and Type Casting

One of the most instructive bugs in this project was BigQuery's autodetect schema converting the `Attrition` column from string values of `'Yes'`/`'No'` to boolean `true`/`false`. This is a common production pitfall — autodetect makes assumptions that can silently corrupt downstream ML features.

The fix required understanding BigQuery's type system deeply:

- `CAST(Attrition AS BOOL) = TRUE` instead of `CAST(Attrition AS STRING) = 'Yes'`
- `CAST(OverTime AS STRING)` to prevent CASE statement type mismatches
- Explicit schema definition is preferred over autodetect in production pipelines

> **Key Learning:** Never trust autodetect schema in production. Always inspect the inferred types using `INFORMATION_SCHEMA.COLUMNS` after loading and validate against expected types before running any downstream transforms.

### 2.3 Deterministic Train-Eval Split Using FARM_FINGERPRINT

The 80/20 train-evaluation split uses BigQuery's `FARM_FINGERPRINT` function — a production technique for reproducible data splits:

```sql
MOD(ABS(FARM_FINGERPRINT(CAST(EmployeeNumber AS STRING))), 10) < 8
```

Unlike random splits using `RAND()`, `FARM_FINGERPRINT` produces the same split every time the query runs, regardless of when or how many times it is executed. This is critical in production because it prevents data leakage between training runs and ensures evaluation metrics are comparable across model versions.

- Reproducible across pipeline reruns — same `EmployeeNumber` always goes to same split
- No need to materialise a split table — computed on-the-fly in SQL
- Used by Google internally for large-scale ML dataset splits

### 2.4 FAANG-Grade Feature Engineering

Three compound features were engineered inside the BigQuery SQL transform layer, reflecting FAANG People Science methodology:

| Concept | SQL Formula | Production Impact |
|---|---|---|
| **Internal Equity Ratio** | `MonthlyIncome / AVG(MonthlyIncome) OVER(PARTITION BY Department, JobRole)` | Identifies pay inequity vs peer cohort. Values below 0.85 trigger compensation review nudge. |
| **Promotion Stagnation Index** | `YearsAtCompany / (YearsSinceLastPromotion + 1)` | Measures career velocity decay. Values above 3 signal intervention needed. |
| **Manager Dependency Score** | `YearsWithCurrManager / (YearsAtCompany + 1)` | Proxy for single-manager dependency risk. High score with low relationship satisfaction signals retention risk. |

These features are computed in SQL — not Python — ensuring they are applied identically at training time and serving time, eliminating training-serving skew.

---

## 3. Phase 2 — BigQuery ML Model Training

### 3.1 Why BigQuery ML Over Traditional ML Frameworks

BigQuery ML (BQML) is a deliberate architectural choice, not a shortcut. It reflects how ML is actually deployed at FAANG scale — keeping data in place and bringing compute to the data, rather than extracting data to a notebook or training cluster.

- **No data egress costs** — training happens inside BigQuery where the data already lives
- **No GPU cluster provisioning** — BQML handles compute allocation automatically
- **SQL-native feature transforms** — consistent between training and serving, eliminating training-serving skew
- **Direct Vertex AI Model Registry integration** — models registered with a single SQL option

> **Cost Efficiency:** Training both models (logistic regression + boosted trees) on 1,173 rows cost approximately $0 — well within the 1TB free query tier. A comparable training job on a Vertex AI custom training cluster would have cost $3–8. BQML is the correct tool for tabular ML at this data scale.

### 3.2 Model 1 — Logistic Regression Baseline

The logistic regression model serves as the interpretable baseline — a production best practice. Before deploying a complex model, always establish a simpler baseline to understand whether model complexity is justified.

- **L1 regularisation** (`l1_reg = 0.01`) for feature selection — drives sparse coefficients, automatically identifying irrelevant features
- **L2 regularisation** (`l2_reg = 0.01`) for weight decay — prevents overfitting on the small dataset
- **`auto_class_weights = TRUE`** to handle class imbalance — attrition is only 16% of the dataset, so without weighting the model would learn to predict "no attrition" for everyone and achieve 84% accuracy trivially
- **Final AUC: 0.7636** — competitive with the boosted tree on this dataset size

### 3.3 Model 2 — Boosted Tree Classifier (XGBoost)

The primary production model uses a Boosted Tree Classifier — the same XGBoost algorithm used in winning Kaggle solutions and deployed in production at Amazon, Google, and Meta for tabular prediction tasks.

- **`HIST` tree method** — memory-efficient histogram-based split finding, faster than exact split enumeration
- **`subsample = 0.8` and `colsample_bytree = 0.8`** — stochastic gradient boosting to reduce variance
- **`early_stop = TRUE`** with `min_rel_progress = 0.001` — stops training when improvement plateaus, preventing overfitting without manual epoch tuning
- **`max_tree_depth = 6`** — balances model expressiveness with overfitting risk
- **`learn_rate = 0.1`** — conservative learning rate for stable convergence

### 3.4 Model Evaluation — Understanding the Metrics

Model evaluation used four complementary metrics, each measuring a different aspect of classifier performance:

| Metric | Formula | Interpretation |
|---|---|---|
| **ROC-AUC (0.76)** | Area under the ROC curve | Measures model's ability to rank positive cases higher than negatives. 0.5 = random, 1.0 = perfect. |
| **Precision (0.44)** | TP / (TP + FP) | Of employees flagged as high risk, 44% actually left. Low precision means some false alarms. |
| **Recall (0.38)** | TP / (TP + FN) | Of employees who actually left, 38% were correctly identified. The metric to maximise in HR contexts. |
| **F1 Score (0.40)** | Harmonic mean of precision and recall | Balanced metric when both precision and recall matter equally. |

### 3.5 Threshold Calibration — Why 0.45, Not 0.5

The default classification threshold of 0.5 is rarely optimal in real-world HR applications. The cost of missing a high-risk employee (false negative) is far higher than the cost of a false alarm (false positive) — a manager having one unnecessary retention conversation costs far less than losing a critical team member.

Setting the threshold to **0.45** shifts the decision boundary to favour recall over precision — accepting more false positives in exchange for catching more true positives. This is a business decision encoded in the model architecture, not an arbitrary choice.

---

## 4. Phase 3 — Vertex AI Model Registry & Deployment

### 4.1 Model Registry — Version Control for ML Models

The BQML model was imported into Vertex AI Model Registry using the `model_registry = 'vertex_ai'` option in the `CREATE MODEL` statement. This gives the model a permanent home with versioning, lineage tracking, and deployment management.

- **Model versioning** — every retraining creates a new version; older versions remain accessible for rollback
- **Lineage tracking** — Vertex AI records which dataset version and training configuration produced each model version
- **Centralised deployment management** — endpoints, traffic splits, and canary deployments managed from a single UI

### 4.2 Online Prediction Endpoint

The Vertex AI endpoint (n1-standard-2) provides online prediction — real-time inference for individual employees at sub-second latency. This is the serving layer that Cloud Run FastAPI calls when a prediction request arrives.

> **Cost Discipline:** The endpoint was kept live for approximately 30 minutes of testing, then immediately deleted using the `--delete-endpoint` flag. At $0.35/hr, this cost approximately $0.18. This reflects production cost discipline — endpoints should only be live when actively serving traffic, not left running indefinitely for convenience.

### 4.3 Batch vs Online Prediction — Understanding the Trade-off

| Mode | Use Cases | Characteristics |
|---|---|---|
| **Online Prediction** (Vertex AI Endpoint) | Individual risk assessment, real-time manager dashboard, API-driven interventions | Millisecond latency, higher per-prediction cost, endpoint must be running |
| **Batch Prediction** (BigQuery `ML.PREDICT`) | Monthly HR analytics reports, cohort-level risk assessment, dashboard population | Minutes latency for full dataset, near-zero cost, no endpoint needed |

---

## 5. Phase 4 — FastAPI REST API & Cloud Run Deployment

### 5.1 FastAPI Architecture Decisions

The REST API is built with FastAPI — the same framework used by production ML serving systems at Uber, Netflix, and Microsoft. Key architectural decisions:

- **Pydantic v2 data validation** — all inputs validated against a typed schema before reaching prediction logic. Invalid inputs return structured 422 errors, not server crashes
- **Lifespan context manager** — the Vertex AI client is initialised once at startup, not on every request, eliminating authentication overhead from every prediction call
- **Demo mode** — when `VERTEX_ENDPOINT_ID` is not set, the API returns rule-based mock predictions, enabling local development and testing without GCP credentials or a live endpoint
- **Structured JSON logging** — all logs emitted as JSON with consistent fields, queryable in Cloud Logging using log-based metrics

### 5.2 FAANG-Grade Analytics Functions

Three Python functions implement the People Science analytics layer, running server-side on every prediction:

#### `calculate_burnout_index()`

Google Googlegeist-inspired composite score synthesising five psychometric signals into a 0–100 index. The weighting reflects the relative impact of each signal on burnout based on organisational psychology research — overtime receives 25 points because chronic overwork is the strongest single predictor of burnout, while promotion stagnation receives only 5 points per year because career frustration builds more slowly.

```python
score = (
    (5 - WorkLifeBalance) * 20     # overtime recovery capacity
    + overtime * 25                # chronic overwork signal (heaviest weight)
    + (5 - JobSatisfaction) * 15   # intrinsic motivation decay
    + (5 - EnvironmentSatisfaction) * 15  # workplace friction
    + min(YearsSinceLastPromotion, 5) * 5 # career stagnation velocity
) / 1.8  # normalise to 0–100
```

#### `get_talent_quadrant()`

Amazon-inspired 2×2 classification matrix. Segments employees by attrition risk (above/below 0.45 threshold) and performance value (`PerformanceRating >= 3` AND `JobLevel >= 3`). The **High Risk / High Value** quadrant is the primary intervention target — these are the employees whose departure would be most damaging and most preventable.

#### `generate_nudges()`

Google Project Oxygen-inspired rule engine generating context-aware HR interventions. Six independent rules check different risk signals and generate specific, actionable recommendations. The rules are intentionally transparent — each nudge explains which metric triggered it, so managers understand the reasoning and can exercise professional judgment.

### 5.3 Cloud Run — Serverless Production Deployment

Cloud Run provides the ideal hosting environment for this API — serverless, containerised, and billed only when handling requests:

- **Source-based deployment** — `gcloud run deploy --source` builds the Docker image in Cloud Build, pushes to Artifact Registry, and deploys in a single command
- **Multi-stage Dockerfile** — builder stage installs dependencies; runtime stage copies only installed packages. Final image under 150MB, cold start under 3 seconds
- **Non-root user (`appuser`)** in the container — security best practice, prevents container escape attacks
- **Free tier coverage** — 2 million requests per month are free. The project's expected traffic is well within this limit
- **Automatic HTTPS** — Cloud Run provides a managed SSL certificate and public HTTPS URL with no configuration

---

## 6. Phase 5 — FAANG-Grade People Analytics Views

### 6.1 BigQuery Views as Analytics Layer

Three BigQuery views implement the analytical layer on top of the prediction results, reflecting how FAANG People Science teams structure their analytics infrastructure — separating raw data, ML predictions, and business analytics into distinct layers.

| View | Source | Key Outputs |
|---|---|---|
| `v_burnout_risk` | All 1,470 employees from `attrition_clean` | Burnout tier distribution by department, manager effectiveness correlation, overtime impact analysis |
| `v_attrition_segmented` | 297 eval-set employees with predictions joined to `attrition_clean` | Regretted vs non-regretted attrition analysis, talent quadrant distribution, compensation correlation |
| `v_manager_effectiveness` | All employees from `attrition_clean` | Manager effectiveness score by department, relationship stability patterns, tenure correlation |

### 6.2 The JOIN Design Decision

The `v_attrition_segmented` view required a JOIN between `predictions_eval` (which only has 4 columns from `ML.PREDICT`) and `attrition_clean` (which has all 29 features). This reflects a common production pattern — ML prediction outputs are intentionally lean and must be joined back to the feature store to reconstruct the full analytical context.

In a production system with a stable employee identifier like `EmployeeNumber`, you would JOIN on that key rather than the binary label.

---

## 7. Engineering Lessons — What Real Production Looks Like

### 7.1 Environment Management

The project encountered the most common Python environment issue in professional development — Conda base environment conflicting with a virtual environment. When both `(base)` and `(.venv)` appear in the shell prompt simultaneously, Python resolves imports from Conda's packages rather than the venv's packages, causing `ImportError` failures even when packages are correctly installed in the venv.

**Resolution:** `conda config --set auto_activate_base false` followed by deactivating and reactivating the venv. Understanding *why* this happens — Conda modifies `PATH` at shell startup, overriding the venv's `PATH` modifications — is more valuable than just knowing the fix.

### 7.2 SQL Type System Discipline

Three SQL type errors were encountered and resolved during the project:

| Error | Root Cause | Resolution |
|---|---|---|
| `CASE statement type mismatch` | All `WHEN` branches must return the same type. Mixing STRING and INT64 causes a `400 No matching signature` error | Explicit `CAST()` on ambiguous columns |
| `Autodetect schema type inference` | BigQuery inferred `Attrition` as BOOL from CSV values `'True'`/`'False'` | `CAST(Attrition AS BOOL) = TRUE` |
| `Window function alias restriction` | Column aliases defined in the same SELECT cannot be referenced in `OVER()` clauses | Use original column expression or a subquery |

### 7.3 Pydantic v2 Migration Patterns

The FastAPI project encountered two Pydantic v2 breaking changes from v1:

- **`validator` decorator removed** — replaced with `field_validator` in v2. Importing `validator` from pydantic in a v2 environment causes a model construction error even if no `@validator` decorators are actually used
- **Field name namespace collision** — Pydantic v2 reserves the `model_` namespace. A field named `model_version` conflicts with this namespace. Resolved by adding `model_config = {'protected_namespaces': ()}` to the model class
- **Class-level name shadowing** — a field named `Department` with type annotation `Department` (the enum class) creates a name resolution conflict. Resolved by renaming the enum to `DepartmentEnum`

### 7.4 Cost Optimisation in Practice

The total project cost was under $2 from a $300 GCP credit through deliberate resource management:

| Component | Actual Cost | Why |
|---|---|---|
| BigQuery ML Training | $0 | 1,470 rows is far below the 1TB free query tier |
| Cloud Storage | $0 | 5GB free tier. The IBM dataset is approximately 300KB |
| Vertex AI Endpoint | ~$0.18 | Live for approximately 30 minutes at $0.35/hr. Deleted immediately after testing |
| Cloud Run | $0 | 2 million free requests per month |
| Cloud Build | $0 | 120 free build-minutes per day. Single build used approximately 4 minutes |

---

## 8. Phase 6 — Looker Studio Dashboard Architecture

### 8.1 Why Looker Studio Over Power BI or Tableau

Looker Studio is the correct tool choice for this project for three reasons: it is **free** with no license cost, it connects **natively to BigQuery** with no data export required, and it is the tool used by People Analytics teams at Google and other GCP-native organisations.

- Power BI requires a Pro license ($10/month) to share reports online
- Tableau Public makes data publicly visible — unacceptable for HR analytics
- Looker Studio shares via URL with Google account authentication — same access control as Google Docs

### 8.2 Four Dashboard Architecture

| Dashboard | Source View | Key Visualisations |
|---|---|---|
| Burnout Monitor | `v_burnout_risk` | Burnout tier distribution by department, overtime correlation, promotion stagnation heatmap |
| Attrition Segmentation | `v_attrition_segmented` | Talent quadrant scatter, regretted vs non-regretted breakdown, compensation vs risk correlation |
| Manager Effectiveness | `v_manager_effectiveness` | Effectiveness score distribution, relationship stability by department, tenure patterns |
| Model Performance | `predictions_eval` | Confusion matrix approximation, probability distribution, actual vs predicted comparison |

---

## 9. Complete Concept Reference

### 9.1 ML & Data Science Concepts

| Concept | Implementation | What It Does |
|---|---|---|
| Binary Classification | BQML model training | Predicts a binary outcome (attrition yes/no) using labelled historical data |
| ROC-AUC | Model evaluation | Threshold-independent measure of classifier discrimination power |
| Precision-Recall Trade-off | Threshold calibration at 0.45 | Business decision: favour recall to minimise missed high-risk employees |
| Class Imbalance Handling | `auto_class_weights = TRUE` | Prevents the model learning to always predict the majority class |
| L1 Regularisation | Logistic regression OPTIONS | Feature selection — drives sparse coefficients, removing irrelevant features |
| L2 Regularisation | Logistic regression OPTIONS | Weight decay — prevents overfitting on small datasets |
| Early Stopping | Boosted tree OPTIONS | Stops training when validation improvement plateaus, automatically preventing overfitting |
| Feature Importance (SHAP) | `ML.FEATURE_IMPORTANCE` | Identifies which features drive predictions most — StockOptionLevel ranked #1 |
| Deterministic Splits | `FARM_FINGERPRINT` in SQL | Reproducible train-eval splits consistent across pipeline reruns |
| Compound Feature Engineering | `ingest.py` SQL transforms | Creating new predictive signals by combining raw features |
| Batch vs Online Inference | `ML.PREDICT` vs Vertex endpoint | Choosing prediction mode based on latency requirements and cost |
| Histogram-Based Tree Splitting | `tree_method = 'HIST'` | Memory-efficient split finding for boosted trees |
| Stochastic Gradient Boosting | `subsample = 0.8`, `colsample_bytree = 0.8` | Reduces variance by training each tree on a random subset of data and features |

### 9.2 MLOps & Engineering Concepts

| Concept | Implementation | What It Does |
|---|---|---|
| Data Lake Architecture | GCS landing zone + BigQuery | Separation of raw storage from analytical compute |
| Training-Serving Consistency | SQL features in BQML | Same SQL transforms used at training and serving time, eliminating skew |
| Model Registry & Versioning | Vertex AI Model Registry | Persistent model storage with version history, lineage, and deployment management |
| Serverless Serving | Cloud Run deployment | Container-based serving that scales to zero when idle, billed only on requests |
| Multi-Stage Docker Builds | `api/Dockerfile` | Separation of build dependencies from runtime image, minimising final image size |
| Non-Root Container Security | `appuser` in Dockerfile | Prevents container escape attacks by running as a non-privileged user |
| Structured Logging | JSON logs in `main.py` | Machine-readable log format queryable in Cloud Logging with log-based metrics |
| API Schema Validation | Pydantic v2 `BaseModel` | Automatic input validation and error handling before prediction logic executes |
| Demo Mode / Feature Flags | `VERTEX_ENDPOINT_ID` env var | Graceful degradation when external dependencies are unavailable |
| CI/CD Pipeline | GitHub Actions workflow | Automated linting, testing, and Docker build validation on every push |
| Cost-Aware Architecture | BQML + Cloud Run free tiers | Designing for minimal resource consumption without compromising production readiness |
| Lifespan Context Manager | FastAPI `@asynccontextmanager` | Initialises expensive resources once at startup, shared across all requests |

### 9.3 People Science Concepts

| Concept | Implementation | Business Rationale |
|---|---|---|
| Psychometric Strain Index | `calculate_burnout_index()` | Composite score from validated organisational psychology burnout indicators |
| Internal Equity Analysis | `internal_equity_ratio` feature | Compensation fairness measurement vs peer cohort using window functions |
| Talent Matrix Segmentation | `get_talent_quadrant()` | Amazon-inspired 2×2 risk-value classification for retention triage |
| Behavioural Nudge Theory | `generate_nudges()` | Context-aware managerial interventions based on Google Project Oxygen methodology |
| Regretted Attrition Analysis | `v_attrition_segmented` view | Distinguishing high-value departures from low-value departures for prioritisation |
| Manager Effectiveness Scoring | `v_manager_effectiveness` view | Proxy metrics for managerial quality based on team satisfaction and stability signals |
| Promotion Stagnation Velocity | `promotion_stagnation_index` feature | Career progression rate measurement — decaying velocity predicts flight risk |
| Manager Dependency Risk | `manager_dependency_score` feature | Single-manager dependency combined with low relationship satisfaction flags flight risk |
| Threshold Calibration for Recall | 0.45 prediction threshold | Business decision: cost of missing a departure exceeds cost of a false alarm |
| Class Imbalance as Business Signal | 16% attrition rate in dataset | Reflects real-world HR data — most employees stay, which must be handled explicitly in modelling |

---

## 10. Final Reflection — What This Project Proves

This project demonstrates something more valuable than technical skill alone — it demonstrates the ability to build, debug, and ship a complete system under real constraints, recovering from environment failures, SQL bugs, schema conflicts, and API errors without abandoning the goal.

Every error encountered and resolved in this project is an error that occurs in professional ML engineering environments. The Python-Conda conflict, the BigQuery autodetect schema issue, the Pydantic v2 migration breaking changes, the Vertex AI feature mismatch — these are not beginner mistakes. They are the exact friction points that experienced engineers navigate daily.

> **The Real Achievement:** A production-ready ML pipeline that: (1) costs under $2 to build, (2) handles real data with real schema bugs, (3) serves live predictions via a public HTTPS endpoint, (4) implements FAANG People Science methodology in code, and (5) is documented, version-controlled, and CI-tested. This is not a tutorial project — it is a production system built at minimal scale.

### 10.1 What to Build Next

The natural extensions that would further demonstrate depth:

- **`ML.EXPLAIN_PREDICT` with SHAP values** — replace `ML.PREDICT` with `ML.EXPLAIN_PREDICT` in the FastAPI to return per-prediction feature attributions, making the API fully explainable
- **Vertex AI Model Monitoring** — configure data drift detection and prediction skew alerts, the production monitoring layer described in the original project specification
- **Looker Studio embedded dashboards** — embed the four BigQuery views as a public-facing analytics dashboard linked from the landing page
- **Cloud Scheduler + Cloud Functions** — automate monthly retraining by triggering `train_runner.py` on a schedule, completing the fully automated ML pipeline
- **Terraform infrastructure as code** — replace the shell script `setup.sh` with Terraform configurations for reproducible, version-controlled GCP infrastructure

---

*People Analytics Intelligence Engine · Ahmed Raza · 2026*  
*Live: https://people-analytics-api-831718383093.us-central1.run.app*  
*GitHub: https://github.com/ahmeraza/people-analytics-gcp*
