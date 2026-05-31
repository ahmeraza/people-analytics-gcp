-- pipeline/train.sql
-- BigQuery ML model training statements
-- Run via: python pipeline/train_runner.py  (which substitutes variables)
-- or execute directly in the BigQuery console after substituting {PROJECT_ID} and {DATASET_ID}

-- ── Model 1: Logistic Regression (baseline) ───────────────────────────────

CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.model_logistic_v1`
OPTIONS (
  model_type              = 'LOGISTIC_REG',
  input_label_cols        = ['label'],
  data_split_method       = 'NO_SPLIT',
  auto_class_weights      = TRUE,
  max_iterations          = 50,
  l1_reg                  = 0.01,
  l2_reg                  = 0.01,
  model_registry          = 'vertex_ai',
  vertex_ai_model_id      = 'attrition-logistic',
  enable_global_explain   = TRUE
)
AS
SELECT
  * EXCEPT(data_split)
FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean`
WHERE data_split = 'TRAIN';


-- ── Model 2: Boosted Trees (primary model) ────────────────────────────────

CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.model_boosted_v1`
OPTIONS (
  model_type              = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols        = ['label'],
  data_split_method       = 'NO_SPLIT',
  auto_class_weights      = TRUE,
  num_parallel_tree       = 1,
  max_iterations          = 100,
  tree_method             = 'HIST',
  min_tree_child_weight   = 5,
  max_tree_depth          = 6,
  subsample               = 0.8,
  colsample_bytree        = 0.8,
  learn_rate              = 0.1,
  early_stop              = TRUE,
  min_rel_progress        = 0.001,
  model_registry          = 'vertex_ai',
  vertex_ai_model_id      = 'attrition-boosted',
  enable_global_explain   = TRUE
)
AS
SELECT
  * EXCEPT(data_split)
FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean`
WHERE data_split = 'TRAIN';


-- ── Evaluation queries (run after training) ───────────────────────────────

-- Logistic regression metrics
SELECT
  'Logistic Regression' AS model,
  roc_auc,
  precision,
  recall,
  f1_score,
  log_loss,
  accuracy
FROM ML.EVALUATE(
  MODEL `{PROJECT_ID}.{DATASET_ID}.model_logistic_v1`,
  (SELECT * EXCEPT(data_split) FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean`
   WHERE data_split = 'EVAL')
);

-- Boosted trees metrics
SELECT
  'Boosted Trees' AS model,
  roc_auc,
  precision,
  recall,
  f1_score,
  log_loss,
  accuracy
FROM ML.EVALUATE(
  MODEL `{PROJECT_ID}.{DATASET_ID}.model_boosted_v1`,
  (SELECT * EXCEPT(data_split) FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean`
   WHERE data_split = 'EVAL')
);

-- Feature importance (boosted model)
SELECT
  feature,
  importance_gain,
  importance_weight,
  importance_cover
FROM ML.FEATURE_IMPORTANCE(MODEL `{PROJECT_ID}.{DATASET_ID}.model_boosted_v1`)
ORDER BY importance_gain DESC;

-- Confusion matrix at default threshold (0.5)
SELECT *
FROM ML.CONFUSION_MATRIX(
  MODEL `{PROJECT_ID}.{DATASET_ID}.model_boosted_v1`,
  (SELECT * EXCEPT(data_split) FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean`
   WHERE data_split = 'EVAL'),
  STRUCT(0.45 AS threshold)   -- tuned threshold (favour recall for HR use case)
);

-- Batch prediction on eval set (used for offline reporting)
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.predictions_eval` AS
SELECT
  predicted_label,
  predicted_label_probs,
  label AS actual_label
FROM ML.PREDICT(
  MODEL `{PROJECT_ID}.{DATASET_ID}.model_boosted_v1`,
  (SELECT * EXCEPT(data_split, label) FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean`
   WHERE data_split = 'EVAL'),
  STRUCT(0.45 AS threshold)
);
