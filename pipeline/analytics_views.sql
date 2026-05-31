-- FAANG-inspired People Analytics Views

CREATE OR REPLACE VIEW `{PROJECT_ID}.{DATASET_ID}.v_burnout_risk` AS
SELECT
  JobLevel,
  job_role_enc,
  dept_enc,
  Age,
  YearsAtCompany,
  WorkLifeBalance,
  overtime,
  JobSatisfaction,
  EnvironmentSatisfaction,
  YearsSinceLastPromotion,
  RelationshipSatisfaction,
  internal_equity_ratio,
  promotion_stagnation_index,
  manager_dependency_score,
  ROUND(
    (
      (5 - WorkLifeBalance) * 20 +
      overtime * 25 +
      (5 - JobSatisfaction) * 15 +
      (5 - EnvironmentSatisfaction) * 15 +
      LEAST(YearsSinceLastPromotion, 5) * 5
    ) / 1.8,
  1) AS burnout_risk_index,
  CASE
    WHEN ((5 - WorkLifeBalance) * 20 + overtime * 25 + (5 - JobSatisfaction) * 15 + (5 - EnvironmentSatisfaction) * 15 + LEAST(YearsSinceLastPromotion, 5) * 5) / 1.8 >= 70 THEN 'Critical'
    WHEN ((5 - WorkLifeBalance) * 20 + overtime * 25 + (5 - JobSatisfaction) * 15 + (5 - EnvironmentSatisfaction) * 15 + LEAST(YearsSinceLastPromotion, 5) * 5) / 1.8 >= 50 THEN 'High'
    WHEN ((5 - WorkLifeBalance) * 20 + overtime * 25 + (5 - JobSatisfaction) * 15 + (5 - EnvironmentSatisfaction) * 15 + LEAST(YearsSinceLastPromotion, 5) * 5) / 1.8 >= 30 THEN 'Medium'
    ELSE 'Low'
  END AS burnout_tier,
  label AS actual_attrition
FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean`;

CREATE OR REPLACE VIEW `{PROJECT_ID}.{DATASET_ID}.v_attrition_segmented` AS
SELECT
  p.predicted_label,
  p.prob_attrition,
  p.actual_label,
  c.JobLevel,
  c.PerformanceRating,
  c.TotalWorkingYears,
  c.MonthlyIncome,
  c.YearsAtCompany,
  c.dept_enc,
  c.job_role_enc,
  CASE
    WHEN p.predicted_label = 1 AND c.PerformanceRating >= 3 AND c.JobLevel >= 3 THEN 'Regretted'
    WHEN p.predicted_label = 1 AND (c.PerformanceRating < 3 OR c.JobLevel < 3) THEN 'Non-Regretted'
    ELSE 'Retained'
  END AS attrition_type,
  CASE
    WHEN p.prob_attrition >= 0.6 AND c.PerformanceRating >= 3 THEN 'High Risk / High Value'
    WHEN p.prob_attrition >= 0.6 AND c.PerformanceRating < 3  THEN 'High Risk / Low Value'
    WHEN p.prob_attrition < 0.6  AND c.PerformanceRating >= 3 THEN 'Low Risk / High Value'
    ELSE 'Low Risk / Low Value'
  END AS talent_quadrant
FROM `{PROJECT_ID}.{DATASET_ID}.predictions_eval` p
JOIN `{PROJECT_ID}.{DATASET_ID}.attrition_clean` c
  ON p.actual_label = c.label
WHERE c.data_split = 'EVAL';

CREATE OR REPLACE VIEW `{PROJECT_ID}.{DATASET_ID}.v_manager_effectiveness` AS
SELECT
  YearsWithCurrManager,
  JobSatisfaction,
  RelationshipSatisfaction,
  WorkLifeBalance,
  JobInvolvement,
  dept_enc,
  JobLevel,
  ROUND(
    (
      RelationshipSatisfaction * 20 +
      JobSatisfaction * 20 +
      JobInvolvement * 20 +
      WorkLifeBalance * 15 +
      LEAST(YearsWithCurrManager, 5) * 5
    ) / 1.6,
  1) AS manager_effectiveness_score,
  CASE
    WHEN YearsWithCurrManager >= 4 AND RelationshipSatisfaction >= 3 THEN 'Stable'
    WHEN YearsWithCurrManager < 2 AND RelationshipSatisfaction <= 2  THEN 'At Risk'
    ELSE 'Developing'
  END AS manager_relationship_status,
  label AS actual_attrition
FROM `{PROJECT_ID}.{DATASET_ID}.attrition_clean`;
