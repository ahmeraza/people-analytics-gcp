"""
api/main.py

Phase 4 — Cloud Run REST API
FastAPI application that wraps the Vertex AI endpoint for online predictions.
Deployed to Cloud Run (serverless, scales to zero).

Environment variables required:
    GCP_PROJECT_ID  — GCP project ID
    GCP_REGION      — GCP region (default: us-central1)
    VERTEX_ENDPOINT_ID — Vertex AI endpoint resource ID (numeric)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional

import google.auth
import google.auth.transport.requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import aiplatform
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, HTMLResponse


# ── FAANG-grade analytics functions ──────────────────────────────────────────

def calculate_burnout_index(features: dict):
    """Calculate burnout risk index (Google Googlegeist-inspired)."""
    score = (
        (5 - features.get("WorkLifeBalance", 3)) * 20 +
        features.get("overtime", 0) * 25 +
        (5 - features.get("JobSatisfaction", 3)) * 15 +
        (5 - features.get("EnvironmentSatisfaction", 3)) * 15 +
        min(features.get("YearsSinceLastPromotion", 0), 5) * 5
    ) / 1.8

    score = round(min(max(score, 0), 100), 1)

    if score >= 70:
        tier = "Critical"
    elif score >= 50:
        tier = "High"
    elif score >= 30:
        tier = "Medium"
    else:
        tier = "Low"

    return score, tier


def get_talent_quadrant(attrition_prob: float, performance_rating: int, job_level: int) -> str:
    """Classify employee into talent quadrant (Amazon-inspired)."""
    high_value = performance_rating >= 3 and job_level >= 3
    high_risk = attrition_prob >= 0.45

    if high_risk and high_value:
        return "High Risk / High Value"
    elif high_risk and not high_value:
        return "High Risk / Low Value"
    elif not high_risk and high_value:
        return "Low Risk / High Value"
    else:
        return "Low Risk / Low Value"


def generate_nudges(features: dict, attrition_prob: float) -> List[str]:
    """Generate actionable HR interventions (Google Project Oxygen-inspired)."""
    nudges = []

    if features.get("overtime", 0) == 1 and features.get("WorkLifeBalance", 4) <= 2 and attrition_prob > 0.70:
        nudges.append("Critical: Review overtime allocation immediately — burnout risk is high.")

    if features.get("internal_equity_ratio", 1.0) < 0.85:
        nudges.append("Perform out-of-cycle compensation equity review vs peer cohort.")

    if features.get("promotion_stagnation_index", 0) > 3:
        nudges.append("Initiate career development conversation — promotion stagnation detected.")

    if features.get("JobSatisfaction", 4) <= 2 and features.get("JobInvolvement", 4) <= 2:
        nudges.append("Schedule skip-level check-in — disengagement pattern detected.")

    if features.get("manager_dependency_score", 0) > 0.7 and features.get("RelationshipSatisfaction", 4) <= 2:
        nudges.append("Review manager relationship — high dependency with low satisfaction.")

    if features.get("EnvironmentSatisfaction", 4) <= 2:
        nudges.append("Assess workplace environment — satisfaction below threshold.")

    if not nudges:
        nudges.append("No immediate intervention required — monitor quarterly.")

    return nudges


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("GCP_REGION", "us-central1")
ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID")
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.45"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

_endpoint: Optional[aiplatform.Endpoint] = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _endpoint

    if PROJECT_ID and ENDPOINT_ID:
        log.info("Initializing Vertex AI client...")
        aiplatform.init(project=PROJECT_ID, location=REGION)
        _endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
        )
        log.info("Vertex AI endpoint ready: %s", ENDPOINT_ID)
    else:
        log.warning(
            "GCP_PROJECT_ID or VERTEX_ENDPOINT_ID not set — "
            "running in demo mode (predictions will be mocked)"
        )

    yield
    log.info("Shutting down API...")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="People Analytics Attrition Prediction API",
    description=(
        "Predicts employee attrition probability using a BigQuery ML "
        "boosted tree model. Returns burnout risk, talent quadrant, "
        "and FAANG-grade HR intervention nudges."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Schema ────────────────────────────────────────────────────────────────────
class DepartmentEnum(str, Enum):
    sales = "Sales"
    rd = "Research & Development"
    hr = "Human Resources"


class OverTimeEnum(str, Enum):
    yes = "Yes"
    no = "No"


class EmployeeFeatures(BaseModel):
    """Input features for attrition prediction."""

    Age: int = Field(..., ge=18, le=70, description="Employee age in years")
    MonthlyIncome: float = Field(..., ge=1000, description="Monthly income in USD")
    YearsAtCompany: int = Field(..., ge=0, le=40)
    YearsSinceLastPromotion: int = Field(default=1, ge=0, le=15)
    YearsWithCurrManager: int = Field(default=2, ge=0, le=17)
    NumCompaniesWorked: int = Field(default=2, ge=0, le=9)
    DistanceFromHome: int = Field(default=10, ge=1, le=29)
    PercentSalaryHike: int = Field(default=14, ge=11, le=25)
    TrainingTimesLastYear: int = Field(default=3, ge=0, le=6)
    TotalWorkingYears: int = Field(default=8, ge=0, le=40)
    JobSatisfaction: int = Field(default=3, ge=1, le=4, description="1=Low 4=Very High")
    WorkLifeBalance: int = Field(default=3, ge=1, le=4, description="1=Bad 4=Best")
    EnvironmentSatisfaction: int = Field(default=3, ge=1, le=4)
    RelationshipSatisfaction: int = Field(default=3, ge=1, le=4)
    JobInvolvement: int = Field(default=3, ge=1, le=4)
    PerformanceRating: int = Field(default=3, ge=1, le=4)
    Education: int = Field(default=3, ge=1, le=5)
    JobLevel: int = Field(default=2, ge=1, le=5)
    StockOptionLevel: int = Field(default=1, ge=0, le=3)
    OverTime: OverTimeEnum = Field(default=OverTimeEnum.no)
    dept: DepartmentEnum = Field(default=DepartmentEnum.rd, description="Department")
    internal_equity_ratio: float = Field(default=1.0, ge=0.0, le=3.0)
    promotion_stagnation_index: float = Field(default=2.0, ge=0.0, le=10.0)
    manager_dependency_score: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_bqml_instance(self) -> dict:
        """Convert to the feature vector expected by the BQML model."""
        dept_map = {"Sales": 1, "Research & Development": 2, "Human Resources": 3}
        return {
            "Age": self.Age,
            "MonthlyIncome": self.MonthlyIncome,
            "YearsAtCompany": self.YearsAtCompany,
            "YearsSinceLastPromotion": self.YearsSinceLastPromotion,
            "YearsWithCurrManager": self.YearsWithCurrManager,
            "NumCompaniesWorked": self.NumCompaniesWorked,
            "DistanceFromHome": self.DistanceFromHome,
            "PercentSalaryHike": self.PercentSalaryHike,
            "TrainingTimesLastYear": self.TrainingTimesLastYear,
            "TotalWorkingYears": self.TotalWorkingYears,
            "overtime": 1 if self.OverTime == OverTimeEnum.yes else 0,
            "dept_enc": dept_map.get(self.dept.value, 0),
            "job_role_enc": 1,
            "marital_enc": 1,
            "edu_field_enc": 1,
            "JobSatisfaction": self.JobSatisfaction,
            "WorkLifeBalance": self.WorkLifeBalance,
            "EnvironmentSatisfaction": self.EnvironmentSatisfaction,
            "RelationshipSatisfaction": self.RelationshipSatisfaction,
            "JobInvolvement": self.JobInvolvement,
            "PerformanceRating": self.PerformanceRating,
            "Education": self.Education,
            "JobLevel": self.JobLevel,
            "StockOptionLevel": self.StockOptionLevel,
            "internal_equity_ratio": self.internal_equity_ratio,
            "promotion_stagnation_index": self.promotion_stagnation_index,
            "manager_dependency_score": self.manager_dependency_score,
        }
    
class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    attrition_probability: float = Field(..., description="Probability of attrition (0-1)")
    prediction: str = Field(..., description="High Risk or Low Risk")
    burnout_risk_index: float = Field(..., description="Burnout risk score 0-100")
    burnout_tier: str = Field(..., description="Critical / High / Medium / Low")
    talent_quadrant: str = Field(..., description="Amazon-style talent quadrant")
    nudges: List[str] = Field(..., description="Actionable HR interventions")
    threshold_used: float
    model_version: str
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>People Analytics API — Workforce Intelligence Engine</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #070a0f;
    --surface: #0d1117;
    --surface2: #161b22;
    --border: rgba(255,255,255,0.08);
    --accent: #00d4aa;
    --accent2: #4f9eff;
    --accent3: #ff6b6b;
    --accent4: #ffd166;
    --text: #e6edf3;
    --muted: #7d8590;
    --font-display: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    font-weight: 300;
    overflow-x: hidden;
    min-height: 100vh;
  }

  /* ── Grid background ── */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,212,170,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,170,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
  }

  /* ── Gradient orbs ── */
  .orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(120px);
    pointer-events: none;
    z-index: 0;
  }
  .orb-1 {
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0,212,170,0.12), transparent 70%);
    top: -100px; left: -100px;
    animation: drift 20s ease-in-out infinite alternate;
  }
  .orb-2 {
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(79,158,255,0.1), transparent 70%);
    bottom: 100px; right: -100px;
    animation: drift 25s ease-in-out infinite alternate-reverse;
  }
  @keyframes drift {
    from { transform: translate(0, 0); }
    to { transform: translate(40px, 40px); }
  }

  /* ── Layout ── */
  .container {
    position: relative;
    z-index: 1;
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 32px;
  }

  /* ── Nav ── */
  nav {
    position: relative;
    z-index: 10;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 24px 40px;
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(12px);
  }

  .nav-logo {
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
  }

  .nav-links {
    display: flex;
    gap: 32px;
    list-style: none;
  }

  .nav-links a {
    color: var(--muted);
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 400;
    letter-spacing: 0.05em;
    transition: color 0.2s;
  }

  .nav-links a:hover { color: var(--text); }

  /* ── Hero ── */
  .hero {
    padding: 100px 40px 80px;
    text-align: center;
  }

  .hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,212,170,0.08);
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 32px;
    animation: fadeUp 0.6s ease both;
  }

  .hero-eyebrow::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 2s ease infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
  }

  h1 {
    font-family: var(--font-display);
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.03em;
    margin-bottom: 16px;
    animation: fadeUp 0.6s 0.1s ease both;
  }

  h1 .accent { color: var(--accent); }
  h1 .accent2 { color: var(--accent2); }

  /* ── Scrolling ticker ── */
  .ticker-wrap {
    overflow: hidden;
    margin: 24px 0 48px;
    animation: fadeUp 0.6s 0.2s ease both;
    border-top: 1px solid rgba(0,212,170,0.2);
    border-bottom: 1px solid rgba(0,212,170,0.2);
    padding: 12px 0;
    background: rgba(0,212,170,0.03);
  }

  .ticker {
    display: flex;
    white-space: nowrap;
    animation: ticker 30s linear infinite;
  }

  .ticker-item {
    font-size: 1.1rem;
    font-weight: 300;
    color: var(--muted);
    padding-right: 60px;
    flex-shrink: 0;
  }

  .ticker-item span {
    color: var(--accent);
    font-weight: 500;
  }

  @keyframes ticker {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
  }

  /* ── CTA buttons ── */
  .cta-group {
    display: flex;
    gap: 16px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 80px;
    animation: fadeUp 0.6s 0.3s ease both;
  }

  .btn-primary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--accent);
    color: #070a0f;
    padding: 14px 28px;
    border-radius: 8px;
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    text-decoration: none;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 0 30px rgba(0,212,170,0.25);
  }

  .btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 50px rgba(0,212,170,0.4);
  }

  .btn-secondary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: transparent;
    color: var(--text);
    padding: 14px 28px;
    border-radius: 8px;
    border: 1px solid var(--border);
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    text-decoration: none;
    transition: border-color 0.2s, background 0.2s;
  }

  .btn-secondary:hover {
    border-color: rgba(255,255,255,0.2);
    background: rgba(255,255,255,0.04);
  }

  /* ── Interactive Metrics 3D Flip Grid ── */
  .metrics-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 80px;
    animation: fadeUp 0.6s 0.4s ease both;
    overflow: visible;
  }

  .metric-cell {
    background: transparent;
    perspective: 1000px;
    height: 140px;
  }

  .metric-card-inner {
    position: relative;
    width: 100%;
    height: 100%;
    text-align: center;
    transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    transform-style: preserve-3d;
    cursor: pointer;
  }

  .metric-cell:hover .metric-card-inner {
    transform: rotateY(180deg);
  }

  .metric-front, .metric-back {
    position: absolute;
    width: 100%;
    height: 100%;
    -webkit-backface-visibility: hidden;
    backface-visibility: hidden;
    border-radius: 16px;
    border: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 24px;
  }

  .metric-front {
    background: var(--surface);
    transition: background 0.2s;
  }

  .metric-cell:hover .metric-front {
    background: var(--surface2);
  }

  .metric-back {
    background: var(--surface2);
    transform: rotateY(180deg);
    border-color: rgba(0, 212, 170, 0.3);
    box-shadow: 0 0 20px rgba(0,212,170,0.05);
  }

  .metric-value {
    font-family: var(--font-display);
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 8px;
  }

  .metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  .metric-back-text {
    font-size: 0.75rem;
    line-height: 1.4;
    color: var(--text);
    font-weight: 400;
  }

  /* ── Section title ── */
  .section-title {
    font-family: var(--font-display);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* ── Feature cards ── */
  .features-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-bottom: 80px;
  }

  .feature-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.3s;
    cursor: default;
  }

  .feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--card-accent, var(--accent));
    opacity: 0;
    transition: opacity 0.3s;
  }

  .feature-card:hover {
    border-color: rgba(255,255,255,0.15);
    transform: translateY(-4px);
  }

  .feature-card:hover::before { opacity: 1; }

  /* Feature Inline Flex Rows */
  .feature-header-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }

  .feature-icon {
    width: 36px; height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    background: var(--card-bg, rgba(0,212,170,0.1));
    flex-shrink: 0;
  }

  .feature-tag {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--card-accent, var(--accent));
    background: var(--card-bg, rgba(0,212,170,0.1));
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 12px;
  }

  .feature-card h3 {
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
  }

  .feature-card p {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--muted);
  }

  /* ── Pipeline section ── */
  .pipeline {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 80px;
  }

  .pipeline-steps {
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
    margin-top: 24px;
  }

  .pipeline-step {
    flex: 1;
    min-width: 120px;
    text-align: center;
    padding: 16px 8px;
    position: relative;
  }

  .pipeline-step:not(:last-child)::after {
    content: '→';
    position: absolute;
    right: -8px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--muted);
    font-size: 1rem;
  }

  .step-num {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 0.8rem;
    color: var(--accent);
    margin: 0 auto 10px;
  }

  .step-name {
    font-family: var(--font-display);
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
  }

  .step-tech {
    font-size: 0.7rem;
    color: var(--muted);
  }

  /* ── Tech badges ── */
  .tech-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 80px;
  }

  .tech-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text);
    letter-spacing: 0.03em;
    transition: border-color 0.2s;
  }

  .tech-badge:hover { border-color: rgba(255,255,255,0.2); }
  .tech-badge .dot { width: 6px; height: 6px; border-radius: 50%; }

  /* ── Footer ── */
  footer {
    border-top: 1px solid var(--border);
    padding: 32px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    z-index: 1;
  }

  .footer-text {
    font-size: 0.8rem;
    color: var(--muted);
  }

  .footer-links {
    display: flex;
    gap: 24px;
  }

  .footer-links a {
    font-size: 0.8rem;
    color: var(--muted);
    text-decoration: none;
    transition: color 0.2s;
  }

  .footer-links a:hover { color: var(--text); }

  /* ── Animations ── */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    nav { padding: 20px 24px; }
    .hero { padding: 60px 24px 40px; }
    .metrics-bar { grid-template-columns: repeat(2, 1fr); gap: 12px; }
    .features-grid { grid-template-columns: 1fr; }
    .pipeline-steps { gap: 8px; }
    .pipeline-step:not(:last-child)::after { display: none; }
    footer { flex-direction: column; gap: 16px; text-align: center; }
  }
</style>
</head>
<body>

<div class="orb orb-1"></div>
<div class="orb orb-2"></div>

<nav>
  <div class="nav-logo">PA · API</div>
  <ul class="nav-links">
    <li><a href="/docs">API Docs</a></li>
    <li><a href="/health">Status</a></li>
    <li><a href="https://github.com/ahmeraza/people-analytics-gcp" target="_blank">GitHub</a></li>
  </ul>
</nav>

<div class="hero">
  <div class="hero-eyebrow">Live · GCP Production</div>

  <h1>
    People <span class="accent">Analytics</span><br>
    <span class="accent2">Intelligence</span> Engine
  </h1>

  <div class="ticker-wrap">
    <div class="ticker">
      <div class="ticker-item">⚡ [SYSTEM STATUS: NOMINAL] &nbsp;·&nbsp; INFRASTRUCTURE: <span>Google Cloud Platform (GCP)</span> &nbsp;·&nbsp; LATENCY: <span>&lt;1ms (ONLINE INFERENCE)</span> &nbsp;·&nbsp;</div>
      <div class="ticker-item">📊 DATA WAREHOUSE: <span>BigQuery ML Pipelines Active</span> &nbsp;·&nbsp; MODEL ENDPOINT ID: <span>VERTEX-AI-RUN-v1</span> &nbsp;·&nbsp;</div>
      <div class="ticker-item">🛡️ SERVICE LAYER: <span>FastAPI Serverless Deployment</span> &nbsp;·&nbsp; CONTAINER RUNTIME: <span>Docker + Cloud Run</span> &nbsp;·&nbsp;</div>
      <div class="ticker-item">⚡ [SYSTEM STATUS: NOMINAL] &nbsp;·&nbsp; INFRASTRUCTURE: <span>Google Cloud Platform (GCP)</span> &nbsp;·&nbsp; LATENCY: <span>&lt;1ms (ONLINE INFERENCE)</span> &nbsp;·&nbsp;</div>
      <div class="ticker-item">📊 DATA WAREHOUSE: <span>BigQuery ML Pipelines Active</span> &nbsp;·&nbsp; MODEL ENDPOINT ID: <span>VERTEX-AI-RUN-v1</span> &nbsp;·&nbsp;</div>
      <div class="ticker-item">🛡️ SERVICE LAYER: <span>FastAPI Serverless Deployment</span> &nbsp;·&nbsp; CONTAINER RUNTIME: <span>Docker + Cloud Run</span> &nbsp;·&nbsp;</div>
    </div>
  </div>

  <div class="cta-group">
    <a href="/docs" class="btn-primary">↗ Explore API Docs</a>
    <a href="/health" class="btn-secondary">◉ System Status</a>
  </div>
</div>

<div class="container">

  <div class="metrics-bar">
    
    <div class="metric-cell">
      <div class="metric-card-inner">
        <div class="metric-front">
          <div class="metric-value">0.76</div>
          <div class="metric-label">ROC-AUC Score</div>
        </div>
        <div class="metric-back">
          <p class="metric-back-text"><strong>Model Discriminative Power:</strong> Threshold calibrated to 0.45 to prioritize flight recall over precision—ensuring critical burnout departures are captured early.</p>
        </div>
      </div>
    </div>

    <div class="metric-cell">
      <div class="metric-card-inner">
        <div class="metric-front">
          <div class="metric-value">1,470</div>
          <div class="metric-label">Employees Modelled</div>
        </div>
        <div class="metric-back">
          <p class="metric-back-text"><strong>Cohort Sample Size:</strong> Modeled utilizing baseline enterprise validation architectures to ensure statistically sound, bias-mitigated trend extrapolation.</p>
        </div>
      </div>
    </div>

    <div class="metric-cell">
      <div class="metric-card-inner">
        <div class="metric-front">
          <div class="metric-value">26</div>
          <div class="metric-label">Predictive Features</div>
        </div>
        <div class="metric-back">
          <p class="metric-back-text"><strong>Psychometric Vector Depth:</strong> Integrates compound features like Promotion Stagnation and Manager Dependency scores alongside traditional operational data.</p>
        </div>
      </div>
    </div>

    <div class="metric-cell">
      <div class="metric-card-inner">
        <div class="metric-front">
          <div class="metric-value">&lt;1ms</div>
          <div class="metric-label">Inference Latency</div>
        </div>
        <div class="metric-back">
          <p class="metric-back-text"><strong>Real-time Decision Support:</strong> Serverless architecture via Cloud Run + FastAPI allows micro-latency online predictions at point of intervention.</p>
        </div>
      </div>
    </div>

  </div>

  <div class="section-title">Core Capabilities</div>

  <div class="features-grid">

    <div class="feature-card" style="--card-accent: #00d4aa; --card-bg: rgba(0,212,170,0.08); padding-bottom: 16px;">
      <div class="feature-tag">Predictive Modeling</div>
      <div class="feature-header-row">
        <div class="feature-icon">⚡</div>
        <h3>Flight Risk & Retention Intelligence</h3>
      </div>
      <p style="margin-bottom: 20px;">Boosted Tree Classifier trained on 26 psychometric and organizational variables. Threshold calibrated to 0.45 favouring recall — minimising the cost of undetected high-risk talent departure.</p>
      
      <div class="ticker-wrap" style="margin: 0 -32px -16px; border-left: none; border-right: none; background: rgba(0,212,170,0.02); padding: 8px 0;">
        <div class="ticker" style="animation-duration: 20s;">
          <div class="ticker-item" style="font-size: 0.75rem; color: #00d4aa;">⚠️ ALERT: High flight risk detected in Engineering Cohort (Prob: 0.84) &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: var(--muted);">SIGNAL: Tenure transition inflection point identified at Month 18 &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: #00d4aa;">⚠️ ALERT: High flight risk detected in Engineering Cohort (Prob: 0.84) &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: var(--muted);">SIGNAL: Tenure transition inflection point identified at Month 18 &nbsp;·&nbsp;</div>
        </div>
      </div>
    </div>

    <div class="feature-card" style="--card-accent: #ff6b6b; --card-bg: rgba(255,107,107,0.08); padding-bottom: 16px;">
      <div class="feature-tag">Psychometric Analysis</div>
      <div class="feature-header-row">
        <div class="feature-icon">🔥</div>
        <h3>Macro-Burnout & Psychometric Strain Index</h3>
      </div>
      <p style="margin-bottom: 20px;">Composite score (0–100) synthesising WorkLifeBalance, OverTime exposure, JobSatisfaction, EnvironmentSatisfaction, and promotion stagnation velocity. Tiered into Critical / High / Medium / Low intervention bands.</p>
      
      <div class="ticker-wrap" style="margin: 0 -32px -16px; border-left: none; border-right: none; border-top-color: rgba(255,107,107,0.2); border-bottom-color: rgba(255,107,107,0.2); background: rgba(255,107,107,0.02); padding: 8px 0;">
        <div class="ticker" style="animation-duration: 22s;">
          <div class="ticker-item" style="font-size: 0.75rem; color: #ff6b6b;">💥 CRITICAL: Product Team Strain Index at 82% (Chronic Overtime Pattern) &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: var(--muted);">VARIANCE: Environment satisfaction trailing baseline by 14% &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: #ff6b6b;">💥 CRITICAL: Product Team Strain Index at 82% (Chronic Overtime Pattern) &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: var(--muted);">VARIANCE: Environment satisfaction trailing baseline by 14% &nbsp;·&nbsp;</div>
        </div>
      </div>
    </div>

    <div class="feature-card" style="--card-accent: #4f9eff; --card-bg: rgba(79,158,255,0.08); padding-bottom: 16px;">
      <div class="feature-tag">Performance Psychology</div>
      <div class="feature-header-row">
        <div class="feature-icon">◈</div>
        <h3>Strategic Talent Matrix Optimization</h3>
      </div>
      <p style="margin-bottom: 20px;">Amazon-inspired 2×2 human capital classification framework. Segments workforce by attrition probability and performance value into quadrants — enabling People teams to triage retention investment with precision.</p>
      
      <div class="ticker-wrap" style="margin: 0 -32px -16px; border-left: none; border-right: none; border-top-color: rgba(79,158,255,0.2); border-bottom-color: rgba(79,158,255,0.2); background: rgba(79,158,255,0.02); padding: 8px 0;">
        <div class="ticker" style="animation-duration: 25s;">
          <div class="ticker-item" style="font-size: 0.75rem; color: #4f9eff;">📊 MATRIX: 12 High-Value / High-Risk assets mapped to Segment Alpha &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: var(--muted);">ACTION: Allocating equity refreshing structures to retention lockbox &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: #4f9eff;">📊 MATRIX: 12 High-Value / High-Risk assets mapped to Segment Alpha &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: var(--muted);">ACTION: Allocating equity refreshing structures to retention lockbox &nbsp;·&nbsp;</div>
        </div>
      </div>
    </div>

    <div class="feature-card" style="--card-accent: #ffd166; --card-bg: rgba(255,209,102,0.08); padding-bottom: 16px;">
      <div class="feature-tag">Behavioral Economics</div>
      <div class="feature-header-row">
        <div class="feature-icon">🎯</div>
        <h3>Prescriptive Behavioral Interventions</h3>
      </div>
      <p style="margin-bottom: 20px;">Google Project Oxygen-inspired nudge engine. Generates systemic, context-aware managerial interventions per prediction — surfacing compensation inequity, promotion stagnation, manager dependency, and disengagement signals automatically.</p>
      
      <div class="ticker-wrap" style="margin: 0 -32px -16px; border-left: none; border-right: none; border-top-color: rgba(255,209,102,0.2); border-bottom-color: rgba(255,209,102,0.2); background: rgba(255,209,102,0.02); padding: 8px 0;">
        <div class="ticker" style="animation-duration: 18s;">
          <div class="ticker-item" style="font-size: 0.75rem; color: #ffd166;">🎯 NUDGE: Triggering out-of-cycle comp equity review (Peer Ratio &lt; 0.85) &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: var(--muted);">NUDGE: Initiating skip-level stay conversations due to stagnation score &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: #ffd166;">🎯 NUDGE: Triggering out-of-cycle comp equity review (Peer Ratio &lt; 0.85) &nbsp;·&nbsp;</div>
          <div class="ticker-item" style="font-size: 0.75rem; color: var(--muted);">NUDGE: Initiating skip-level stay conversations due to stagnation score &nbsp;·&nbsp;</div>
        </div>
      </div>
    </div>

  </div>

  <div class="section-title">MLOps Architecture</div>

  <div class="pipeline">
    <div class="pipeline-steps">
      <div class="pipeline-step">
        <div class="step-num">01</div>
        <div class="step-name">Ingest</div>
        <div class="step-tech">Cloud Storage + BigQuery</div>
      </div>
      <div class="pipeline-step">
        <div class="step-num">02</div>
        <div class="step-name">Transform</div>
        <div class="step-tech">BigQuery SQL + Feature Eng.</div>
      </div>
      <div class="pipeline-step">
        <div class="step-num">03</div>
        <div class="step-name">Train</div>
        <div class="step-tech">BigQuery ML XGBoost</div>
      </div>
      <div class="pipeline-step">
        <div class="step-num">04</div>
        <div class="step-name">Register</div>
        <div class="step-tech">Vertex AI Model Registry</div>
      </div>
      <div class="pipeline-step">
        <div class="step-num">05</div>
        <div class="step-name">Serve</div>
        <div class="step-tech">Cloud Run + FastAPI</div>
      </div>
      <div class="pipeline-step">
        <div class="step-num">06</div>
        <div class="step-name">Observe</div>
        <div class="step-tech">Cloud Logging + Monitoring</div>
      </div>
    </div>
  </div>

  <div class="section-title">Technology Stack</div>

  <div class="tech-row">
    <div class="tech-badge"><div class="dot" style="background:#4285f4"></div>BigQuery ML</div>
    <div class="tech-badge"><div class="dot" style="background:#00d4aa"></div>Vertex AI</div>
    <div class="tech-badge"><div class="dot" style="background:#4f9eff"></div>Cloud Run</div>
    <div class="tech-badge"><div class="dot" style="background:#ff6b6b"></div>Cloud Storage</div>
    <div class="tech-badge"><div class="dot" style="background:#ffd166"></div>Cloud Logging</div>
    <div class="tech-badge"><div class="dot" style="background:#00d4aa"></div>FastAPI</div>
    <div class="tech-badge"><div class="dot" style="background:#4f9eff"></div>Python 3.11</div>
    <div class="tech-badge"><div class="dot" style="background:#ff6b6b"></div>Docker</div>
    <div class="tech-badge"><div class="dot" style="background:#ffd166"></div>Pydantic v2</div>
    <div class="tech-badge"><div class="dot" style="background:#4285f4"></div>GitHub Actions CI</div>
  </div>

</div>

<footer>
  <div class="footer-text">People Analytics API · Built on Google Cloud Platform</div>
  <div class="footer-links">
    <a href="/docs">API Reference</a>
    <a href="https://github.com/ahmeraza/people-analytics-gcp" target="_blank">Source Code</a>
    <a href="/health">Status</a>
  </div>
</footer>

</body>
</html>
"""

@app.get("/health", tags=["system"])
async def health_check():
    """Liveness probe for Cloud Run."""
    return {
        "status": "healthy",
        "endpoint_configured": _endpoint is not None,
        "model_version": MODEL_VERSION,
    }

@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(employee: EmployeeFeatures, request: Request):
    """
    Predict attrition probability for a single employee.
    Returns probability, burnout index, talent quadrant, and HR nudges.
    """
    start = time.monotonic()
    instance = employee.to_bqml_instance()

    log.info(
        '{"event":"predict_request","overtime":%s,"job_satisfaction":%s}',
        instance["overtime"], instance["JobSatisfaction"],
    )

    if _endpoint is not None:
        try:
            response = _endpoint.predict(instances=[instance])
            pred = response.predictions[0]

            if isinstance(pred, dict) and "label_probs" in pred:
                label_values = pred.get("label_values", ["1", "0"])
                label_probs = pred.get("label_probs", [0.5, 0.5])
                try:
                    idx = label_values.index("1")
                    attrition_prob = float(label_probs[idx])
                except ValueError:
                    attrition_prob = float(label_probs[0])
            elif isinstance(pred, dict) and "scores" in pred:
                attrition_prob = float(pred["scores"][1])
            elif isinstance(pred, dict) and "predicted_label_probs" in pred:
                probs = pred["predicted_label_probs"]
                attrition_prob = next(
                    (p["prob"] for p in probs if p["label"] == 1), 0.5
                )
            else:
                attrition_prob = float(pred) if isinstance(pred, (int, float)) else 0.5

        except Exception as exc:
            log.error("Vertex AI prediction failed: %s", str(exc))
            raise HTTPException(status_code=502, detail=f"Prediction service error: {exc}")
    else:
        log.warning("Demo mode: returning mock prediction")
        high_risk_signals = (
            instance["overtime"] == 1
            or instance["JobSatisfaction"] <= 2
            or instance["WorkLifeBalance"] <= 2
            or instance["YearsAtCompany"] <= 2
        )
        attrition_prob = 0.78 if high_risk_signals else 0.21

    latency_ms = (time.monotonic() - start) * 1000
    burnout_index, burnout_tier = calculate_burnout_index(instance)

    result = PredictionResponse(
        attrition_probability=round(attrition_prob, 4),
        prediction="High Risk" if attrition_prob >= PREDICTION_THRESHOLD else "Low Risk",
        burnout_risk_index=burnout_index,
        burnout_tier=burnout_tier,
        talent_quadrant=get_talent_quadrant(
            attrition_prob,
            instance.get("PerformanceRating", 3),
            instance.get("JobLevel", 2),
        ),
        nudges=generate_nudges(instance, attrition_prob),
        threshold_used=PREDICTION_THRESHOLD,
        model_version=MODEL_VERSION,
        latency_ms=round(latency_ms, 1),
    )

    log.info(
        '{"event":"predict_response","prob":%.4f,"risk":"%s","latency_ms":%.1f}',
        result.attrition_probability, result.prediction, result.latency_ms,
    )

    return result


@app.post("/predict/batch", tags=["prediction"])
async def predict_batch(employees: List[EmployeeFeatures]):
    """Predict attrition for up to 50 employees in a single request."""
    if len(employees) > 50:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeds limit of 50. Use BigQuery ML for larger datasets.",
        )

    if _endpoint is None:
        raise HTTPException(status_code=503, detail="Vertex AI endpoint not configured")

    instances = [e.to_bqml_instance() for e in employees]
    start = time.monotonic()

    try:
        response = _endpoint.predict(instances=instances)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    results = []
    for pred in response.predictions:
        if isinstance(pred, dict) and "scores" in pred:
            prob = float(pred["scores"][1])
        else:
            prob = 0.5
        results.append({
            "attrition_probability": round(prob, 4),
            "prediction": "High Risk" if prob >= PREDICTION_THRESHOLD else "Low Risk",
        })

    return {
        "count": len(results),
        "predictions": results,
        "latency_ms": round((time.monotonic() - start) * 1000, 1),
    }


# ── Error handlers ────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
