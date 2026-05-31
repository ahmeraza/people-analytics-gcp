"""
pipeline/analytics_runner.py
Creates FAANG-inspired people analytics views in BigQuery.
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
DATASET_ID = os.environ["BQ_DATASET"]


def strip_comments(sql_text):
    return re.sub(r'--[^\n]*', '', sql_text)


def main():
    client = bigquery.Client(project=PROJECT_ID)
    sql = Path("pipeline/analytics_views.sql").read_text()
    sql = sql.replace("{PROJECT_ID}", PROJECT_ID)
    sql = sql.replace("{DATASET_ID}", DATASET_ID)
    sql = strip_comments(sql)

    statements = [
        s.strip() for s in sql.split(";")
        if s.strip()
    ]

    for i, stmt in enumerate(statements):
        print(f"Running view {i+1}/{len(statements)}...")
        client.query(stmt).result()
        print(f"✅ Done")

    print("\n✅ All analytics views created. Check BigQuery for:")
    print("   - v_burnout_risk")
    print("   - v_attrition_segmented")
    print("   - v_manager_effectiveness")


if __name__ == "__main__":
    main()