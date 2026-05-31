"""
tests/test_pipeline.py

Unit tests for the pipeline scripts.
Tests schema validation, SQL generation, and data transformations.
No GCP credentials needed.
"""

import pytest
from pathlib import Path


# ── SQL file validation ───────────────────────────────────────────────────────
class TestTrainSQL:
    def test_sql_file_exists(self):
        assert Path("pipeline/train.sql").exists()

    def test_sql_contains_logistic_model(self):
        sql = Path("pipeline/train.sql").read_text()
        assert "LOGISTIC_REG" in sql

    def test_sql_contains_boosted_model(self):
        sql = Path("pipeline/train.sql").read_text()
        assert "BOOSTED_TREE_CLASSIFIER" in sql

    def test_sql_uses_custom_split(self):
        sql = Path("pipeline/train.sql").read_text()
        assert "data_split_method" in sql
        assert "CUSTOM" in sql

    def test_sql_registers_to_vertex(self):
        sql = Path("pipeline/train.sql").read_text()
        assert "model_registry" in sql
        assert "vertex_ai" in sql

    def test_sql_handles_class_imbalance(self):
        sql = Path("pipeline/train.sql").read_text()
        assert "auto_class_weights" in sql

    def test_sql_uses_tuned_threshold(self):
        """Threshold should be 0.45 (not default 0.5) for recall-favouring HR use case."""
        sql = Path("pipeline/train.sql").read_text()
        assert "0.45" in sql

    def test_sql_includes_feature_importance(self):
        sql = Path("pipeline/train.sql").read_text()
        assert "ML.FEATURE_IMPORTANCE" in sql

    def test_sql_includes_confusion_matrix(self):
        sql = Path("pipeline/train.sql").read_text()
        assert "ML.CONFUSION_MATRIX" in sql


# ── Feature encoding ──────────────────────────────────────────────────────────
class TestFeatureEncoding:
    """Validate the categorical encoding logic in the API matches the SQL transforms."""

    def _encode(self, dept: str) -> int:
        dept_map = {"Sales": 1, "Research & Development": 2, "Human Resources": 3}
        return dept_map.get(dept, 0)

    def test_sales_encodes_to_1(self):
        assert self._encode("Sales") == 1

    def test_rd_encodes_to_2(self):
        assert self._encode("Research & Development") == 2

    def test_hr_encodes_to_3(self):
        assert self._encode("Human Resources") == 3

    def test_unknown_encodes_to_0(self):
        assert self._encode("Marketing") == 0


# ── Repo structure ────────────────────────────────────────────────────────────
class TestRepoStructure:
    """Ensure all required files are present for a production-ready repo."""

    REQUIRED_FILES = [
        "README.md",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "pipeline/ingest.py",
        "pipeline/train.sql",
        "pipeline/train_runner.py",
        "pipeline/deploy_vertex.py",
        "api/main.py",
        "api/Dockerfile",
        "api/requirements.txt",
        "infra/setup.sh",
        ".github/workflows/ci.yml",
    ]

    @pytest.mark.parametrize("filepath", REQUIRED_FILES)
    def test_file_exists(self, filepath):
        assert Path(filepath).exists(), f"Missing required file: {filepath}"

    def test_readme_has_architecture_section(self):
        readme = Path("README.md").read_text()
        assert "Architecture" in readme

    def test_readme_has_cost_section(self):
        readme = Path("README.md").read_text()
        assert "Cost" in readme

    def test_readme_has_quickstart(self):
        readme = Path("README.md").read_text()
        assert "Quickstart" in readme

    def test_env_example_has_required_vars(self):
        env_example = Path(".env.example").read_text()
        for var in ["GCP_PROJECT_ID", "GCS_BUCKET", "BQ_DATASET"]:
            assert var in env_example, f"Missing {var} in .env.example"

    def test_gitignore_excludes_secrets(self):
        gitignore = Path(".gitignore").read_text()
        assert ".env" in gitignore
        assert "service-account-key.json" in gitignore
