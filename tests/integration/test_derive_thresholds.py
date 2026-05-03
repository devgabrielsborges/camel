from __future__ import annotations

import json
import tempfile
from pathlib import Path

import duckdb
import numpy as np

from camel.domain.services.threshold_derivation import derive_thresholds
from camel.infrastructure.adapters.duckdb_reference_scores import load_reference_scores
from camel.infrastructure.adapters.threshold_repository import ThresholdProfileRepository


def _create_synthetic_db(db_path: str, n_per_category: int = 50, seed: int = 42) -> None:
    """Create a temporary DuckDB with synthetic scored data for two models."""
    rng = np.random.default_rng(seed)
    conn = duckdb.connect(db_path)

    conn.execute(
        """
        CREATE TABLE fct_inference_results (
            session_id VARCHAR,
            run_id VARCHAR,
            timestamp TIMESTAMP,
            input VARCHAR,
            output VARCHAR,
            model VARCHAR,
            data_category_QA VARCHAR,
            language BIGINT
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE fct_evaluation_scores (
            session_id VARCHAR,
            run_id VARCHAR,
            timestamp TIMESTAMP,
            data_category_QA VARCHAR,
            language BIGINT,
            correctness BOOLEAN,
            guidelines BOOLEAN,
            token_overlap_f1 DOUBLE,
            class_exact_match BOOLEAN,
            refusal_detection BOOLEAN,
            groundedness DOUBLE,
            pass_at_k BOOLEAN,
            pass_at_k_best_score DOUBLE,
            failure_mode VARCHAR
        )
    """
    )

    idx = 0
    for model in ("model-a", "model-b"):
        run_id = f"run_{model}"
        for category in ("positivo", "negativo"):
            for i in range(n_per_category):
                session_id = f"s_{idx}"
                idx += 1

                overlap = float(rng.normal(0.5, 0.1))
                refusal = bool(rng.choice([True, False], p=[0.1, 0.9]))
                groundedness = float(rng.normal(0.7, 0.15))

                conn.execute(
                    "INSERT INTO fct_inference_results VALUES (?, ?, CURRENT_TIMESTAMP, '', '', ?, ?, 0)",
                    [session_id, run_id, model, category],
                )
                conn.execute(
                    """INSERT INTO fct_evaluation_scores VALUES
                    (?, ?, CURRENT_TIMESTAMP, ?, 0, true, true, ?, false, ?, ?, false, 0.0, '')""",
                    [session_id, run_id, category, overlap, refusal, groundedness],
                )

    conn.close()


class TestDeriveThresholdsIntegration:
    def test_round_trip_synthetic_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            profile_path = str(Path(tmpdir) / "profile.json")

            _create_synthetic_db(db_path, n_per_category=30)

            collections = load_reference_scores(db_path, ["model-a", "model-b"])
            assert len(collections) == 2

            profile = derive_thresholds(
                collections,
                bootstrap_b=1000,
                seed=42,
                percentile=5,
                alpha=0.05,
                benchmark="test-integration",
                reference_models=("model-a", "model-b"),
                reference_run_ids=("run_model-a", "run_model-b"),
            )

            repo = ThresholdProfileRepository()
            repo.save(profile, profile_path)

            assert Path(profile_path).exists()

            loaded = repo.load(profile_path)
            assert loaded is not None
            assert loaded == profile

    def test_category_score_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            _create_synthetic_db(db_path, n_per_category=20)

            collections = load_reference_scores(db_path, ["model-a"])
            categories = {c.category for c in collections}
            assert "positivo" in categories
            assert "negativo" in categories

            for c in collections:
                for metric, scores in c.scores.items():
                    assert len(scores) == len(
                        c.session_ids
                    ), f"{c.category}/{metric}: {len(scores)} scores vs {len(c.session_ids)} sessions"

    def test_profile_json_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            profile_path = str(Path(tmpdir) / "profile.json")

            _create_synthetic_db(db_path, n_per_category=25)

            collections = load_reference_scores(db_path, ["model-a", "model-b"])
            profile = derive_thresholds(
                collections,
                bootstrap_b=500,
                seed=42,
                percentile=5,
                alpha=0.05,
                benchmark="test-json",
                reference_models=("model-a", "model-b"),
                reference_run_ids=("run_model-a", "run_model-b"),
            )

            repo = ThresholdProfileRepository()
            repo.save(profile, profile_path)

            raw = json.loads(Path(profile_path).read_text())
            assert raw["version"] == "1.0.0"
            assert raw["bootstrap_b"] == 500
            assert "positivo" in raw["category_thresholds"]
            assert len(raw["global_thresholds"]) > 0

    def test_load_nonexistent_returns_none(self) -> None:
        repo = ThresholdProfileRepository()
        result = repo.load("/tmp/nonexistent_profile_xyzzy.json")
        assert result is None
