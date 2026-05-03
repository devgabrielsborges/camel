from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import numpy as np

from camel.domain.services.hypothesis_tests import run_all_tests
from camel.domain.services.statistical_verdict import compute_statistical_verdict
from camel.domain.services.threshold_derivation import derive_thresholds
from camel.domain.services.verdict import Verdict
from camel.domain.value_objects.category_score_collection import CategoryScoreCollection
from camel.infrastructure.adapters.duckdb_reference_scores import load_reference_scores
from camel.infrastructure.adapters.threshold_repository import ThresholdProfileRepository


def _create_synthetic_db(db_path: str, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE fct_inference_results (
            session_id VARCHAR, run_id VARCHAR, timestamp TIMESTAMP,
            input VARCHAR, output VARCHAR, model VARCHAR,
            data_category_QA VARCHAR, language BIGINT
        )
    """)
    conn.execute("""
        CREATE TABLE fct_evaluation_scores (
            session_id VARCHAR, run_id VARCHAR, timestamp TIMESTAMP,
            data_category_QA VARCHAR, language BIGINT,
            correctness BOOLEAN, guidelines BOOLEAN,
            token_overlap_f1 DOUBLE, class_exact_match BOOLEAN,
            refusal_detection BOOLEAN, groundedness DOUBLE,
            pass_at_k BOOLEAN, pass_at_k_best_score DOUBLE,
            failure_mode VARCHAR
        )
    """)
    idx = 0
    for model in ("ref-model-a", "ref-model-b"):
        run_id = f"run_{model}"
        for category in ("positivo", "negativo"):
            for _ in range(40):
                sid = f"s_{idx}"
                idx += 1
                overlap = float(rng.normal(0.5, 0.1))
                refusal = bool(rng.choice([True, False], p=[0.1, 0.9]))
                groundedness = float(rng.normal(0.7, 0.15))
                conn.execute(
                    "INSERT INTO fct_inference_results VALUES (?,?,CURRENT_TIMESTAMP,'','',?,?,0)",
                    [sid, run_id, model, category],
                )
                conn.execute(
                    "INSERT INTO fct_evaluation_scores VALUES (?,?,CURRENT_TIMESTAMP,?,0,true,true,?,false,?,?,false,0.0,'')",
                    [sid, run_id, category, overlap, refusal, groundedness],
                )
    conn.close()


_CRITICAL = {
    ("token_overlap_f1", "positivo"),
    ("token_overlap_f1", "negativo"),
    ("refusal_detection", "negativo"),
    ("discrimination_delta", "global"),
}


class TestStatisticalVerdictPipeline:
    def test_capable_model_end_to_end(self) -> None:
        """Derive thresholds from reference → test similar candidate → CAPABLE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            profile_path = str(Path(tmpdir) / "profile.json")
            _create_synthetic_db(db_path)

            ref_collections = load_reference_scores(db_path, ["ref-model-a", "ref-model-b"])
            profile = derive_thresholds(
                ref_collections,
                bootstrap_b=500, seed=42, percentile=5, alpha=0.05,
                benchmark="test",
                reference_models=("ref-model-a", "ref-model-b"),
                reference_run_ids=("run_ref-model-a", "run_ref-model-b"),
            )
            repo = ThresholdProfileRepository()
            repo.save(profile, profile_path)

            rng = np.random.default_rng(99)
            n = 30
            candidate = [
                CategoryScoreCollection(
                    category="positivo",
                    session_ids=tuple(f"c{i}" for i in range(n)),
                    scores={
                        "token_overlap_f1": tuple(rng.normal(0.5, 0.1, n).tolist()),
                        "refusal_detection": tuple(
                            rng.choice([0.0, 1.0], n, p=[0.9, 0.1]).tolist()
                        ),
                    },
                ),
                CategoryScoreCollection(
                    category="negativo",
                    session_ids=tuple(f"c{i}" for i in range(n, 2 * n)),
                    scores={
                        "token_overlap_f1": tuple(rng.normal(0.5, 0.1, n).tolist()),
                        "refusal_detection": tuple(
                            rng.choice([0.0, 1.0], n, p=[0.9, 0.1]).tolist()
                        ),
                    },
                ),
            ]

            loaded_profile = repo.load(profile_path)
            assert loaded_profile is not None

            test_results = run_all_tests(candidate, ref_collections, loaded_profile)
            verdict = compute_statistical_verdict(
                test_results, _CRITICAL, alpha=0.05,
            )
            assert verdict.verdict in {Verdict.CAPABLE, Verdict.INCONCLUSIVE}

    def test_degraded_model_not_capable(self) -> None:
        """Candidate with much worse scores → NOT_CAPABLE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            _create_synthetic_db(db_path)

            ref_collections = load_reference_scores(db_path, ["ref-model-a", "ref-model-b"])
            profile = derive_thresholds(
                ref_collections,
                bootstrap_b=500, seed=42, percentile=5, alpha=0.05,
                benchmark="test",
                reference_models=("ref-model-a", "ref-model-b"),
                reference_run_ids=("run_ref-model-a", "run_ref-model-b"),
            )

            rng = np.random.default_rng(99)
            n = 40
            candidate = [
                CategoryScoreCollection(
                    category="positivo",
                    session_ids=tuple(f"c{i}" for i in range(n)),
                    scores={
                        "token_overlap_f1": tuple(rng.normal(0.05, 0.02, n).tolist()),
                        "refusal_detection": tuple([0.0] * n),
                    },
                ),
                CategoryScoreCollection(
                    category="negativo",
                    session_ids=tuple(f"c{i}" for i in range(n, 2 * n)),
                    scores={
                        "token_overlap_f1": tuple(rng.normal(0.05, 0.02, n).tolist()),
                        "refusal_detection": tuple([0.0] * n),
                    },
                ),
            ]

            test_results = run_all_tests(candidate, ref_collections, profile)
            verdict = compute_statistical_verdict(
                test_results, _CRITICAL, alpha=0.05,
            )
            assert verdict.verdict == Verdict.NOT_CAPABLE
            assert len(verdict.critical_failures) > 0

    def test_profile_round_trip_in_pipeline(self) -> None:
        """Verify derive → save → load → test → verdict round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            profile_path = str(Path(tmpdir) / "profile.json")
            _create_synthetic_db(db_path)

            ref = load_reference_scores(db_path, ["ref-model-a"])
            profile = derive_thresholds(
                ref, bootstrap_b=300, seed=42, percentile=5, alpha=0.05,
                benchmark="test",
                reference_models=("ref-model-a",),
                reference_run_ids=("run_ref-model-a",),
            )

            repo = ThresholdProfileRepository()
            repo.save(profile, profile_path)

            loaded = repo.load(profile_path)
            assert loaded is not None
            assert loaded.version == profile.version

            test_results = run_all_tests(ref, ref, loaded)
            verdict = compute_statistical_verdict(test_results, _CRITICAL, alpha=0.05)
            assert verdict.verdict in {Verdict.CAPABLE, Verdict.INCONCLUSIVE}
