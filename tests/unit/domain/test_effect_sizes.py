from __future__ import annotations

import math

import numpy as np

from camel.domain.services.effect_sizes import classify_magnitude, cohens_d, odds_ratio


class TestCohensD:
    def test_identical_distributions(self) -> None:
        a = (0.5, 0.5, 0.5, 0.5)
        b = (0.5, 0.5, 0.5, 0.5)
        d = cohens_d(a, b)
        assert abs(d) < 1e-10

    def test_known_shift(self) -> None:
        """Two groups with known means and equal std should yield d ≈ (mean_a - mean_b) / std."""
        rng = np.random.default_rng(42)
        a = tuple(rng.normal(1.0, 1.0, 1000).tolist())
        b = tuple(rng.normal(0.2, 1.0, 1000).tolist())
        d = cohens_d(a, b)
        assert 0.6 < d < 1.0, f"Expected d ≈ 0.8, got {d}"

    def test_negative_shift(self) -> None:
        rng = np.random.default_rng(42)
        a = tuple(rng.normal(0.0, 1.0, 500).tolist())
        b = tuple(rng.normal(0.8, 1.0, 500).tolist())
        d = cohens_d(a, b)
        assert d < 0

    def test_returns_float(self) -> None:
        d = cohens_d((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
        assert isinstance(d, float)


class TestOddsRatio:
    def test_known_table(self) -> None:
        """2x2 table: [[10, 5], [3, 12]] → OR = b/c = 5/3 ≈ 1.667."""
        table = ((10, 5), (3, 12))
        or_val = odds_ratio(table)
        assert abs(or_val - 5.0 / 3.0) < 1e-10

    def test_zero_cell_handling(self) -> None:
        """When c=0, should use continuity correction (add 0.5)."""
        table = ((10, 5), (0, 15))
        or_val = odds_ratio(table)
        assert or_val > 0 and math.isfinite(or_val)

    def test_symmetric_table(self) -> None:
        """When b == c, odds ratio should be 1.0."""
        table = ((10, 5), (5, 10))
        or_val = odds_ratio(table)
        assert abs(or_val - 1.0) < 1e-10

    def test_returns_float(self) -> None:
        table = ((10, 5), (3, 12))
        assert isinstance(odds_ratio(table), float)


class TestClassifyMagnitude:
    def test_negligible_continuous(self) -> None:
        assert classify_magnitude(0.1, "continuous") == "negligible"

    def test_small_continuous(self) -> None:
        assert classify_magnitude(0.3, "continuous") == "small"

    def test_medium_continuous(self) -> None:
        assert classify_magnitude(0.6, "continuous") == "medium"

    def test_large_continuous(self) -> None:
        assert classify_magnitude(0.9, "continuous") == "large"

    def test_negligible_binary(self) -> None:
        assert classify_magnitude(1.2, "binary") == "negligible"

    def test_small_binary(self) -> None:
        assert classify_magnitude(1.6, "binary") == "small"

    def test_medium_binary(self) -> None:
        assert classify_magnitude(2.5, "binary") == "medium"

    def test_large_binary(self) -> None:
        assert classify_magnitude(3.5, "binary") == "large"

    def test_negative_uses_absolute_value_continuous(self) -> None:
        assert classify_magnitude(-0.9, "continuous") == "large"
