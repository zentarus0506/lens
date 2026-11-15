"""Statistical guarantees the dashboard depends on."""

from __future__ import annotations

import random

import pytest

from lens.eval import stats


def test_bootstrap_interval_on_constant_values_collapses():
    ci = stats.bootstrap_ci([0.5] * 10)
    assert ci.low == pytest.approx(0.5)
    assert ci.high == pytest.approx(0.5)


def test_bootstrap_interval_is_reproducible():
    values = [0.1, 0.4, 0.9, 0.2, 0.7]
    assert stats.bootstrap_ci(values, seed=7) == stats.bootstrap_ci(values, seed=7)


def test_interval_widens_with_small_samples():
    """A three-case run must not produce a confident-looking interval."""
    ci = stats.bootstrap_ci([0.0, 1.0, 1.0])
    assert ci.high - ci.low > 0.5


def test_paired_bootstrap_detects_total_separation():
    comparison = stats.paired_bootstrap([0.0] * 20, [1.0] * 20)
    assert comparison.p_value < 0.05
    assert comparison.wins == 20
    assert comparison.verdict() == "B is better"


def test_paired_bootstrap_does_not_call_noise_significant():
    rng = random.Random(3)
    a = [rng.random() for _ in range(25)]
    b = [x + rng.gauss(0, 0.05) for x in a]
    comparison = stats.paired_bootstrap(a, b)
    assert not comparison.significant


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        stats.paired_bootstrap([0.1, 0.2], [0.1])


def test_kappa_rewards_agreement_and_punishes_opposition():
    assert stats.cohens_kappa([1, 0, 1, 1, 0], [1, 0, 1, 1, 0]) == pytest.approx(1.0)
    assert stats.cohens_kappa([1, 1, 1, 0, 0], [0, 0, 0, 1, 1]) < 0


def test_kappa_corrects_for_class_imbalance():
    """90% surface agreement, but both raters default to 'pass' — chance inflated it."""
    judge = [1] * 9 + [0]
    human = [1] * 8 + [0, 0]
    kappa = stats.cohens_kappa(judge, human)
    agreement = 9 / 10
    assert agreement > 0.85
    assert kappa < 0.8


def test_spearman_detects_monotonic_relationship():
    assert stats.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_judge_reliability_grades_are_ordered():
    strong = stats.judge_reliability([1, 0, 1, 0, 1], [1, 0, 1, 0, 1])
    weak = stats.judge_reliability([1, 1, 1, 1, 0], [1, 1, 1, 0, 0])
    assert strong["reliability"] == "strong"
    assert weak["kappa"] < strong["kappa"]
