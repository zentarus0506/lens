"""Statistics for evaluation results.

The reason this module exists: showing that variant B scored 0.81 and variant A
scored 0.78 is not evidence. With twenty test cases, that gap is noise. Every
comparison here reports an interval, because an interval is the difference
between "B looks better" and "B is better, and here is how much we would bet".

All resampling is seeded so a rerun reproduces the same interval — an
irreproducible p-value is worse than none.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n: int

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True)
class Comparison:
    delta: float          # b - a
    low: float
    high: float
    p_value: float
    n: int
    wins: int             # cases where b > a
    losses: int           # cases where b < a
    ties: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def verdict(self, higher_is_better: bool = True) -> str:
        if not self.significant:
            return "no measurable difference"
        better = self.delta > 0 if higher_is_better else self.delta < 0
        return "B is better" if better else "A is better"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def bootstrap_ci(
    values: list[float],
    *,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap for the mean.

    Chosen over a t-test because scores are bounded (0..1), often skewed, and
    frequently bimodal — a normality assumption would be fiction.
    """
    if not values:
        return Interval(0.0, 0.0, 0.0, 0)
    if len(values) == 1:
        return Interval(values[0], values[0], values[0], 1)

    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        estimates.append(total / n)
    estimates.sort()

    low_idx = int(resamples * (alpha / 2))
    high_idx = min(int(resamples * (1 - alpha / 2)), resamples - 1)
    return Interval(mean(values), estimates[low_idx], estimates[high_idx], n)


def paired_bootstrap(
    a: list[float],
    b: list[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> Comparison:
    """Compare two variants evaluated on the *same* cases.

    Pairing matters: case difficulty dominates score variance, so resampling the
    paired differences removes that confound and yields a far tighter interval
    than comparing two independent means would.
    """
    if len(a) != len(b):
        raise ValueError(f"paired comparison needs equal lengths, got {len(a)} and {len(b)}")
    if not a:
        return Comparison(0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0)

    diffs = [yb - ya for ya, yb in zip(a, b)]
    delta = mean(diffs)
    wins = sum(1 for d in diffs if d > 1e-9)
    losses = sum(1 for d in diffs if d < -1e-9)
    ties = len(diffs) - wins - losses

    rng = random.Random(seed)
    n = len(diffs)
    boot = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        boot.append(total / n)
    boot.sort()

    low_idx = int(resamples * 0.025)
    high_idx = min(int(resamples * 0.975), resamples - 1)
    low, high = boot[low_idx], boot[high_idx]

    # Two-sided p-value: how often the resampled mean crosses zero.
    extreme = sum(1 for v in boot if v <= 0.0)
    p_value = min(1.0, 2 * min(extreme, resamples - extreme) / resamples)

    return Comparison(delta, low, high, p_value, n, wins, losses, ties)


def cohens_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    """Agreement between two raters, corrected for chance.

    Used to answer: can an automated judge be trusted as a stand-in for a human?
    A raw agreement rate of 80% sounds fine until you notice that both raters
    said "pass" for 79% of cases, in which case chance alone gets you there.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same items")
    if not rater_a:
        return 0.0

    n = len(rater_a)
    observed = sum(1 for x, y in zip(rater_a, rater_b) if x == y) / n
    labels = set(rater_a) | set(rater_b)
    expected = 0.0
    for label in labels:
        pa = sum(1 for x in rater_a if x == label) / n
        pb = sum(1 for y in rater_b if y == label) / n
        expected += pa * pb
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, used to check that a judge's scores track human ranking
    even when the absolute values disagree."""
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                result[order[k]] = avg_rank
            i = j + 1
        return result

    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return num / den if den else 0.0


def judge_reliability(judge_labels: list[int], human_labels: list[int]) -> dict:
    """A single place to answer 'is this judge trustworthy enough to gate a release'."""
    kappa = cohens_kappa(judge_labels, human_labels)
    agreement = (
        sum(1 for a, b in zip(judge_labels, human_labels) if a == b) / len(judge_labels)
        if judge_labels
        else 0.0
    )
    grade = "unusable" if kappa < 0.2 else "weak" if kappa < 0.4 else "moderate" if kappa < 0.6 else "substantial" if kappa < 0.8 else "strong"
    return {
        "n": len(judge_labels),
        "agreement": round(agreement, 4),
        "kappa": round(kappa, 4),
        "reliability": grade,
    }
