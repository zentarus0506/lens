"""Evaluation: datasets, scorers, statistics."""

from .judges import CallableScorer, Contains, ExactMatch, LLMJudge, NumericTolerance, Regex, Score, Scorer
from .runner import Case, CaseResult, Summary, compare_labels, load_dataset, run_eval
from .stats import (
    Comparison,
    Interval,
    bootstrap_ci,
    cohens_kappa,
    judge_reliability,
    mean,
    paired_bootstrap,
    spearman,
    std,
)

__all__ = [
    "CallableScorer",
    "Case",
    "CaseResult",
    "Comparison",
    "Contains",
    "ExactMatch",
    "Interval",
    "LLMJudge",
    "NumericTolerance",
    "Regex",
    "Score",
    "Scorer",
    "Summary",
    "bootstrap_ci",
    "cohens_kappa",
    "compare_labels",
    "judge_reliability",
    "load_dataset",
    "mean",
    "paired_bootstrap",
    "run_eval",
    "spearman",
    "std",
]
