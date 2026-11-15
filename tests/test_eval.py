"""Evaluation runner and label comparison."""

from __future__ import annotations

import json

import pytest

from lens import Lens, Store
from lens.eval import Contains, ExactMatch, compare_labels, load_dataset, run_eval


@pytest.fixture()
def lens(tmp_path):
    return Lens(project="eval-test", store=Store(tmp_path / "eval.db"),
                provider="mock", model="mock-1")


@pytest.fixture()
def dataset(tmp_path):
    path = tmp_path / "cases.jsonl"
    rows = [
        {"id": "a", "input": "span 是什么", "expected": "span", "keywords": [["span"]]},
        {"id": "b", "input": "延迟怎么查", "expected": "latency", "keywords": [["latency"]]},
        {"id": "c", "input": "成本怎么算", "expected": "cost", "keywords": [["cost"]]},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def test_load_dataset_reads_keyword_metadata(dataset):
    cases = load_dataset(dataset)
    assert len(cases) == 3
    assert cases[0].meta["keywords"] == [["span"]]
    assert cases[0].expected == "span"


def test_run_eval_writes_results_and_summary(lens, dataset):
    cases = load_dataset(dataset)
    summary = run_eval(
        lens, cases, lambda q: "cost latency span", label="v1",
        batch="b1", scorers=[Contains()], concurrency=1,
    )
    assert summary.n == 3
    assert summary.mean_score == pytest.approx(1.0)
    assert summary.interval.low <= summary.mean_score <= summary.interval.high
    assert lens.store.eval_summary("b1")[0]["n"] == 3


def test_compare_reports_wins_and_losses(lens, dataset):
    cases = load_dataset(dataset)
    run_eval(lens, cases, lambda q: "cost", label="weak", batch="b2",
             scorers=[Contains()], concurrency=1)
    run_eval(lens, cases, lambda q: "cost latency span", label="strong", batch="b2",
             scorers=[Contains()], concurrency=1)

    result = compare_labels(lens.store, "b2", "weak", "strong")
    assert "error" not in result
    # case a and b: weak misses the keyword, strong hits it -> 2 wins.
    # case c: both contain "cost" -> a tie, which must not be counted as a win.
    assert result["wins"] == 2
    assert result["ties"] == 1
    assert result["losses"] == 0
    assert result["delta"] > 0


def test_compare_rejects_mismatched_case_sets(lens, tmp_path):
    """Comparing labels evaluated on different cases is the classic way numbers lie."""
    cases = [
        type("C", (), {"id": "x", "input": "q", "expected": "q", "meta": {}})(),
        type("C", (), {"id": "y", "input": "q2", "expected": "q2", "meta": {}})(),
    ]
    run_eval(lens, cases[:2], lambda q: q, label="full", batch="b3",
             scorers=[ExactMatch()], concurrency=1)
    run_eval(lens, cases[:1], lambda q: q, label="partial", batch="b3",
             scorers=[ExactMatch()], concurrency=1)

    result = compare_labels(lens.store, "b3", "full", "partial")
    assert "error" in result
    assert "differ" in result["error"]


def test_scorer_crash_does_not_abort_batch(lens, dataset):
    class Exploding:
        name = "boom"

        def score(self, **kwargs):
            raise ValueError("bad scorer")

    summary = run_eval(lens, load_dataset(dataset), lambda q: "x", label="v",
                       batch="b4", scorers=[Exploding()], concurrency=1)
    assert summary.n == 3
    assert summary.mean_score == 0.0


def test_rerunning_a_label_overwrites_instead_of_duplicating(lens, dataset):
    """Re-running a batch must not silently double the sample size.

    Duplicated cases inflate n and narrow the confidence interval, which makes a
    noisy result look statistically solid — the worst possible failure mode for
    a measurement tool.
    """
    cases = load_dataset(dataset)
    for _ in range(2):
        run_eval(lens, cases, lambda q: "cost latency span", label="v1",
                 batch="dup", scorers=[Contains()], concurrency=1)

    summary = lens.store.eval_summary("dup")
    assert len(summary) == 1, "duplicate label rows"
    assert summary[0]["n"] == 3, f"expected 3 cases, got {summary[0]['n']}"


def test_interrupted_batch_keeps_completed_scores(lens, tmp_path):
    """Scores must be durable per case, not flushed only at the end of a batch.

    A long run against a slow provider gets interrupted all the time (timeout,
    ctrl-c, laptop lid). Losing the completed measurements makes the expensive
    part of the run worthless.
    """
    path = tmp_path / "three.jsonl"
    rows = [{"id": f"c{i}", "input": f"q{i}", "expected": "cost"} for i in range(3)]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    calls = {"n": 0}

    def agent(_q: str) -> str:
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return "cost"

    with pytest.raises(KeyboardInterrupt):
        run_eval(lens, load_dataset(path), agent, label="v", batch="cut",
                 scorers=[Contains()], concurrency=1)

    summary = lens.store.eval_summary("cut")
    assert summary, "no scores survived the interrupt"
    assert summary[0]["n"] == 2, f"expected the 2 finished cases, got {summary[0]['n']}"
