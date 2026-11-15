"""Tracing behaviour: nesting, error propagation, and the implicit-run safety net."""

from __future__ import annotations

import pytest

from lens import Lens, Store


@pytest.fixture()
def lens(tmp_path):
    # Model is explicit on purpose: without it the fixture inherits LENS_MODEL
    # from .env, and the pricing assertions below would depend on the operator's
    # local configuration instead of on the code under test.
    return Lens(project="test", store=Store(tmp_path / "test.db"),
                provider="mock", model="mock-1")


def test_nested_spans_keep_their_parent(lens):
    with lens.trace("outer") as run:
        with lens.span("plan", kind="custom") as plan:
            with lens.span("retrieve", kind="retrieval") as inner:
                pass
        assert inner.parent_id == plan.id
        assert plan.parent_id is None

    assert len(run.spans) == 2
    assert {s.run_id for s in run.spans} == {run.id}


def test_llm_span_records_tokens_and_cost(lens):
    with lens.trace("call") as run:
        result = lens.llm([{"role": "user", "content": "hello there"}])

    assert result.text
    llm_span = [s for s in run.spans if s.kind == "llm"][0]
    assert llm_span.prompt_tokens > 0
    assert llm_span.completion_tokens > 0
    assert llm_span.model
    # mock-1 is not in the price table, so pricing must be reported as unknown
    # rather than silently costing zero.
    assert llm_span.attributes.get("pricing") == "unknown"


def test_error_propagates_to_parent_and_run(lens):
    with pytest.raises(RuntimeError):
        with lens.trace("failing") as run:
            with lens.span("step", kind="custom"):
                raise RuntimeError("boom")

    saved = lens.store.list_runs(project="test")[0]
    assert saved["status"] == "error"
    assert "boom" in (saved["error"] or "")

    detail = lens.store.get_run(saved["id"])
    assert all(s["status"] == "error" for s in detail["spans"])


def test_llm_without_trace_is_still_recorded(lens):
    """The safety net: forgetting to open a trace must not lose the call."""
    lens.llm([{"role": "user", "content": "unwrapped"}])
    runs = lens.store.list_runs(project="test")
    assert len(runs) == 1
    detail = lens.store.get_run(runs[0]["id"])
    assert len(detail["spans"]) == 1


def test_run_totals_are_summed_from_spans(lens):
    with lens.trace("two-calls") as run:
        lens.llm([{"role": "user", "content": "one"}])
        lens.llm([{"role": "user", "content": "two"}])

    saved = lens.store.list_runs(project="test")[0]
    expected_tokens = sum(s.prompt_tokens + s.completion_tokens for s in run.spans)
    assert saved["total_tokens"] == expected_tokens
    assert len(run.spans) == 2


def test_overview_percentiles_are_ordered(lens):
    for i in range(5):
        with lens.trace(f"run-{i}"):
            lens.llm([{"role": "user", "content": f"q{i}"}])

    overview = lens.store.overview(project="test")
    assert overview["runs"] == 5
    assert overview["p50_latency_ms"] <= overview["p95_latency_ms"]
