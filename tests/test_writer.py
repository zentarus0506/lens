"""Sampling and the async writer.

These cover the two properties that make Lens usable in a service rather than
only on a laptop: a toggle that cannot change program behaviour, and a writer
that cannot block the request path.
"""

from __future__ import annotations

import queue

import pytest

from lens import Lens, Store
from lens.writer import AsyncWriter, WriterStats


def make_lens(tmp_path, name="t.db", **kwargs) -> Lens:
    return Lens(project="test", store=Store(tmp_path / name),
                provider="mock", model="mock-1", **kwargs)


# ------------------------------------------------------------------- sampling


def test_sample_rate_one_keeps_everything(tmp_path):
    lens = make_lens(tmp_path, sample_rate=1.0)
    with lens.trace("kept"):
        lens.llm([{"role": "user", "content": "hi"}])
    assert len(lens.store.list_runs()) == 1


def test_sample_rate_zero_writes_nothing(tmp_path):
    lens = make_lens(tmp_path, sample_rate=0.0)
    with lens.trace("dropped") as run:
        lens.llm([{"role": "user", "content": "hi"}])
    assert run.sampled is False
    assert lens.store.list_runs() == []


def test_sampling_is_deterministic(tmp_path):
    """A retried run must get the same decision, and an id must reproduce it."""
    lens = make_lens(tmp_path, sample_rate=0.5)
    ids = [f"run_{i:04d}" for i in range(200)]
    first = [lens.is_sampled(i) for i in ids]
    second = [lens.is_sampled(i) for i in ids]
    assert first == second


def test_sampling_rate_is_approximately_honest(tmp_path):
    lens = make_lens(tmp_path, sample_rate=0.3)
    hits = sum(1 for i in range(5000) if lens.is_sampled(f"run_{i}"))
    assert 0.27 < hits / 5000 < 0.33, f"got {hits / 5000:.3f}"


def test_unsampled_run_still_executes_user_code(tmp_path):
    """Sampling must not change behaviour, only persistence.

    A toggle that silently skips work would make tests pass for the wrong reason.
    """
    lens = make_lens(tmp_path, sample_rate=0.0)
    called = {"n": 0}
    with lens.trace("r") as run:
        with lens.span("step"):
            called["n"] += 1
        result = lens.llm([{"role": "user", "content": "hi"}])
    assert called["n"] == 1
    assert result.text
    assert len(run.spans) == 2
    assert lens.store.list_runs() == []


# --------------------------------------------------------------------- writer


def test_async_writer_persists_after_flush(tmp_path):
    lens = make_lens(tmp_path, name="async.db", async_write=True)
    try:
        with lens.trace("buffered"):
            lens.llm([{"role": "user", "content": "hi"}])
        lens.flush()
        lens.close()

        store = Store(tmp_path / "async.db")
        runs = store.list_runs()
        assert len(runs) == 1
        detail = store.get_run(runs[0]["id"])
        assert len(detail["spans"]) == 1
        assert detail["spans"][0]["kind"] == "llm"
    finally:
        lens.close()


def test_async_writer_recomputes_totals_once_per_run(tmp_path):
    lens = make_lens(tmp_path, name="totals.db", async_write=True)
    try:
        with lens.trace("two-spans") as run:
            lens.llm([{"role": "user", "content": "one"}])
            lens.llm([{"role": "user", "content": "two"}])
        expected_tokens = sum(s.prompt_tokens + s.completion_tokens for s in run.spans)
        lens.flush()
        lens.close()

        saved = Store(tmp_path / "totals.db").list_runs()[0]
        assert saved["total_tokens"] == expected_tokens
    finally:
        lens.close()


def test_writer_counts_drops_when_queue_is_full(tmp_path):
    """Dropping telemetry beats blocking a request, but it must be counted.

    Built by hand rather than by racing a live worker, so the assertion does not
    depend on thread timing.
    """
    writer = AsyncWriter.__new__(AsyncWriter)
    writer._queue = queue.Queue(maxsize=1)
    writer.stats = WriterStats()
    writer._queue.put_nowait(("spans", []))

    writer.submit_spans([{"id": "s1", "run_id": "r", "name": "x", "started_at": 0.0}])
    assert writer.stats.spans_dropped == 1


def test_writer_stats_shape(tmp_path):
    lens = make_lens(tmp_path, name="stats.db", async_write=True)
    try:
        with lens.trace("r"):
            lens.llm([{"role": "user", "content": "hi"}])
        lens.flush()
        stats = lens.writer_stats()
        assert stats is not None
        for key in ("spans_written", "runs_written", "spans_dropped", "flushes", "errors"):
            assert key in stats
    finally:
        lens.close()


def test_sync_mode_has_no_writer(tmp_path):
    lens = make_lens(tmp_path, name="sync.db")
    assert lens.writer is None
    assert lens.writer_stats() is None
    lens.flush()  # must not raise
