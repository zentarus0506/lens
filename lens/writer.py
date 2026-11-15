"""Background batch writer.

Two production problems the synchronous path has:

1. **fsync per span.** Every `save_span` commits, so a service doing 200 requests
   per second performs 200 commits per second. SQLite survives this but the
   write-ahead log grows and lock contention shows up as latency.
2. **Back-pressure into the request path.** If the database is slow, a synchronous
   write makes the API slow. Observability must never be the reason a request
   fails.

This writer puts spans on a bounded queue, drains them on a background thread in
batches, and drops (with a counter) when the queue is full. Dropping telemetry
under load is the correct trade: losing a trace is annoying, losing a customer
request is not.

Deliberately not using a thread per write or `asyncio`: the SDK is called from
plain synchronous code, and a single writer thread keeps ordering obvious.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

from .db import Store


@dataclass
class WriterStats:
    spans_written: int = 0
    runs_written: int = 0
    spans_dropped: int = 0
    flushes: int = 0
    last_flush_ms: float = 0.0
    errors: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "spans_written": self.spans_written,
            "runs_written": self.runs_written,
            "spans_dropped": self.spans_dropped,
            "flushes": self.flushes,
            "last_flush_ms": round(self.last_flush_ms, 2),
            "errors": self.errors,
        }


class AsyncWriter:
    def __init__(
        self,
        store: Store,
        *,
        batch_size: int = 50,
        flush_interval: float = 1.0,
        max_queue: int = 10_000,
    ) -> None:
        self.store = store
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.stats = WriterStats()
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lens-writer", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------- producers

    def submit_run(self, run: dict) -> None:
        self._put(("run", run))

    def submit_spans(self, spans: list[dict]) -> None:
        if spans:
            self._put(("spans", spans))

    def _put(self, item: tuple[str, object]) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Never block the caller: a full queue means the database is behind,
            # and dropping telemetry is cheaper than stalling the request.
            if item[0] == "spans":
                self.stats.spans_dropped += len(item[1])  # type: ignore[arg-type]
            else:
                self.stats.spans_dropped += 1

    # --------------------------------------------------------------- worker

    def _run(self) -> None:
        pending_spans: list[dict] = []
        pending_runs: list[dict] = []
        last_flush = time.monotonic()

        while not self._stop.is_set() or not self._queue.empty():
            timeout = max(0.05, self.flush_interval - (time.monotonic() - last_flush))
            try:
                kind, payload = self._queue.get(timeout=timeout)
                if kind == "spans":
                    pending_spans.extend(payload)  # type: ignore[arg-type]
                else:
                    pending_runs.append(payload)  # type: ignore[arg-type]
            except queue.Empty:
                pass

            due = (
                len(pending_spans) >= self.batch_size
                or (time.monotonic() - last_flush) >= self.flush_interval
                or (self._stop.is_set() and (pending_spans or pending_runs))
            )
            if due and (pending_spans or pending_runs):
                self._flush(pending_runs, pending_spans)
                pending_runs, pending_spans = [], []
                last_flush = time.monotonic()

        if pending_spans or pending_runs:
            self._flush(pending_runs, pending_spans)

    def _flush(self, runs: list[dict], spans: list[dict]) -> None:
        started = time.perf_counter()
        try:
            # Runs first: spans carry a foreign key, and a missing run would make
            # the span insert fail on a database with FK enforcement on.
            for run in runs:
                self.store.save_run(run)
                self.stats.runs_written += 1
            if spans:
                self.store.save_spans_batch(spans)
                self.stats.spans_written += len(spans)
            self.stats.flushes += 1
            self.stats.last_flush_ms = (time.perf_counter() - started) * 1000
        except Exception:  # noqa: BLE001 - a write failure must not kill the thread
            self.stats.errors += 1

    # --------------------------------------------------------------- control

    def flush(self) -> None:
        """Block until the queue is drained. Call before reading in tests."""
        deadline = time.monotonic() + 5.0
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(self.flush_interval * 0.2)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
