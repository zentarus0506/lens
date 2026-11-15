"""Measure what batching actually buys.

The claim is that one transaction per span is the wrong shape for a service. This
turns that claim into a number in two ways:

* **write-only** — isolates the persistence path, which is where the claim lives.
  A synchronous writer commits twice per trace (run + spans); the batching writer
  commits once per N traces.
* **end-to-end** — includes the simulated model call, which is what a caller
  actually experiences, and which dilutes the write savings.

Reporting both keeps the honest framing: batching makes writes much cheaper, and
that matters more the less time the model itself takes.

Run:
    python examples/bench_writer.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lens import Lens, Store  # noqa: E402
from lens.db import new_id  # noqa: E402


def _fake_run(index: int) -> dict:
    now = time.time()
    return {"id": new_id("run"), "project": "bench", "name": f"r{index}",
            "status": "ok", "started_at": now, "ended_at": now,
            "duration_ms": 1.0, "total_cost_usd": 0.0, "total_tokens": 0,
            "origin": "app"}


def _fake_span(run_id: str, index: int) -> dict:
    return {"id": new_id("sp"), "run_id": run_id, "name": f"llm:mock-{index}",
            "kind": "llm", "status": "ok", "started_at": time.time(),
            "duration_ms": 1.0, "model": "mock-1", "provider": "mock",
            "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.0,
            "output": "x" * 200, "attributes": {"i": index}}


def bench_write_only(workdir: Path, n: int, *, batched: bool, batch_size: int = 100) -> float:
    store = Store(workdir / ("w_async.db" if batched else "w_sync.db"))
    lens = Lens(project="bench", store=store, provider="mock", model="mock-1",
                async_write=batched,
                writer_kwargs={"batch_size": batch_size, "flush_interval": 0.2} if batched else None)

    started = time.perf_counter()
    for i in range(n):
        run = _fake_run(i)
        spans = [_fake_span(run["id"], i), _fake_span(run["id"], i)]
        if lens.writer is not None:
            lens.writer.submit_run(run)
            lens.writer.submit_spans(spans)
        else:
            store.save_run(run)
            store.save_spans_batch(spans)
    if lens.writer is not None:
        lens.flush()
    lens.close()
    return time.perf_counter() - started


def bench_end_to_end(workdir: Path, n: int, *, batched: bool) -> float:
    store = Store(workdir / ("e_async.db" if batched else "e_sync.db"))
    lens = Lens(project="bench", store=store, provider="mock", model="mock-1",
                async_write=batched,
                writer_kwargs={"batch_size": 100, "flush_interval": 0.2} if batched else None)

    started = time.perf_counter()
    for i in range(n):
        with lens.trace(f"bench-{i}"):
            lens.llm([{"role": "user", "content": f"question {i}"}])
    if lens.writer is not None:
        lens.flush()
    lens.close()
    return time.perf_counter() - started


def main() -> int:
    n = 400
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        w_sync = bench_write_only(workdir, n, batched=False)
        w_async = bench_write_only(workdir, n, batched=True)
        e_sync = bench_end_to_end(workdir, n, batched=False)
        e_async = bench_end_to_end(workdir, n, batched=True)

    print(f"{n} 次执行，每次 1 个 run + 2 个 span\n")
    print(f"{'场景':22} {'同步':>10} {'异步批量':>10} {'提升':>8}")
    print("-" * 54)
    print(f"{'纯写入 (run/s)':22} {n / w_sync:>10.0f} {n / w_async:>10.0f} {w_sync / w_async:>7.1f}x")
    print(f"{'端到端 (run/s)':22} {n / e_sync:>10.0f} {n / e_async:>10.0f} {e_sync / e_async:>7.1f}x")
    print()
    print(f"纯写入耗时：同步 {w_sync:.2f}s → 异步 {w_async:.2f}s")
    print("端到端含模拟模型调用（每次约 3ms），所以写入收益被稀释——")
    print("真实场景中模型越慢，写入优化的相对收益越小，但绝对写入成本不变。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
