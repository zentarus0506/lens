"""End-to-end evaluation demo: one prompt strategy against another.

The agent here is a *test double* running on the deterministic mock provider, so
this script costs nothing and always produces the same numbers. What is real is
the machinery: paired cases, bootstrap intervals, and a significance verdict.

The comparison is intentionally honest about small samples — swap `mock` for a
real provider and the same code measures a real prompt change.

Run:
    python examples/run_eval.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lens import Lens  # noqa: E402
from lens.eval import Contains, compare_labels, load_dataset, run_eval  # noqa: E402
from lens.eval.stats import bootstrap_ci  # noqa: E402

DATASET = Path(__file__).parent / "datasets" / "qa_demo.jsonl"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", help="reuse an existing batch id so labels stay comparable")
    parser.add_argument("--only", choices=["v1", "v2"], help="run a single variant")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    # Follows .env: real provider when a key is configured, deterministic mock
    # otherwise. The comparison machinery is identical either way.
    lens = Lens(project="eval-demo")
    cases = load_dataset(DATASET)
    scorer = Contains()
    batch = args.batch or f"demo-{time.strftime('%m%d-%H%M')}"

    # A priced model name so the cost column is real arithmetic against the
    # models.dev table. Tokens are simulated by the mock provider; the price is not.
    PRICED_MODEL = "deepseek-ai/DeepSeek-V3.2"

    def agent_single(question: str) -> str:
        """Baseline: one sample, take what you get."""
        return lens.llm([{"role": "user", "content": question}],
                        seed=0, model=PRICED_MODEL).text

    def agent_best_of_3(question: str) -> str:
        """Sample three times and keep the richest answer.

        This is the cheapest real improvement most teams can make, and it triples
        the cost — which is exactly the trade-off the cost columns should expose.
        """
        candidates = [
            lens.llm([{"role": "user", "content": question}],
                     seed=seed, model=PRICED_MODEL).text
            for seed in range(3)
        ]
        return max(candidates, key=len)

    print(f"数据集 {DATASET.name}: {len(cases)} 条用例 · batch={batch}\n")

    variants = [("v1-single", agent_single), ("v2-best-of-3", agent_best_of_3)]
    if args.only:
        variants = [v for v in variants if v[0].startswith(args.only)]

    for name, agent in variants:
        summary = run_eval(
            lens, cases, agent,
            label=name, batch=batch,
            scorers=[scorer],
            # Modest parallelism: real providers rate-limit, and a rate-limit error
            # would be recorded as a failed case rather than as throttling.
            concurrency=args.concurrency,
            progress=lambda d, t: print(f"\r  运行 {name}: {d}/{t}", end="", flush=True),
        )
        print(f"\r  {name:14} 均分 {summary.mean_score:.3f} "
              f"[{summary.interval.low:.3f}, {summary.interval.high:.3f}] "
              f"通过率 {summary.pass_rate:.0%} "
              f"成本 ${summary.cost_usd:.6f} "
              f"延迟 {summary.mean_latency_ms:.1f}ms")

    print()
    result = compare_labels(lens.store, batch, "v1-single", "v2-best-of-3")
    if "error" in result:
        print("对比失败:", result["error"])
        return 1

    a, b = result["a"], result["b"]
    print("配对比较 (v2 - v1)")
    print(f"  v1 均分 {a['mean']:.3f}  CI [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]")
    print(f"  v2 均分 {b['mean']:.3f}  CI [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}]")
    print(f"  差值 {result['delta']:+.3f}  CI [{result['delta_ci'][0]:+.3f}, {result['delta_ci'][1]:+.3f}]  "
          f"p={result['p_value']:.4f}")
    print(f"  胜/负/平 {result['wins']}/{result['losses']}/{result['ties']}  "
          f"→ {result['verdict']}")
    print(f"\n  注: n={result['n']}，区间宽度反映样本量。用例增加到 200 条后区间会明显收窄。")
    print(f"\n数据已写入 {lens.store.path}")
    print("打开看板: python -m lens.server.app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
