"""A small research-style agent instrumented end to end.

It is deliberately a *stub* agent — no network, no API key — but the shape of
what it records is the shape a real one produces: a planning step, parallel
retrieval, two model calls, and a synthesis step. That shape is the thing worth
looking at on the dashboard.

Run:
    python examples/demo_agent.py
    python examples/demo_agent.py --fault   # inject a failure to see error tracing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lens import Lens  # noqa: E402
from lens.providers import ProviderError  # noqa: E402

QUESTIONS = [
    "为什么 LLM 应用需要链路追踪？",
    "如何判断新版 prompt 比旧版更好？",
    "什么情况下应该用更小的模型？",
    "怎么降低每次请求的 token 成本？",
    "如何验证自动评分器是否可信？",
]


def run_once(lens: Lens, question: str, *, inject_fault: bool = False) -> str:
    with lens.trace("research", input=question, label="demo") as run:
        with lens.span("plan", kind="custom", input=question) as plan:
            plan.set(strategy="retrieve-then-synthesize")

            with lens.span("retrieve:docs", kind="retrieval", input=question) as retrieval:
                retrieval.set(docs=4, index="local")
                retrieval.set(top_score=0.82)

            evidence = lens.llm(
                [
                    {"role": "system", "content": "You summarise retrieved evidence."},
                    {"role": "user", "content": f"归纳要点: {question}"},
                ],
                # Must stay inside the siliconflow voucher whitelist.
                model="Qwen/Qwen3-14B",
                name="llm:summarise",
            )
            plan.set(evidence_chars=len(evidence.text))

            if inject_fault:
                with lens.span("tool:web-search", kind="tool") as tool:
                    raise ProviderError("simulated upstream timeout", status=504)

        answer = lens.llm(
            [
                {"role": "system", "content": "Answer using only the evidence."},
                {"role": "user", "content": f"{question}\n\n证据: {evidence.text}"},
            ],
            # An explicit priced model so the cost columns show real arithmetic
            # against the models.dev table, even though tokens are simulated.
            model="deepseek-ai/DeepSeek-V3.2",
            name="llm:answer",
            temperature=0.2,
        )

        with lens.span("format", kind="custom"):
            final = answer.text

        run.output = final
        return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", action="store_true", help="inject a tool failure")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--project", default="demo")
    args = parser.parse_args()

    # provider=None follows .env / environment instead of forcing a channel,
    # so the same script runs against mock or a real provider unchanged.
    lens = Lens(project=args.project, provider=os.environ.get("LENS_DEMO_PROVIDER") or None)
    total_cost = 0.0
    failures = 0

    for i in range(args.rounds):
        question = QUESTIONS[i % len(QUESTIONS)]
        try:
            text = run_once(lens, question, inject_fault=args.fault and i == 0)
            print(f"[{i + 1}/{args.rounds}] {question}\n           -> {text[:80]}...")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[{i + 1}/{args.rounds}] {question}\n           -> FAILED: {type(exc).__name__}: {exc}")

    for run in lens.store.list_runs(project=args.project, limit=args.rounds):
        total_cost += run["total_cost_usd"]

    print(f"\n记录 {args.rounds} 次执行，失败 {failures} 次，累计成本 ${total_cost:.6f}")
    print(f"数据库: {lens.store.path}")
    print("打开看板: lens-server 或 python -m lens.server.app")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
