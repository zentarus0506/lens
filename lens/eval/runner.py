"""Dataset loading and batch evaluation.

The runner's contract is deliberately narrow: an agent is a function from a
question to an answer. Anything with more moving parts — retrieval, tools,
multi-step planning — hides inside that function and shows up as spans, which is
exactly where you want to look when a case fails.
"""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..db import Store
from ..trace import Lens
from . import stats
from .judges import Score, Scorer

AgentFn = Callable[[str], str]


@dataclass
class Case:
    id: str
    input: str
    expected: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class CaseResult:
    case_id: str
    output: str
    scores: list[Score]
    run_id: str | None
    latency_ms: float
    error: str | None = None

    @property
    def value(self) -> float:
        """Primary score: the first scorer, or the mean when several ran."""
        if not self.scores:
            return 0.0
        return self.scores[0].value if len(self.scores) == 1 else stats.mean(
            [s.value for s in self.scores]
        )

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scores) if self.scores else False


@dataclass
class Summary:
    batch: str
    label: str
    n: int
    mean_score: float
    interval: stats.Interval
    pass_rate: float
    cost_usd: float
    mean_latency_ms: float
    total_tokens: int
    errors: int
    by_scorer: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "batch": self.batch,
            "label": self.label,
            "n": self.n,
            "mean_score": round(self.mean_score, 4),
            "ci_low": round(self.interval.low, 4),
            "ci_high": round(self.interval.high, 4),
            "pass_rate": round(self.pass_rate, 4),
            "cost_usd": round(self.cost_usd, 6),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "total_tokens": self.total_tokens,
            "errors": self.errors,
            "by_scorer": {k: round(v, 4) for k, v in self.by_scorer.items()},
        }


# ------------------------------------------------------------------ datasets


def load_dataset(path: str | Path) -> list[Case]:
    """Load cases from .jsonl, .json or .csv.

    Accepts `input` or `question` as the prompt field and `expected`, `answer` or
    `reference` as the target, because the three datasets people actually have
    lying around never agree on naming.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"dataset not found: {p}")

    rows: list[dict]
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("cases", [])
    elif p.suffix == ".csv":
        with p.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    else:
        raise ValueError(f"unsupported dataset format: {p.suffix}")

    cases: list[Case] = []
    for index, row in enumerate(rows):
        prompt = row.get("input") or row.get("question") or row.get("prompt")
        if not prompt:
            continue
        cases.append(
            Case(
                id=str(row.get("id") or f"case-{index:03d}"),
                input=str(prompt),
                expected=row.get("expected") or row.get("answer") or row.get("reference"),
                meta={k: v for k, v in row.items()
                      if k not in {"id", "input", "question", "prompt", "expected", "answer", "reference"}},
            )
        )
    return cases


# -------------------------------------------------------------------- runner


def run_eval(
    lens: Lens,
    cases: Sequence[Case],
    agent: AgentFn,
    *,
    label: str,
    batch: str | None = None,
    scorers: Sequence[Scorer] = (),
    concurrency: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> Summary:
    batch_id = batch or f"batch-{int(time.time())}"
    store = lens.store
    results: list[CaseResult] = []

    def _one(case: Case) -> CaseResult:
        started = time.perf_counter()
        run_id: str | None = None
        error: str | None = None
        output = ""
        try:
            with lens.trace("eval-case", input=case.input, label=label, origin="eval",
                            metadata={"batch": batch_id, "case_id": case.id}) as run:
                run_id = run.id
                output = agent(case.input)
                run.output = output
        except Exception as exc:  # noqa: BLE001 - a broken case must not stop the batch
            error = f"{type(exc).__name__}: {exc}"

        latency = (time.perf_counter() - started) * 1000
        scores = []
        for scorer in scorers:
            try:
                scores.append(scorer.score(input=case.input, expected=case.expected,
                                           output=output, meta=case.meta))
            except Exception as exc:  # noqa: BLE001
                scores.append(Score(0.0, False, f"scorer crashed: {exc}", scorer.name))
        return CaseResult(case.id, output, scores, run_id, latency, error)

    def _persist(result: CaseResult) -> None:
        """Write a case the moment it finishes.

        Holding all writes until the batch ends means one slow case or one
        interrupt discards every score computed before it. Traces survive (they
        flush per run), but the measurements — the actual product — do not.
        """
        for score in result.scores:
            store.save_eval_result({
                "batch": batch_id,
                "label": label,
                "case_id": result.case_id,
                "run_id": result.run_id,
                "output": result.output,
                "score": score.value,
                "passed": score.passed,
                "scorer": score.scorer,
            })

    if concurrency <= 1:
        for done, case in enumerate(cases, 1):
            result = _one(case)
            _persist(result)
            results.append(result)
            if progress:
                progress(done, len(cases))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_one, case): case for case in cases}
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                _persist(result)
                results.append(result)
                if progress:
                    progress(done, len(cases))

    # Preserve dataset order so paired comparisons line up case by case.
    order = {case.id: i for i, case in enumerate(cases)}
    results.sort(key=lambda r: order.get(r.case_id, 0))

    values = [r.value for r in results]
    interval = stats.bootstrap_ci(values)
    by_scorer: dict[str, list[float]] = {}
    for result in results:
        for score in result.scores:
            by_scorer.setdefault(score.scorer, []).append(score.value)

    # include_eval=True is required: these runs are tagged origin="eval" and would
    # otherwise be filtered out of the very aggregate we are building here.
    runs = store.list_runs(project=lens.project, limit=len(results) * 4, include_eval=True)
    run_ids = {r.run_id for r in results if r.run_id}
    matched = [r for r in runs if r["id"] in run_ids]

    return Summary(
        batch=batch_id,
        label=label,
        n=len(results),
        mean_score=stats.mean(values),
        interval=interval,
        pass_rate=sum(1 for r in results if r.passed) / len(results) if results else 0.0,
        cost_usd=sum(r["total_cost_usd"] for r in matched),
        mean_latency_ms=stats.mean([r.latency_ms for r in results]),
        total_tokens=sum(r["total_tokens"] for r in matched),
        errors=sum(1 for r in results if r.error),
        by_scorer={name: stats.mean(vals) for name, vals in by_scorer.items()},
    )


# ----------------------------------------------------------------- comparison


def compare_labels(store: Store, batch: str, label_a: str, label_b: str) -> dict:
    """Paired comparison of two labels in the same batch.

    Requires identical case sets — comparing A on ten cases against B on fifteen
    is the most common way evaluation numbers become fiction.
    """
    grouped = store.eval_cases(batch)
    if label_a not in grouped or label_b not in grouped:
        missing = [l for l in (label_a, label_b) if l not in grouped]
        return {"error": f"labels not found in batch: {', '.join(missing)}"}

    a_map = {row["case_id"]: row["score"] for row in grouped[label_a]}
    b_map = {row["case_id"]: row["score"] for row in grouped[label_b]}

    shared = sorted(set(a_map) & set(b_map))
    if not shared:
        return {"error": "no overlapping cases between labels"}
    if len(shared) != len(a_map) or len(shared) != len(b_map):
        return {
            "error": "case sets differ between labels",
            "shared": len(shared),
            "a_only": len(set(a_map) - set(b_map)),
            "b_only": len(set(b_map) - set(a_map)),
        }

    a_scores = [a_map[c] for c in shared]
    b_scores = [b_map[c] for c in shared]
    comparison = stats.paired_bootstrap(a_scores, b_scores)
    return {
        "batch": batch,
        "a": {"label": label_a, "mean": round(stats.mean(a_scores), 4),
              "ci": [round(stats.bootstrap_ci(a_scores).low, 4),
                     round(stats.bootstrap_ci(a_scores).high, 4)]},
        "b": {"label": label_b, "mean": round(stats.mean(b_scores), 4),
              "ci": [round(stats.bootstrap_ci(b_scores).low, 4),
                     round(stats.bootstrap_ci(b_scores).high, 4)]},
        "delta": round(comparison.delta, 4),
        "delta_ci": [round(comparison.low, 4), round(comparison.high, 4)],
        "p_value": round(comparison.p_value, 4),
        "significant": comparison.significant,
        "verdict": comparison.verdict(),
        "wins": comparison.wins,
        "losses": comparison.losses,
        "ties": comparison.ties,
        "n": comparison.n,
    }
