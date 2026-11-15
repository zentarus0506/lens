"""Scorers.

Two families, deliberately kept apart:

* **Deterministic** scorers (exact, contains, regex, numeric) are cheap, instant,
  and never lie — but they only work when correctness has a machine-checkable
  shape.
* **Model judges** cover open-ended answers, but they are only useful if you can
  prove they agree with a human. `judge_reliability()` in `stats.py` is the
  companion piece; a judge with no calibration number is an opinion generator.

Every scorer returns the same `Score` shape so the runner never branches on type.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence


@dataclass
class Score:
    value: float                 # normalised to 0..1
    passed: bool
    reason: str = ""
    scorer: str = ""

    def __post_init__(self) -> None:
        self.value = max(0.0, min(1.0, float(self.value)))


class Scorer(Protocol):
    name: str

    def score(self, *, input: str, expected: str | None, output: str, meta: dict) -> Score:
        ...


# --------------------------------------------------------------- deterministic


class ExactMatch:
    def __init__(self, case_sensitive: bool = False, strip: bool = True) -> None:
        self.name = f"exact_match(cs={case_sensitive})"
        self.case_sensitive = case_sensitive
        self.strip = strip

    def _norm(self, text: str) -> str:
        value = text.strip() if self.strip else text
        return value if self.case_sensitive else value.lower()

    def score(self, *, input: str, expected: str | None, output: str, meta: dict) -> Score:
        if expected is None:
            return Score(0.0, False, "no reference answer", self.name)
        hit = self._norm(expected) == self._norm(output)
        return Score(1.0 if hit else 0.0, hit, "" if hit else "answers differ", self.name)


class Contains:
    """All keyword groups must appear. A group is a list of synonyms — any one
    of them satisfies it, which is how you encode 'correct but differently phrased'."""

    def __init__(self, groups: Sequence[Sequence[str]] | None = None,
                 case_sensitive: bool = False) -> None:
        self.groups = [list(g) for g in (groups or [])]
        self.case_sensitive = case_sensitive
        self.name = "contains"

    def score(self, *, input: str, expected: str | None, output: str, meta: dict) -> Score:
        haystack = output if self.case_sensitive else output.lower()
        # Per-case keyword groups win, so one dataset can carry mixed shapes.
        configured = meta.get("keywords")
        groups = self.groups or configured or ([[expected]] if expected else [])
        if not groups:
            return Score(0.0, False, "no keywords configured", self.name)
        matched = 0
        missing: list[str] = []
        for group in groups:
            if any((w if self.case_sensitive else w.lower()) in haystack for w in group if w):
                matched += 1
            else:
                missing.append(group[0])
        value = matched / len(groups)
        return Score(value, value >= 1.0,
                     f"missing: {', '.join(missing)}" if missing else "", self.name)


class Regex:
    def __init__(self, pattern: str, flags: int = 0) -> None:
        self.pattern = re.compile(pattern, flags)
        self.name = f"regex({pattern})"

    def score(self, *, input: str, expected: str | None, output: str, meta: dict) -> Score:
        hit = bool(self.pattern.search(output))
        return Score(1.0 if hit else 0.0, hit, "" if hit else "pattern not found", self.name)


class NumericTolerance:
    """Extract the first number from the output and compare within a tolerance.

    Numbers written as '1,234.5' or '￥1234' are normalised first; a tool that
    only parses bare floats fails on the outputs real models actually produce."""

    def __init__(self, tolerance: float = 0.0, relative: bool = False) -> None:
        self.tolerance = tolerance
        self.relative = relative
        self.name = f"numeric(tol={tolerance}{'rel' if relative else ''})"

    @staticmethod
    def _extract(text: str) -> float | None:
        match = re.search(r"-?\d[\d,]*\.?\d*", text.replace("，", ","))
        if not match:
            return None
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None

    def score(self, *, input: str, expected: str | None, output: str, meta: dict) -> Score:
        target = self._extract(expected or "")
        actual = self._extract(output)
        if target is None:
            return Score(0.0, False, "reference is not numeric", self.name)
        if actual is None:
            return Score(0.0, False, "no number found in output", self.name)
        allowed = abs(target) * self.tolerance if self.relative else self.tolerance
        diff = abs(actual - target)
        hit = diff <= allowed + 1e-12
        return Score(1.0 if hit else max(0.0, 1 - diff / (abs(target) or 1)), hit,
                     f"expected {target}, got {actual}", self.name)


class CallableScorer:
    """Escape hatch: wrap any function into a scorer."""

    def __init__(self, fn: Callable[[str, str | None, str], float], name: str = "custom") -> None:
        self.fn = fn
        self.name = name

    def score(self, *, input: str, expected: str | None, output: str, meta: dict) -> Score:
        value = float(self.fn(input, expected, output))
        return Score(value, value >= 0.5, "", self.name)


# ------------------------------------------------------------------ model judge

JUDGE_PROMPT = """You are grading an AI answer against a reference.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{output}

{rubric}

Score the candidate from 0.0 to 1.0:
- 1.0 = fully consistent with the reference
- 0.5 = partially correct, missing or vague on the key point
- 0.0 = contradicts the reference or fails to answer

Respond with JSON only, no prose:
{{"score": <number between 0 and 1>, "reason": "<one short sentence>"}}"""


class LLMJudge:
    """A model grading another model.

    Deliberately paranoid about its own output: judges drift toward verbosity and
    often wrap JSON in markdown fences or add a preamble, so parsing degrades
    through three strategies before giving up with a score of 0 rather than
    silently inflating results.
    """

    def __init__(
        self,
        lens,
        *,
        rubric: str = "",
        threshold: float = 0.7,
        model: str | None = None,
        name: str | None = None,
    ) -> None:
        self.lens = lens
        self.rubric = rubric
        self.threshold = threshold
        self.model = model
        self.name = name or f"llm_judge({model or lens.model})"

    def score(self, *, input: str, expected: str | None, output: str, meta: dict) -> Score:
        if expected is None:
            return Score(0.0, False, "judge needs a reference answer", self.name)

        prompt = JUDGE_PROMPT.format(
            question=input,
            reference=expected,
            output=output,
            rubric=self.rubric or "Focus on factual agreement, not wording.",
        )
        try:
            reply = self.lens.llm(
                [{"role": "user", "content": prompt}],
                model=self.model,
                name="judge",
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - a failed judge must not crash a run
            return Score(0.0, False, f"judge failed: {type(exc).__name__}: {exc}", self.name)

        value, reason = self._parse(reply.text)
        if value is None:
            return Score(0.0, False, f"unparseable judge output: {reply.text[:120]}", self.name)
        return Score(value, value >= self.threshold, reason, self.name)

    @staticmethod
    def _parse(text: str) -> tuple[float | None, str]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "score" in data:
                return float(data["score"]), str(data.get("reason", ""))[:200]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', cleaned)
        if match:
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned)
            return float(match.group(1)), (reason_match.group(1) if reason_match else "")[:200]

        match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", cleaned)
        if match:
            return float(match.group(1)), cleaned[:200]

        return None, ""
