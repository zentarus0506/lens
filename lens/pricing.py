"""Model pricing, snapshotted from models.dev.

Cost accounting is only useful if it is honest. When a model cannot be matched
against the table we return `None` rather than 0.0 — an unpriced call should look
unpriced, not free. Stale numbers get a snapshot date so a surprising bill can be
traced back to a stale table instead of blamed on the code.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

_COMMENT = "#"


@dataclass(frozen=True)
class Price:
    input_per_1m: float
    output_per_1m: float
    matched: str
    snapshot: str


def _pricing_file() -> Path:
    return Path(os.environ.get("LENS_PRICING", Path(__file__).with_name("pricing.json")))


_cache: dict | None = None
_cache_mtime: float = 0.0


def _load_table() -> dict:
    global _cache, _cache_mtime
    path = _pricing_file()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"providers": {}, "fetched_at": "unknown"}
    if _cache is not None and mtime == _cache_mtime:
        return _cache
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"providers": {}, "fetched_at": "unknown"}
    _cache, _cache_mtime = data, mtime
    return data


def _normalize(model: str) -> str:
    name = model.strip().lower()
    name = name.rsplit("/", 1)[-1]          # drop vendor prefix
    name = name.split(":")[0]              # drop ollama tag
    for suffix in ("-latest", "-preview", "-exp"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace(".", "_")


def _search(models: dict, model: str) -> tuple[int, str, dict] | None:
    """Score every candidate id against the requested model name."""
    target = _normalize(model)
    best: tuple[int, str, dict] | None = None
    for model_id, price in models.items():
        if model_id == model:
            score = 100
        elif model_id.endswith("/" + model) or model.endswith("/" + model_id):
            score = 85
        else:
            normalized = _normalize(model_id)
            if normalized == target:
                score = 80
            elif len(target) >= 5 and (target in normalized or normalized in target):
                score = 55
            else:
                continue
        if best is None or score > best[0]:
            best = (score, model_id, price)
    return best


def lookup(provider: str, model: str) -> Price | None:
    """Resolve a price, preferring the provider actually used.

    Falls back to a global search because the same model is resold by several
    gateways under the same name, and a missing price in one table should not
    silently zero out a cost you actually paid. A cross-provider match is a
    estimate, not a fact, so the matched id is reported alongside.
    """
    table = _load_table()
    snapshot = table.get("fetched_at", "unknown")
    providers = table.get("providers") or {}

    hit = _search(providers.get(provider) or {}, model)
    if hit is None:
        for other, models in providers.items():
            if other == provider:
                continue
            candidate = _search(models, model)
            if candidate and (hit is None or candidate[0] > hit[0]):
                hit = candidate

    if hit is None:
        return None
    _, model_id, price = hit
    return Price(
        input_per_1m=float(price.get("input") or 0.0),
        output_per_1m=float(price.get("output") or 0.0),
        matched=model_id,
        snapshot=snapshot,
    )


def cost_usd(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = lookup(provider, model)
    if price is None:
        return None
    return (
        prompt_tokens / 1_000_000 * price.input_per_1m
        + completion_tokens / 1_000_000 * price.output_per_1m
    )


def snapshot_info() -> dict:
    table = _load_table()
    providers = table.get("providers") or {}
    path = _pricing_file()
    try:
        age_hours = round((time.time() - path.stat().st_mtime) / 3600, 1)
    except OSError:
        age_hours = None
    return {
        "source": table.get("source", "unknown"),
        "snapshot": table.get("fetched_at", "unknown"),
        "age_hours": age_hours,
        "models": {name: len(models) for name, models in providers.items()},
        "stale": bool(age_hours is None or age_hours > 24 * 30),
    }
