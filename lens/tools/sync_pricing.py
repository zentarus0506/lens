"""Refresh the price table from models.dev.

Kept as a tool rather than a runtime fetch: a dashboard that hits the network on
every page load is slower and fails in ways you cannot debug. Refresh on demand,
write the snapshot, keep the fetched date so a weird bill can be traced back.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SOURCE = "https://models.dev/api.json"
TARGETS = {
    "zenmux": "zenmux",
    "siliconflow-cn": "siliconflow",
    "deepseek": "deepseek",
    "moonshotai": "moonshot",
    "openai": "openai",
}
OUT = Path(__file__).resolve().parent.parent / "pricing.json"


def fetch(url: str, attempts: int = 3, timeout: int = 30) -> dict:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "lens/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            print(f"  第 {attempt} 次失败: {type(exc).__name__}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise RuntimeError(f"无法获取 {url}: {last}")


def main() -> int:
    print(f"从 {SOURCE} 拉取价格...")
    data = fetch(SOURCE)

    providers: dict[str, dict] = {}
    for source_id, our_name in TARGETS.items():
        models = (data.get(source_id) or {}).get("models") or {}
        priced = {}
        for model_id, meta in models.items():
            cost = meta.get("cost") or {}
            if cost.get("input") is None and cost.get("output") is None:
                continue
            priced[model_id] = {
                "input": cost.get("input") or 0,
                "output": cost.get("output") or 0,
            }
        providers[our_name] = priced
        print(f"  {our_name:14} {len(priced):3d} 个模型有定价")

    payload = {
        "source": "models.dev",
        "fetched_at": time.strftime("%Y-%m-%d"),
        "unit": "usd_per_1m_tokens",
        "providers": providers,
    }

    # Write to a temp file first: a half-written price table silently zeroes costs.
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(OUT)
    print(f"\n已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
