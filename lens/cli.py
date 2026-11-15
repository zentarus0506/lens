"""Small operator CLI.

Only commands that answer a question you actually have while debugging:
"which provider is live", "is my price table stale", "what did the last run do".
"""

from __future__ import annotations

import argparse
import sys

from . import pricing
from .config import KNOWN_PROVIDERS, default_model, provider_status
from .db import Store


def cmd_providers(_: argparse.Namespace) -> int:
    rows = provider_status()
    width = max(len(r["provider"]) for r in rows)
    print(f"{'provider'.ljust(width)}  ready  model                        source")
    print("-" * (width + 52))
    for row in rows:
        ready = "yes" if row["ready"] else "no "
        model = default_model(row["provider"], respect_env=False)[:28]
        print(f"{row['provider'].ljust(width)}  {ready}    {model.ljust(28)} {row['source']}")
    print()
    print("凭证解析顺序: 环境变量 -> .env -> 复用本机 opencode/auth.json")
    print("切换: LENS_PROVIDER=<name> LENS_MODEL=<name>")
    return 0


def cmd_pricing(_: argparse.Namespace) -> int:
    info = pricing.snapshot_info()
    print(f"来源 {info['source']} · 快照 {info['snapshot']} · 距今 {info['age_hours']} 小时")
    if info["stale"]:
        print("警告: 价格表超过 30 天，成本数字可能已经不准")
    print()
    for name, count in sorted(info["models"].items()):
        print(f"  {name:14} {count:3d} 个模型有定价")
    print()
    print("刷新: python -m lens.tools.sync_pricing")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    store = Store()
    runs = store.list_runs(project=args.project, limit=args.limit)
    if not runs:
        print("没有记录。先运行 python examples/demo_agent.py")
        return 0
    print(f"{'name':22} {'status':7} {'ms':>9} {'tokens':>8} {'cost':>11}  started")
    for run in runs:
        started = run["started_at"]
        import time as _t
        stamp = _t.strftime("%m-%d %H:%M:%S", _t.localtime(started))
        ms = f"{run['duration_ms']:.0f}" if run["duration_ms"] is not None else "-"
        print(f"{run['name'][:22]:22} {run['status']:7} {ms:>9} {run['total_tokens']:>8} "
              f"{'$' + format(run['total_cost_usd'], '.6f'):>11}  {stamp}")
    print(f"\n合计 {len(runs)} 条 · 数据库 {store.path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lens", description="Lens operator CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("providers", help="查看 provider 就绪状态").set_defaults(func=cmd_providers)
    sub.add_parser("pricing", help="查看价格表快照").set_defaults(func=cmd_pricing)

    runs = sub.add_parser("runs", help="列出最近的执行记录")
    runs.add_argument("--project", default=None)
    runs.add_argument("--limit", type=int, default=15)
    runs.set_defaults(func=cmd_runs)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
