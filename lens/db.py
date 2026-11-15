"""SQLite persistence layer.

Two-level model: a `run` is one end-to-end execution (one user question, one
eval case), and `spans` are the timed steps inside it. Keeping both in one file
means the whole trace travels with the project — no external service to stand up
before you can measure anything.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    project         TEXT NOT NULL,
    name            TEXT NOT NULL,
    input           TEXT,
    output          TEXT,
    status          TEXT NOT NULL DEFAULT 'ok',
    error           TEXT,
    started_at      REAL NOT NULL,
    ended_at        REAL,
    duration_ms     REAL,
    total_cost_usd  REAL NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    label           TEXT,
    origin          TEXT NOT NULL DEFAULT 'app',
    tags            TEXT,
    metadata        TEXT
);

CREATE TABLE IF NOT EXISTS spans (
    id                TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    parent_id         TEXT,
    name              TEXT NOT NULL,
    kind              TEXT NOT NULL DEFAULT 'custom',
    status            TEXT NOT NULL DEFAULT 'ok',
    error             TEXT,
    started_at        REAL NOT NULL,
    duration_ms       REAL,
    model             TEXT,
    provider          TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL NOT NULL DEFAULT 0,
    input             TEXT,
    output            TEXT,
    attributes        TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch       TEXT NOT NULL,
    label       TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    run_id      TEXT,
    output      TEXT,
    score       REAL NOT NULL DEFAULT 0,
    passed      INTEGER NOT NULL DEFAULT 0,
    scorer      TEXT,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_spans_run       ON spans(run_id, started_at);
CREATE INDEX IF NOT EXISTS idx_spans_kind      ON spans(kind, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_project    ON runs(project, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_label      ON runs(label, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_batch      ON eval_results(batch, label);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _dumps(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _loads(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def ensure_dir(path: Path) -> Path:
    """Create a directory idempotently.

    `exist_ok=True` is not trusted blindly: some sandboxed runtimes raise
    FileExistsError from their mkdir hook even when the directory is present.
    """
    if path.is_dir():
        return path
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    return path


def default_db_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    return ensure_dir(root / "data") / "lens.db"


class Store:
    """Thin synchronous wrapper. One connection per operation keeps threading
    simple; WAL mode lets the dashboard read while an agent is writing."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        ensure_dir(self.path.parent)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Additive migrations only, so an existing trace database keeps working."""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        if "origin" not in columns:
            conn.execute("ALTER TABLE runs ADD COLUMN origin TEXT NOT NULL DEFAULT 'app'")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------------------- writes

    def save_run(self, run: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (id, project, name, input, output, status, error, started_at,
                    ended_at, duration_ms, total_cost_usd, total_tokens, label,
                    origin, tags, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run["id"], run["project"], run["name"], _dumps(run.get("input")),
                    _dumps(run.get("output")), run.get("status", "ok"), run.get("error"),
                    run["started_at"], run.get("ended_at"), run.get("duration_ms"),
                    run.get("total_cost_usd", 0.0), run.get("total_tokens", 0),
                    run.get("label"), run.get("origin", "app"),
                    _dumps(run.get("tags")), _dumps(run.get("metadata")),
                ),
            )

    def save_spans_batch(self, spans: list[dict]) -> None:
        """Write many spans in a single transaction.

        One transaction per model call is fine for a demo and fatal for a service:
        at a few hundred requests per second the fsync rate becomes the bottleneck,
        and SQLite starts returning `database is locked` under contention.

        Run totals are recomputed once per affected run rather than once per span,
        so the cost of bookkeeping stays proportional to runs, not spans.
        """
        if not spans:
            return

        rows = [
            (
                span["id"], span["run_id"], span.get("parent_id"), span["name"],
                span.get("kind", "custom"), span.get("status", "ok"), span.get("error"),
                span["started_at"], span.get("duration_ms"), span.get("model"),
                span.get("provider"), span.get("prompt_tokens", 0),
                span.get("completion_tokens", 0), span.get("cost_usd", 0.0),
                _dumps(span.get("input")), _dumps(span.get("output")),
                _dumps(span.get("attributes")),
            )
            for span in spans
        ]

        with self.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO spans
                   (id, run_id, parent_id, name, kind, status, error, started_at,
                    duration_ms, model, provider, prompt_tokens, completion_tokens,
                    cost_usd, input, output, attributes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            for run_id in {span["run_id"] for span in spans}:
                conn.execute(
                    """UPDATE runs SET
                         total_cost_usd = (SELECT COALESCE(SUM(cost_usd), 0) FROM spans WHERE run_id = ?),
                         total_tokens   = (SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0)
                                           FROM spans WHERE run_id = ?)
                       WHERE id = ?""",
                    (run_id, run_id, run_id),
                )

    def save_span(self, span: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO spans
                   (id, run_id, parent_id, name, kind, status, error, started_at,
                    duration_ms, model, provider, prompt_tokens, completion_tokens,
                    cost_usd, input, output, attributes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    span["id"], span["run_id"], span.get("parent_id"), span["name"],
                    span.get("kind", "custom"), span.get("status", "ok"), span.get("error"),
                    span["started_at"], span.get("duration_ms"), span.get("model"),
                    span.get("provider"), span.get("prompt_tokens", 0),
                    span.get("completion_tokens", 0), span.get("cost_usd", 0.0),
                    _dumps(span.get("input")), _dumps(span.get("output")),
                    _dumps(span.get("attributes")),
                ),
            )
            conn.execute(
                """UPDATE runs SET
                     total_cost_usd = (SELECT COALESCE(SUM(cost_usd), 0) FROM spans WHERE run_id = ?),
                     total_tokens   = (SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0)
                                       FROM spans WHERE run_id = ?)
                   WHERE id = ?""",
                (span["run_id"], span["run_id"], span["run_id"]),
            )

    def save_eval_result(self, row: dict) -> None:
        """Idempotent on (batch, label, case_id, scorer).

        Re-running a batch must overwrite a case's score, not append a second row.
        A duplicated case silently doubles the sample size and narrows the
        confidence interval — exactly the kind of bug that makes a bad result look
        statistically solid.
        """
        with self.connect() as conn:
            conn.execute(
                """DELETE FROM eval_results
                   WHERE batch = ? AND label = ? AND case_id = ? AND scorer IS ?""",
                (row["batch"], row["label"], row["case_id"], row.get("scorer")),
            )
            conn.execute(
                """INSERT INTO eval_results
                   (batch, label, case_id, run_id, output, score, passed, scorer, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    row["batch"], row["label"], row["case_id"], row.get("run_id"),
                    _dumps(row.get("output")), row.get("score", 0.0),
                    1 if row.get("passed") else 0, row.get("scorer"), time.time(),
                ),
            )

    # ----------------------------------------------------------------- reads

    def list_runs(
        self,
        project: str | None = None,
        limit: int = 50,
        *,
        include_eval: bool = False,
    ) -> list[dict]:
        """`include_eval=False` keeps evaluation noise out of the default view.

        A 200-case eval run creates 200 runs. Without this filter the traces you
        actually want to inspect are buried within minutes of the first eval.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("project = ?")
            params.append(project)
        if not include_eval:
            clauses.append("origin = 'app'")
        query = "SELECT * FROM runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            return [self._run_row(r) for r in conn.execute(query, params)]

    def get_run(self, run_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            run = self._run_row(row)
            spans = conn.execute(
                "SELECT * FROM spans WHERE run_id = ? ORDER BY started_at", (run_id,)
            ).fetchall()
            run["spans"] = [self._span_row(s) for s in spans]
            return run

    def overview(self, project: str | None = None, hours: int = 24, *,
                 include_eval: bool = False) -> dict:
        since = time.time() - hours * 3600
        where = "WHERE r.started_at >= ?"
        params: list[Any] = [since]
        if project:
            where += " AND r.project = ?"
            params.append(project)
        if not include_eval:
            where += " AND r.origin = 'app'"
        with self.connect() as conn:
            agg = conn.execute(
                f"""SELECT COUNT(*) AS runs,
                           COALESCE(SUM(r.total_cost_usd), 0) AS cost,
                           COALESCE(SUM(r.total_tokens), 0) AS tokens,
                           COALESCE(AVG(r.duration_ms), 0) AS avg_ms,
                           SUM(CASE WHEN r.status = 'error' THEN 1 ELSE 0 END) AS errors
                    FROM runs r {where}""",
                params,
            ).fetchone()
            durs = [
                row["duration_ms"]
                for row in conn.execute(
                    f"SELECT r.duration_ms FROM runs r {where} AND r.duration_ms IS NOT NULL "
                    "ORDER BY r.duration_ms",
                    params,
                )
            ]
            by_model = conn.execute(
                f"""SELECT s.model,
                           COUNT(*) AS calls,
                           SUM(s.cost_usd) AS cost,
                           SUM(s.prompt_tokens + s.completion_tokens) AS tokens,
                           AVG(s.duration_ms) AS avg_ms
                    FROM spans s JOIN runs r ON r.id = s.run_id
                    {where} AND s.kind = 'llm' AND s.model IS NOT NULL
                    GROUP BY s.model ORDER BY cost DESC""",
                params,
            ).fetchall()

        return {
            "window_hours": hours,
            "runs": agg["runs"],
            "cost_usd": round(agg["cost"], 6),
            "tokens": agg["tokens"],
            "errors": agg["errors"] or 0,
            "avg_latency_ms": round(agg["avg_ms"] or 0, 1),
            "p50_latency_ms": round(_percentile(durs, 50), 1),
            "p95_latency_ms": round(_percentile(durs, 95), 1),
            "by_model": [dict(r) for r in by_model],
        }

    def eval_summary(self, batch: str | None = None, *, recent_batches: int = 5) -> list[dict]:
        """Aggregate scores per label.

        Defaults to the most recent few batches: every rerun creates a new batch,
        and a dashboard listing all of them becomes unreadable within a day.
        """
        query = """SELECT batch, label, COUNT(*) AS n, AVG(score) AS mean_score,
                          SUM(passed) * 1.0 / COUNT(*) AS pass_rate,
                          MIN(created_at) AS created_at
                   FROM eval_results"""
        params: list[Any] = []
        if batch:
            query += " WHERE batch = ?"
            params.append(batch)
        else:
            query += """ WHERE batch IN (
                            SELECT batch FROM eval_results
                            GROUP BY batch ORDER BY MAX(created_at) DESC LIMIT ?
                         )"""
            params.append(recent_batches)
        query += " GROUP BY batch, label ORDER BY created_at DESC, label"
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(query, params)]

    def eval_cases(self, batch: str) -> dict[str, list[dict]]:
        """Group per-case scores by label so the UI can show which cases moved."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT label, case_id, score, passed FROM eval_results
                   WHERE batch = ? ORDER BY case_id""",
                (batch,),
            ).fetchall()
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["label"], []).append(dict(row))
        return grouped

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["tags"] = _loads(data.get("tags"))
        data["metadata"] = _loads(data.get("metadata"))
        return data

    @staticmethod
    def _span_row(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["attributes"] = _loads(data.get("attributes"))
        return data


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)
