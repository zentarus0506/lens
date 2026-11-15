"""Read-only HTTP surface for the dashboard.

Deliberately read-only: an observability tool that can mutate the data it reports
on is much harder to trust. Writes happen through the SDK, reads happen here.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from .. import pricing
from ..config import provider_status
from ..db import Store
from ..eval.runner import compare_labels
from ..eval.stats import bootstrap_ci

STATIC_DIR = Path(__file__).with_name("static")


def create_app(store: Store | None = None) -> FastAPI:
    app = FastAPI(title="Lens", version="0.1.0", docs_url="/api/docs")
    db = store or Store()

    @app.get("/api/overview")
    def overview(
        project: str | None = None,
        hours: int = Query(24, ge=1, le=24 * 365),
        include_eval: bool = False,
    ) -> dict:
        return db.overview(project=project, hours=hours, include_eval=include_eval)

    @app.get("/api/runs")
    def runs(
        project: str | None = None,
        limit: int = Query(50, ge=1, le=500),
        include_eval: bool = False,
    ) -> dict:
        return {
            "runs": db.list_runs(project=project, limit=limit, include_eval=include_eval)
        }

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        run = db.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    @app.get("/api/eval/summary")
    def eval_summary(batch: str | None = None) -> dict:
        rows = db.eval_summary(batch)
        # Attach a bootstrap interval per (batch, label). A mean without an
        # interval is how people talk themselves into shipping noise.
        cases_by_batch: dict[str, dict[str, list[dict]]] = {}
        for row in rows:
            batch_id = row["batch"]
            if batch_id not in cases_by_batch:
                cases_by_batch[batch_id] = db.eval_cases(batch_id)
            scored = cases_by_batch[batch_id].get(row["label"], [])
            interval = bootstrap_ci([c["score"] for c in scored]) if scored else None
            row["ci_low"] = round(interval.low, 4) if interval else None
            row["ci_high"] = round(interval.high, 4) if interval else None
        return {"batches": rows}

    @app.get("/api/eval/compare")
    def eval_compare(batch: str, a: str, b: str) -> dict:
        return compare_labels(db, batch, a, b)

    @app.get("/api/eval/{batch}")
    def eval_cases(batch: str) -> dict:
        return {"batch": batch, "by_label": db.eval_cases(batch)}

    @app.get("/api/providers")
    def providers() -> dict:
        return {"providers": provider_status(), "pricing": pricing.snapshot_info()}

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "db": str(db.path), "exists": Path(db.path).exists()}

    @app.get("/")
    def index() -> FileResponse:
        target = STATIC_DIR / "index.html"
        if not target.is_file():
            return JSONResponse(  # type: ignore[return-value]
                status_code=500,
                content={"error": "dashboard assets missing", "expected": str(target)},
            )
        return FileResponse(target)

    return app


def main() -> None:
    import uvicorn

    # Hosted platforms inject PORT and expect a 0.0.0.0 bind. Locally we stay on
    # loopback unless explicitly told otherwise, so a stray deploy never exposes
    # a developer's trace database to the network.
    hosted = bool(os.environ.get("PORT"))
    port = int(os.environ.get("PORT") or os.environ.get("LENS_PORT", "8765"))
    host = os.environ.get("LENS_HOST") or ("0.0.0.0" if hosted else "127.0.0.1")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
