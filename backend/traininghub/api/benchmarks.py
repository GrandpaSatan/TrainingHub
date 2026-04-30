from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from traininghub.api.dependencies import current_user, settings_dependency
from traininghub.core.config import Settings
from traininghub.core.database import connect, rows_to_dicts
from traininghub.services.benchmark_catalog import BENCHMARKS_BY_ID, benchmark_catalog_payload


router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


@router.get("/catalog")
def catalog(
    _settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> list[dict[str, Any]]:
    return benchmark_catalog_payload()


@router.get("/results")
def results(
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
    model_slug: str | None = None,
    benchmark: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    if benchmark and benchmark not in BENCHMARKS_BY_ID:
        raise HTTPException(status_code=400, detail=f"Unsupported benchmark id: {benchmark}")

    clauses = []
    params: list[Any] = []
    if model_slug:
        clauses.append("br.model_slug = ?")
        params.append(model_slug)
    if benchmark:
        clauses.append("br.benchmark_name = ?")
        params.append(benchmark)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    with connect(settings.database_path) as conn:
        rows = conn.execute(
            f"""
            SELECT br.*, a.artifact_id
            FROM benchmark_results br
            LEFT JOIN artifacts a
              ON a.path = br.result_path
             AND a.artifact_type = 'benchmark_results'
            {where_sql}
            ORDER BY br.created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    records = rows_to_dicts(rows)
    for record in records:
        record["metrics"] = json.loads(record.pop("metrics_json") or "{}")
    return records
