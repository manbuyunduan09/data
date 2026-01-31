from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Literal, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from export_core import (
    ExportEstimate,
    ExportFormat,
    TASK_MANAGER,
    estimate_export,
    export_dashboard_html,
    export_dashboard_xlsx,
)


def _utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class CardPayload(BaseModel):
    name: str
    total: Optional[float] = None
    average: Optional[float] = None


class ChartPayload(BaseModel):
    title: str
    figure_json: str
    table_records: list[dict[str, Any]] = Field(default_factory=list)


class ExportRequest(BaseModel):
    exportFormat: Literal["html", "xlsx"]
    userEmail: str
    productName: str
    languageDefault: Literal["zh", "en"] = "zh"
    cards: list[CardPayload] = Field(default_factory=list)
    charts: list[ChartPayload] = Field(default_factory=list)


class ExportResponse(BaseModel):
    taskId: str
    estimateSeconds: float
    sizeBytes: int
    traceId: str


class StatusResponse(BaseModel):
    taskId: str
    status: str
    progress: int
    estimateSeconds: Optional[float] = None
    sizeBytes: Optional[int] = None
    sha256: Optional[str] = None
    traceId: Optional[str] = None
    errorMessage: Optional[str] = None
    createdAtUtc: str
    finishedAtUtc: Optional[str] = None


app = FastAPI(title="Export API", version="1.0")


@app.post("/api/v1/export", response_model=ExportResponse)
def create_export_task(payload: ExportRequest = Body(...)) -> ExportResponse:
    trace_id = str(uuid.uuid4())
    try:
        fmt: ExportFormat = payload.exportFormat
        estimate = estimate_export(len(payload.cards), len(payload.charts), fmt)

        def runner(set_progress, is_cancelled):
            set_progress(5)
            if is_cancelled():
                raise RuntimeError("cancelled")

            cards = [
                {
                    "name": c.name,
                    "total": c.total,
                    "average": c.average,
                }
                for c in payload.cards
            ]
            set_progress(15)

            charts = []
            if payload.charts:
                import pandas as pd
                import plotly.io as pio

                for c in payload.charts:
                    if is_cancelled():
                        raise RuntimeError("cancelled")
                    fig = pio.from_json(c.figure_json)
                    table = pd.DataFrame.from_records(c.table_records)
                    charts.append({"title": c.title, "figure": fig, "table": table})

            set_progress(55)

            if fmt == "html":
                from export_core import CardSummary, ChartBundle

                card_objs = [CardSummary(name=c["name"], total=c["total"], average=c["average"]) for c in cards]
                chart_objs = [
                    ChartBundle(title=cc["title"], figure=cc["figure"], table=cc["table"]) for cc in charts
                ]
                set_progress(75)
                return export_dashboard_html(
                    product_name=payload.productName,
                    user_email=payload.userEmail,
                    language_default=payload.languageDefault,
                    cards=card_objs,
                    charts=chart_objs,
                )

            from export_core import CardSummary, ChartBundle

            card_objs = [CardSummary(name=c["name"], total=c["total"], average=c["average"]) for c in cards]
            chart_objs = [ChartBundle(title=cc["title"], figure=cc["figure"], table=cc["table"]) for cc in charts]
            set_progress(75)
            return export_dashboard_xlsx(
                product_name=payload.productName,
                user_email=payload.userEmail,
                cards=card_objs,
                charts=chart_objs,
            )

        task = TASK_MANAGER.submit(
            user_email=payload.userEmail,
            export_format=payload.exportFormat,
            estimate=estimate,
            runner=runner,
        )

        return ExportResponse(
            taskId=task.task_id,
            estimateSeconds=estimate.estimate_seconds,
            sizeBytes=estimate.size_bytes,
            traceId=trace_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"message": str(e), "traceId": trace_id, "timeUtc": _utc_iso()},
        )


@app.get("/api/v1/export/status", response_model=StatusResponse)
def export_status(taskId: str = Query(...), userEmail: str = Query(...)) -> StatusResponse:
    task = TASK_MANAGER.get(taskId)
    if not task or task.user_email != userEmail:
        raise HTTPException(status_code=404, detail={"message": "task not found", "traceId": str(uuid.uuid4())})

    return StatusResponse(
        taskId=task.task_id,
        status=task.status,
        progress=task.progress,
        estimateSeconds=task.estimate_seconds,
        sizeBytes=task.size_bytes,
        sha256=task.sha256,
        traceId=task.trace_id,
        errorMessage=task.error_message,
        createdAtUtc=task.created_at_utc,
        finishedAtUtc=task.finished_at_utc,
    )


@app.post("/api/v1/export/cancel", response_model=dict)
def cancel_export(taskId: str = Body(...), userEmail: str = Body(...)) -> dict:
    ok = TASK_MANAGER.cancel(taskId, user_email=userEmail)
    if not ok:
        raise HTTPException(status_code=404, detail={"message": "task not found", "traceId": str(uuid.uuid4())})
    return {"taskId": taskId, "cancelled": True}
