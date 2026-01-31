from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from fastapi import Body, FastAPI, HTTPException, Query, Request


app = FastAPI(title="Export API (Vercel)", version="1.0")


def _backend_base_url() -> Optional[str]:
    url = os.environ.get("EXPORT_BACKEND_BASE_URL")
    if not url:
        return None
    return url.rstrip("/")


async def _proxy(
    *, request: Request, path: str, json_body: Any | None = None, query: dict[str, Any] | None = None
) -> Any:
    base_url = _backend_base_url()
    if not base_url:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Vercel 轻量部署不包含导出运行时，请配置 EXPORT_BACKEND_BASE_URL 以转发到导出后端。",
                "example": "https://your-export-backend.example.com",
                "path": path,
            },
        )

    headers: dict[str, str] = {}
    for k, v in request.headers.items():
        lk = k.lower()
        if lk in {"host", "content-length"}:
            continue
        headers[k] = v

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0, follow_redirects=True) as client:
        res = await client.request(request.method, path, headers=headers, params=query, json=json_body)
        content_type = res.headers.get("content-type", "")
        if "application/json" in content_type:
            return res.status_code, res.json()
        return res.status_code, {"raw": res.text}


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "data-dashboard-export-api-vercel",
        "mode": "proxy",
        "exportBackendConfigured": bool(_backend_base_url()),
        "paths": ["/api/v1/export", "/api/v1/export/status", "/api/v1/export/cancel"],
    }


@app.post("/api/v1/export")
async def create_export_task(request: Request, payload: dict[str, Any] = Body(...)) -> Any:
    status_code, body = await _proxy(request=request, path="/api/v1/export", json_body=payload)
    if status_code >= 400:
        raise HTTPException(status_code=status_code, detail=body)
    return body


@app.get("/api/v1/export/status")
async def export_status(request: Request, taskId: str = Query(...), userEmail: str = Query(...)) -> Any:
    status_code, body = await _proxy(
        request=request,
        path="/api/v1/export/status",
        query={"taskId": taskId, "userEmail": userEmail},
    )
    if status_code >= 400:
        raise HTTPException(status_code=status_code, detail=body)
    return body


@app.post("/api/v1/export/cancel")
async def cancel_export(request: Request, payload: dict[str, Any] = Body(...)) -> Any:
    status_code, body = await _proxy(request=request, path="/api/v1/export/cancel", json_body=payload)
    if status_code >= 400:
        raise HTTPException(status_code=status_code, detail=body)
    return body

