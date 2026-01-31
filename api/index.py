from __future__ import annotations

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from fastapi import FastAPI

from export_api import app as export_app


app: FastAPI = export_app


@app.get("/")
def health() -> dict:
    return {
        "ok": True,
        "service": "data-dashboard-export-api",
        "paths": ["/api/v1/export", "/api/v1/export/status", "/api/v1/export/cancel"],
    }

