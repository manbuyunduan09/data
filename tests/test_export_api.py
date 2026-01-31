import time

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi.testclient import TestClient

import export_api


def test_export_api_lifecycle_html():
    client = TestClient(export_api.app)
    payload = {
        "exportFormat": "html",
        "userEmail": "a@example.com",
        "productName": "Prod",
        "languageDefault": "zh",
        "cards": [{"name": "PV", "total": 1.0, "average": 1.0}],
        "charts": [],
    }
    r = client.post("/api/v1/export", json=payload)
    assert r.status_code == 200
    body = r.json()
    task_id = body["taskId"]

    for _ in range(50):
        s = client.get("/api/v1/export/status", params={"taskId": task_id, "userEmail": "a@example.com"})
        assert s.status_code == 200
        status = s.json()["status"]
        if status in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.05)

    assert status == "succeeded"


def test_export_api_lifecycle_xlsx_with_chart():
    client = TestClient(export_api.app)
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
    fig = go.Figure(data=[go.Scatter(x=df["x"], y=df["y"])])
    fig_json = pio.to_json(fig)

    payload = {
        "exportFormat": "xlsx",
        "userEmail": "b@example.com",
        "productName": "Prod",
        "languageDefault": "en",
        "cards": [{"name": "UV", "total": 2.0, "average": 1.0}],
        "charts": [
            {
                "title": "Chart",
                "figure_json": fig_json,
                "table_records": df.to_dict(orient="records"),
            }
        ],
    }
    r = client.post("/api/v1/export", json=payload)
    assert r.status_code == 200
    task_id = r.json()["taskId"]

    for _ in range(300):
        s = client.get("/api/v1/export/status", params={"taskId": task_id, "userEmail": "b@example.com"})
        assert s.status_code == 200
        status = s.json()["status"]
        if status in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.1)

    assert status == "succeeded"


def test_export_status_not_found_returns_404():
    client = TestClient(export_api.app)
    r = client.get("/api/v1/export/status", params={"taskId": "nope", "userEmail": "a@example.com"})
    assert r.status_code == 404
    body = r.json()
    assert "detail" in body


def test_cancel_endpoint_can_cancel_running_task():
    client = TestClient(export_api.app)

    df = pd.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
    fig = go.Figure(data=[go.Scatter(x=df["x"], y=df["y"])])
    fig_json = pio.to_json(fig)

    charts = [
        {"title": f"C{i}", "figure_json": fig_json, "table_records": df.to_dict(orient="records")}
        for i in range(80)
    ]

    payload = {
        "exportFormat": "xlsx",
        "userEmail": "c@example.com",
        "productName": "Prod",
        "languageDefault": "zh",
        "cards": [],
        "charts": charts,
    }

    r = client.post("/api/v1/export", json=payload)
    assert r.status_code == 200
    task_id = r.json()["taskId"]

    c = client.post("/api/v1/export/cancel", json={"taskId": task_id, "userEmail": "c@example.com"})
    assert c.status_code == 200
    assert c.json()["cancelled"] is True

    final = None
    for _ in range(200):
        s = client.get("/api/v1/export/status", params={"taskId": task_id, "userEmail": "c@example.com"})
        assert s.status_code == 200
        final = s.json()["status"]
        if final in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.05)

    assert final in {"cancelled", "failed"}
