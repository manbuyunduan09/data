from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass(frozen=True)
class Result:
    ok: bool
    status_code: int
    elapsed_ms: float
    error: Optional[str] = None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = max(0, min(len(vs) - 1, int(round((p / 100.0) * (len(vs) - 1)))) )
    return vs[k]


async def run_once(client: httpx.AsyncClient, method: str, url: str, json_body: dict | None) -> Result:
    start = time.perf_counter()
    try:
        res = await client.request(method, url, json=json_body)
        elapsed_ms = (time.perf_counter() - start) * 1000
        ok = 200 <= res.status_code < 300
        return Result(ok=ok, status_code=res.status_code, elapsed_ms=elapsed_ms)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return Result(ok=False, status_code=0, elapsed_ms=elapsed_ms, error=str(e))


async def run_concurrent(
    *,
    base_url: str,
    path: str,
    method: str,
    json_body: dict | None,
    total: int,
    concurrency: int,
    timeout_s: float,
) -> list[Result]:
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s, follow_redirects=True) as client:
        async def task() -> Result:
            async with sem:
                return await run_once(client, method, path, json_body)

        return await asyncio.gather(*[task() for _ in range(total)])


def summarize(name: str, results: list[Result]) -> None:
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    lat = [r.elapsed_ms for r in results]
    lat_ok = [r.elapsed_ms for r in ok]

    print(f"\n== {name} ==")
    print(f"total={len(results)} ok={len(ok)} err={len(bad)} err_rate={len(bad)/max(1,len(results)):.2%}")
    print(f"p50={percentile(lat,50):.1f}ms p90={percentile(lat,90):.1f}ms p99={percentile(lat,99):.1f}ms")
    if lat_ok:
        print(f"ok_avg={statistics.mean(lat_ok):.1f}ms ok_max={max(lat_ok):.1f}ms")
    if bad:
        codes = {}
        for r in bad:
            codes[str(r.status_code)] = codes.get(str(r.status_code), 0) + 1
        print(f"errors_by_status={codes}")
        sample = bad[0]
        if sample.error:
            print(f"sample_error={sample.error}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--total", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--timeout", type=float, default=10.0)
    args = p.parse_args()

    base_url = args.base_url.rstrip("/")

    results_health = asyncio.run(
        run_concurrent(
            base_url=base_url,
            path="/",
            method="GET",
            json_body=None,
            total=args.total,
            concurrency=args.concurrency,
            timeout_s=args.timeout,
        )
    )
    summarize("GET /", results_health)

    payload = {
        "exportFormat": "html",
        "userEmail": "bench@example.com",
        "productName": "bench",
        "languageDefault": "zh",
        "cards": [{"name": "PV", "total": 1, "average": 1}],
        "charts": [],
    }
    results_export = asyncio.run(
        run_concurrent(
            base_url=base_url,
            path="/api/v1/export",
            method="POST",
            json_body=payload,
            total=args.total,
            concurrency=max(1, min(args.concurrency, 20)),
            timeout_s=args.timeout,
        )
    )
    summarize("POST /api/v1/export", results_export)


if __name__ == "__main__":
    main()

