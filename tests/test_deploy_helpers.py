"""Pure-logic tests for the Phase 5.5b deploy Lambdas (canary + cost guard).

These load the Lambda handler modules by path (they live under infra/lambdas and
are not part of the installed package) and exercise only the pure helpers - no
AWS, no network. boto3 is imported lazily inside the handlers, so importing the
modules here needs nothing beyond the stdlib.
"""
from __future__ import annotations

import base64
import importlib.util
from pathlib import Path

INFRA = Path(__file__).resolve().parents[1] / "infra"


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _canary():
    return _load("canary_handler", INFRA / "lambdas" / "canary" / "handler.py")


def _cost_guard():
    return _load("cost_guard_handler", INFRA / "lambdas" / "cost_guard" / "handler.py")


def test_build_multipart_body_structure():
    mod = _canary()
    payload = b"\xff\xd8\xff\xe0hello"
    body, content_type = mod.build_multipart_body(
        "file", "canary.jpg", "image/jpeg", payload
    )

    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=", 1)[1]

    assert body.startswith(b"--" + boundary.encode())
    assert body.rstrip(b"\r\n").endswith(b"--" + boundary.encode() + b"--")
    assert b'Content-Disposition: form-data; name="file"; filename="canary.jpg"' in body
    assert b"Content-Type: image/jpeg" in body
    assert payload in body
    # Headers are separated from the body by a blank line (CRLF CRLF).
    assert b"\r\n\r\n" in body


def test_canary_jpeg_is_valid():
    mod = _canary()
    raw = base64.b64decode(mod.CANARY_JPEG_B64)
    assert raw[:3] == b"\xff\xd8\xff"  # JPEG SOI marker
    assert raw[-2:] == b"\xff\xd9"  # JPEG EOI marker
    assert len(raw) > 500


def test_max_daily_cost_picks_costliest_day():
    mod = _cost_guard()
    results = [
        {"TimePeriod": {"Start": "2026-07-07"}, "Total": {"UnblendedCost": {"Amount": "0.30", "Unit": "USD"}}},
        {"TimePeriod": {"Start": "2026-07-08"}, "Total": {"UnblendedCost": {"Amount": "2.75", "Unit": "USD"}}},
        {"TimePeriod": {"Start": "2026-07-09"}, "Total": {"UnblendedCost": {"Amount": "0.10", "Unit": "USD"}}},
    ]
    worst_date, worst_amount = mod.max_daily_cost(results)
    assert worst_date == "2026-07-08"
    assert worst_amount == 2.75


def test_max_daily_cost_empty_is_zero():
    mod = _cost_guard()
    worst_date, worst_amount = mod.max_daily_cost([])
    assert worst_date is None
    assert worst_amount == 0.0
