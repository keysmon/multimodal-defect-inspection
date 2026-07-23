"""Thin HTTP client for the SiteCheck API (MCP server backend).

Holds NO models and NO AWS credentials - it is an honest client of the same
HTTP API the browser uses. Async jobs (202 + poll) are resolved INSIDE each
call so MCP clients get one blocking tool call with the final result.

Default target is a local API (see README "Run locally"); point
SITECHECK_API_URL at any deployed instance instead.
"""
from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path

import httpx

DEFAULT_API_URL = "http://localhost:8000"


class SiteCheckClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float = 300.0,
        poll_interval_s: float = 3.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("SITECHECK_API_URL") or DEFAULT_API_URL
        ).rstrip("/")
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self._http = httpx.Client(
            base_url=self.base_url, timeout=30.0, transport=transport
        )

    def _file_tuple(self, path: str) -> tuple[str, bytes, str]:
        p = Path(path)
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return (p.name, p.read_bytes(), mime)

    def _poll(self, url: str) -> dict:
        deadline = time.monotonic() + self.timeout_s
        while True:
            resp = self._http.get(url)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code != 202:
                detail = ""
                if resp.content:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except ValueError:
                        detail = resp.text
                raise RuntimeError(f"job failed ({resp.status_code}): {detail}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"job at {url} still pending after {self.timeout_s}s")
            time.sleep(self.poll_interval_s)

    def analyze_photo(self, path: str, note: str = "") -> dict:
        """Submit one photo for defect analysis and block until the job resolves.

        Raises:
            httpx.HTTPStatusError: the submit request failed.
            RuntimeError: the job failed server-side.
            TimeoutError: the job did not resolve within timeout_s.
        """
        resp = self._http.post(
            "/analyze-jobs",
            files={"file": self._file_tuple(path)},
            data={"note": note},
        )
        resp.raise_for_status()
        return self._poll(f"/analyze-jobs/{resp.json()['job_id']}")

    def search_standards(self, query: str) -> dict:
        """Search the standards knowledge base for the given query. Synchronous - no poll.

        Raises:
            httpx.HTTPStatusError: the search request failed.
        """
        resp = self._http.post("/search", json={"query": query})
        resp.raise_for_status()
        return resp.json()

    def run_walkthrough(
        self,
        photo_paths: list[str],
        visit_note: str = "",
        photo_notes: list[str] | None = None,
    ) -> dict:
        """Submit a full walkthrough (multiple photos + notes) and block until the
        job resolves.

        Raises:
            httpx.HTTPStatusError: the submit request failed.
            RuntimeError: the job failed server-side.
            TimeoutError: the job did not resolve within timeout_s.
        """
        notes = photo_notes or [""] * len(photo_paths)
        files = [("files", self._file_tuple(p)) for p in photo_paths]
        # httpx encodes repeated form fields from a dict-of-list value, not a
        # list of (key, value) tuples - the latter raises when combined with
        # multipart `files` (httpx 0.28: "expected a bytes-like object, tuple
        # found"). This still produces one "photo_notes" part per photo.
        data = {"visit_note": visit_note, "photo_notes": notes}
        resp = self._http.post("/walkthrough-jobs", files=files, data=data)
        resp.raise_for_status()
        return self._poll(f"/walkthrough-jobs/{resp.json()['job_id']}")
