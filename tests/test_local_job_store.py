"""LocalCpuJobStore: the in-process async path for local dev (no AWS).

The walkthrough has NO sync route, so local development needs an async store:
submit runs the worker in a daemon thread against in-memory storage; status()
polls it exactly like the S3-backed store.
"""
import json
import time

from defectlens.serve.async_jobs import (
    LocalCpuJobStore,
    build_cpu_job_store_from_env,
)


def _wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_local_store_runs_worker_thread_and_serves_result():
    calls = {}

    def fake_worker(app, event, store):
        job_id = event["defectlens_job"]["job_id"]
        calls["input"] = store.get_input(job_id)
        store.put_output(job_id, {"ok": True})

    store = LocalCpuJobStore(worker=fake_worker)
    store.bind(app="the-app")
    job_id = store.submit_payload(json.dumps({"kind": "walkthrough"}).encode())["job_id"]

    assert _wait_until(lambda: store.status(job_id)[0] == "ready")
    state, result = store.status(job_id)
    assert state == "ready" and result == {"ok": True}
    assert calls["input"] == b'{"kind": "walkthrough"}'


def test_local_store_pending_then_error_path():
    def boom_worker(app, event, store):
        store.put_error(event["defectlens_job"]["job_id"], {"error": "boom"})

    store = LocalCpuJobStore(worker=boom_worker)
    store.bind(app=None)
    job_id = store.submit_payload(b"{}")["job_id"]
    assert _wait_until(lambda: store.status(job_id)[0] == "failed")
    assert store.status(job_id)[1] == {"error": "boom"}


def test_local_store_status_pending_for_unknown_job():
    store = LocalCpuJobStore(worker=lambda *a: None)
    assert store.status("nope") == ("pending", None)


def test_local_store_enabled_and_env_selection(monkeypatch):
    store = LocalCpuJobStore(worker=lambda *a: None)
    assert store.enabled is True

    monkeypatch.setenv("DEFECTLENS_LOCAL_JOBS", "1")
    monkeypatch.delenv("CPU_JOBS_S3", raising=False)
    built = build_cpu_job_store_from_env()
    assert isinstance(built, LocalCpuJobStore)

    monkeypatch.delenv("DEFECTLENS_LOCAL_JOBS", raising=False)
    assert build_cpu_job_store_from_env() is None
