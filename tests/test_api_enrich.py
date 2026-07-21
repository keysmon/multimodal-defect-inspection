"""Walkthrough GPU-enrichment routes: fan-out, poll, gate merge, honesty."""
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from defectlens.serve.api import create_app
from defectlens.serve.async_jobs import build_walkthrough_job_payload


def png() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    return buf.getvalue()


def _report(observation="patch of missing concrete, spalled edges"):
    return {
        "concerns": [],
        "per_photo": [
            {"photo_id": "photo_1", "observation": observation,
             "cited": ["hud-008"], "no_evidence": False, "enrichment": None},
            {"photo_id": "photo_2", "observation": "a furnace with a flue",
             "cited": ["hud-008"], "no_evidence": False, "enrichment": None},
        ],
        "summary": {"overall_assessment": "x", "assessment_citations": ["hud-008"],
                    "action_items": [], "answers": []},
        "disclaimer": "Initial diagnostic - verify before acting.",
        "flagged_claims": [],
        "cards": {},
    }


class MemoryStore:
    enabled = True

    def __init__(self, report=None):
        photos = [
            {"photo_id": "photo_1", "image_bytes": png(), "note": "chunks below"},
            {"photo_id": "photo_2", "image_bytes": png(), "note": None},
        ]
        self.inputs = {"wt-1": build_walkthrough_job_payload(photos, "note")}
        self.outputs = {} if report is None else {"wt-1": report}
        self.enrichments = {}

    def bind(self, app):
        pass

    def status(self, job_id):
        if job_id in self.outputs:
            return "ready", self.outputs[job_id]
        return "pending", None

    def get_input(self, job_id):
        return self.inputs[job_id]

    def put_output(self, job_id, obj):
        self.outputs[job_id] = obj

    def put_enrichment(self, job_id, obj):
        self.enrichments[job_id] = obj

    def get_enrichment(self, job_id):
        return self.enrichments.get(job_id)


class FakeGateway:
    enabled = True

    def __init__(self, statuses=None):
        self.submitted = []
        # keyed by output_location; default: everything pending
        self.statuses = statuses or {}

    def submit(self, image_bytes, note):
        n = len(self.submitted) + 1
        self.submitted.append({"image_bytes": image_bytes, "note": note})
        return {
            "job_id": f"sm-{n}",
            "output_location": f"s3://out/{n}.json",
            "failure_location": f"s3://fail/{n}.json",
        }

    def status(self, output_location, failure_location=None):
        return self.statuses.get(output_location, ("pending", None))


def _client(store, gateway):
    app = create_app(cpu_job_store=store, vlm_gateway=gateway)
    return TestClient(app)


def test_submit_fans_out_one_gpu_job_per_photo():
    store, gateway = MemoryStore(report=_report()), FakeGateway()
    client = _client(store, gateway)
    resp = client.post("/walkthrough-jobs/wt-1/enrich")
    assert resp.status_code == 202
    assert resp.json() == {"status": "submitted", "photos": 2}
    assert len(gateway.submitted) == 2
    assert gateway.submitted[0]["note"] == "chunks below"
    mapping = store.enrichments["wt-1"]
    assert mapping["photo_1"]["output_location"] == "s3://out/1.json"


def test_submit_idempotent_no_resubmission():
    store, gateway = MemoryStore(report=_report()), FakeGateway()
    client = _client(store, gateway)
    client.post("/walkthrough-jobs/wt-1/enrich")
    resp = client.post("/walkthrough-jobs/wt-1/enrich")
    assert resp.status_code == 202
    assert resp.json() == {"status": "submitted", "photos": 2}
    assert len(gateway.submitted) == 2  # not 4


def test_submit_409_before_report_ready():
    store, gateway = MemoryStore(report=None), FakeGateway()
    client = _client(store, gateway)
    resp = client.post("/walkthrough-jobs/wt-1/enrich")
    assert resp.status_code == 409


def test_submit_503_without_gpu_gateway():
    client = _client(MemoryStore(report=_report()), None)
    resp = client.post("/walkthrough-jobs/wt-1/enrich")
    assert resp.status_code == 503


def test_poll_pending_while_any_gpu_job_unfinished():
    gateway = FakeGateway(
        statuses={"s3://out/1.json": ("ready", [{"label": "spalling", "score": 0.82}])}
    )
    store = MemoryStore(report=_report())
    client = _client(store, gateway)
    client.post("/walkthrough-jobs/wt-1/enrich")
    resp = client.get("/walkthrough-jobs/wt-1/enrich")
    assert resp.status_code == 202
    assert resp.json() == {"status": "pending", "done": 1, "total": 2}


def test_poll_ready_merges_through_gate_and_persists():
    gateway = FakeGateway(
        statuses={
            # photo_1: consistent spalling -> merged
            "s3://out/1.json": ("ready", [{"label": "spalling", "score": 0.82}]),
            # photo_2: Qwen forces "spalling" onto a furnace -> dropped
            "s3://out/2.json": ("ready", [{"label": "spalling", "score": 0.95}]),
        }
    )
    store = MemoryStore(report=_report())
    client = _client(store, gateway)
    client.post("/walkthrough-jobs/wt-1/enrich")
    resp = client.get("/walkthrough-jobs/wt-1/enrich")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    merged = body["report"]
    assert merged["per_photo"][0]["enrichment"]["label"] == "spalling"
    assert merged["per_photo"][1]["enrichment"] is None
    assert body["gate"]["kept"] == 1
    assert body["gate"]["dropped"][0]["reason"] == "inconsistent_with_observation"
    # persisted: the stored report now carries the enrichment
    assert store.outputs["wt-1"]["per_photo"][0]["enrichment"]["label"] == "spalling"


def test_poll_counts_gpu_failures_in_gate_log():
    gateway = FakeGateway(
        statuses={
            "s3://out/1.json": ("ready", [{"label": "spalling", "score": 0.82}]),
            "s3://out/2.json": ("failed", None),
        }
    )
    store = MemoryStore(report=_report())
    client = _client(store, gateway)
    client.post("/walkthrough-jobs/wt-1/enrich")
    resp = client.get("/walkthrough-jobs/wt-1/enrich")
    assert resp.status_code == 200
    body = resp.json()
    assert body["gate"]["kept"] == 1
    assert {"photo_id": "photo_2", "label": None, "confidence": None,
            "reason": "gpu_failed"} in body["gate"]["dropped"]


def test_poll_404_when_enrichment_never_requested():
    client = _client(MemoryStore(report=_report()), FakeGateway())
    resp = client.get("/walkthrough-jobs/wt-1/enrich")
    assert resp.status_code == 404
