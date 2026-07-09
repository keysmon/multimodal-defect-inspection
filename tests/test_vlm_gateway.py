"""Tests for the SageMaker async VLM gateway (Phase 5.5c).

Pure helpers (payload/response/URI shaping) and env gating are tested directly;
submit()/status() are driven with fake s3 + sagemaker-runtime clients injected via
the constructor, so no AWS is touched. boto3 is imported lazily, so the module
imports with only the stdlib.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys

import pytest

from defectlens.serve.vlm_gateway import (
    SageMakerAsyncGateway,
    build_gateway_from_env,
    build_payload,
    parse_output,
    split_s3_uri,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_build_payload_encodes_image_and_includes_note():
    body = build_payload(b"\xff\xd8imgbytes", note="musty smell")
    data = json.loads(body)
    assert base64.b64decode(data["image_b64"]) == b"\xff\xd8imgbytes"
    assert data["note"] == "musty smell"


def test_build_payload_omits_note_when_none():
    data = json.loads(build_payload(b"x", note=None))
    assert "note" not in data


def test_parse_output_reshapes_classes_to_label_score_dicts():
    raw = json.dumps({"classes": [["crack", 0.87], ["spalling", 0.1]]}).encode()
    assert parse_output(raw) == [
        {"label": "crack", "score": 0.87},
        {"label": "spalling", "score": 0.1},
    ]


def test_split_s3_uri_parses_and_rejects_bad():
    assert split_s3_uri("s3://bucket/a/b/c.json") == ("bucket", "a/b/c.json")
    for bad in ("bucket/key", "s3://bucket", "s3://", "https://x/y", "s3:///key"):
        with pytest.raises(ValueError):
            split_s3_uri(bad)


# ---------------------------------------------------------------------------
# Env gating
# ---------------------------------------------------------------------------


def test_build_gateway_from_env_none_when_endpoint_unset(monkeypatch):
    monkeypatch.delenv("SAGEMAKER_ENDPOINT", raising=False)
    monkeypatch.setenv("ASYNC_INPUT_S3", "s3://b/in/")
    assert build_gateway_from_env() is None


def test_build_gateway_from_env_none_when_input_bucket_unset(monkeypatch):
    monkeypatch.setenv("SAGEMAKER_ENDPOINT", "defectlens-vlm-async")
    monkeypatch.delenv("ASYNC_INPUT_S3", raising=False)
    assert build_gateway_from_env() is None


def test_build_gateway_from_env_builds_when_configured(monkeypatch):
    monkeypatch.setenv("SAGEMAKER_ENDPOINT", "defectlens-vlm-async")
    monkeypatch.setenv("ASYNC_INPUT_S3", "s3://bucket/phase5/sagemaker/async-in/")
    monkeypatch.setenv("AWS_REGION", "ca-central-1")
    gw = build_gateway_from_env()
    assert isinstance(gw, SageMakerAsyncGateway)
    assert gw.enabled
    assert gw.endpoint_name == "defectlens-vlm-async"
    assert gw.region == "ca-central-1"


# ---------------------------------------------------------------------------
# submit() / status() with fake AWS clients
# ---------------------------------------------------------------------------


class _Body:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class _NoSuchKey(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})  # (bucket, key) -> bytes
        self.puts = []

    def put_object(self, Bucket, Key, Body, ContentType):
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body, "ContentType": ContentType})
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _NoSuchKey()
        return {"Body": _Body(self.objects[(Bucket, Key)])}


class FakeRuntime:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke_endpoint_async(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _gateway(s3, runtime):
    return SageMakerAsyncGateway(
        "defectlens-vlm-async",
        "s3://bucket/phase5/sagemaker/async-in/",
        region="ca-central-1",
        s3_client=s3,
        runtime_client=runtime,
    )


def test_submit_uploads_payload_and_invokes_async():
    s3 = FakeS3()
    runtime = FakeRuntime(
        {
            "InferenceId": "job-123",
            "OutputLocation": "s3://bucket/phase5/sagemaker/async-out/job-123.out",
            "FailureLocation": "s3://bucket/phase5/sagemaker/async-fail/job-123.out",
        }
    )
    gw = _gateway(s3, runtime)

    result = gw.submit(b"\xff\xd8image", note="hairline crack")

    # payload landed under async-in/ with the image + note
    assert len(s3.puts) == 1
    put = s3.puts[0]
    assert put["Bucket"] == "bucket"
    assert put["Key"].startswith("phase5/sagemaker/async-in/")
    assert put["Key"].endswith(".json")
    body = json.loads(put["Body"])
    assert base64.b64decode(body["image_b64"]) == b"\xff\xd8image"
    assert body["note"] == "hairline crack"

    # invoked with the uploaded object as InputLocation
    assert len(runtime.calls) == 1
    call = runtime.calls[0]
    assert call["EndpointName"] == "defectlens-vlm-async"
    assert call["InputLocation"] == f"s3://bucket/{put['Key']}"

    assert result == {
        "job_id": "job-123",
        "output_location": "s3://bucket/phase5/sagemaker/async-out/job-123.out",
        "failure_location": "s3://bucket/phase5/sagemaker/async-fail/job-123.out",
    }


def test_status_ready_when_output_written():
    out = "s3://bucket/phase5/sagemaker/async-out/job.out"
    body = json.dumps({"classes": [["crack", 0.9], ["spalling", 0.1]]}).encode()
    s3 = FakeS3({("bucket", "phase5/sagemaker/async-out/job.out"): body})
    gw = _gateway(s3, FakeRuntime({}))

    state, classes = gw.status(out)
    assert state == "ready"
    assert classes == [{"label": "crack", "score": 0.9}, {"label": "spalling", "score": 0.1}]


def test_status_pending_when_no_output_or_failure():
    s3 = FakeS3()  # neither object exists yet
    gw = _gateway(s3, FakeRuntime({}))
    state, classes = gw.status(
        "s3://bucket/phase5/sagemaker/async-out/job.out",
        failure_location="s3://bucket/phase5/sagemaker/async-fail/job.out",
    )
    assert state == "pending"
    assert classes is None


def test_status_failed_when_failure_object_present():
    fail_key = "phase5/sagemaker/async-fail/job.out"
    s3 = FakeS3({("bucket", fail_key): b'{"error": "boom"}'})
    gw = _gateway(s3, FakeRuntime({}))
    state, classes = gw.status(
        "s3://bucket/phase5/sagemaker/async-out/job.out",
        failure_location=f"s3://bucket/{fail_key}",
    )
    assert state == "failed"
    assert classes is None


def test_disabled_gateway_reports_not_enabled():
    gw = SageMakerAsyncGateway("", "", region=None)
    assert gw.enabled is False


def test_module_import_does_not_pull_in_boto3():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import defectlens.serve.vlm_gateway\n"
            "assert 'boto3' not in sys.modules, 'boto3 imported at module level'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
