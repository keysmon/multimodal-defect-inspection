"""Health canary (Phase 5.5b). Stdlib + the runtime's bundled boto3 only.

Every 6h EventBridge invokes this. It hits CloudFront ``/api/health`` and one
baked ``/api/analyze``, and publishes to SNS if the demo is unhealthy.

What "healthy" means here matters:
  * /api/health must return HTTP 200 with ``status == "ok"`` (the app keys "ok"
    on a loaded vector store, which the no-DB cloud path always has - it does NOT
    require a reachable pgvector DB).
  * /api/analyze must return HTTP 200 with a ``classes`` list. It must NOT require
    a non-empty ``description``: Bedrock returns "" until the account's Anthropic
    use-case form is approved, and BedrockDescriber swallows that to "", so a
    description check would false-alarm on a perfectly healthy deploy.

Cold starts: the serving Lambda scales to zero and its image is large, so the
first request after idle can approach the API's 30s integration ceiling. The
canary therefore retries the health check a few times (treating 5xx/timeout as
"still warming") before it will declare failure.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

# A tiny (96x96) synthetic JPEG with a crack-like streak - enough to exercise the
# full CLIP-encode -> classify -> RAG -> describe path. The canary checks that a
# result comes back, not that the classification is correct.
CANARY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAwICQsJCAwLCgsODQwOEh4UEhEREiUbHBYeLCcuLisn"
    "KyoxN0Y7MTRCNCorPVM+QkhKTk9OLztWXFVMW0ZNTkv/2wBDAQ0ODhIQEiQUFCRLMisyS0tLS0tL"
    "S0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0v/wAARCABgAGADASIAAhEB"
    "AxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9"
    "AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6"
    "Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ip"
    "qrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEB"
    "AQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJB"
    "UQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RV"
    "VldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6"
    "wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDNUnSZeRvs"
    "XOeRuMOf5j/P10+CgKtmNh16g8dRSFd6lTiQYO5SvH8/aszLaS5HzPZFsHruhP8Agf8APubh1NNm"
    "KMCRwuAPoPb/AD/Wlc/OMj7w+Uduvcf5/rTA2fnTmM4Ix0x/nFOJBHzE5Axg87vaizQeo5EI37g"
    "SB3POPpQ0eck5PHp0/wAf8+lNk+QfKflH3Qe3rT/4QDuGenOcHHY0vMa8gkyy7Sec4wD+f86Rshm"
    "ZlAYemee9JkjbnBUnj/J+mKWQ/MDg8MTx2HehaaCuxAQUxhQwHcY4x25/zmkYMQCmMgdznv60Hap"
    "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"
    "cLq5//Z"
)

BOUNDARY = "----defectlenscanaryb0undary7a1c"
# Just above the API's 29s integration ceiling: a cold-start 504 comes back from
# APIGW at ~29s, and this only bounds a true origin hang. Keeping it tight keeps
# the whole retry budget well under CanaryFn's Lambda timeout so _notify runs.
_TIMEOUT_S = 35
_HEALTH_ATTEMPTS = 4
_RETRY_SLEEP_S = 10


def build_multipart_body(field_name, filename, content_type, payload, boundary=BOUNDARY):
    """Build a ``multipart/form-data`` body for one file field. Pure + testable."""
    crlf = b"\r\n"
    lines = [
        b"--" + boundary.encode(),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode(),
        f"Content-Type: {content_type}".encode(),
        b"",
        payload,
        b"--" + boundary.encode() + b"--",
        b"",
    ]
    body = crlf.join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def _get_json(url):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return resp.status, json.loads(resp.read().decode())


def _post_multipart_json(url, body, content_type):
    req = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return resp.status, json.loads(resp.read().decode())


def _notify(subject, message):
    import boto3  # bundled in the Lambda runtime; imported lazily so tests need no AWS

    topic_arn = os.environ["SNS_TOPIC_ARN"]
    boto3.client("sns").publish(TopicArn=topic_arn, Subject=subject[:100], Message=message)


def _check_health(base):
    """Retry a few times so a cold start (still loading models) is not a failure.

    The retry budget (attempts x (request + sleep)) is kept well under CanaryFn's
    Lambda timeout so a genuinely-down endpoint still reaches _notify instead of
    the canary being killed mid-retry.
    """
    last = ""
    for attempt in range(_HEALTH_ATTEMPTS):
        try:
            status, data = _get_json(base + "/api/health")
            if status == 200 and data.get("status") == "ok":
                return None
            last = f"health HTTP {status} body={data}"
        except urllib.error.HTTPError as exc:  # 5xx while warming
            last = f"health HTTP {exc.code}"
        except Exception as exc:  # timeout / transient
            last = f"health error: {exc}"
        if attempt < _HEALTH_ATTEMPTS - 1:  # no sleep after the final attempt
            time.sleep(_RETRY_SLEEP_S)
    return last


def _check_analyze(base):
    body, content_type = build_multipart_body(
        "file", "canary.jpg", "image/jpeg", base64.b64decode(CANARY_JPEG_B64)
    )
    try:
        status, data = _post_multipart_json(base + "/api/analyze", body, content_type)
    except Exception as exc:
        return f"analyze error: {exc}"
    if status != 200:
        return f"analyze HTTP {status}"
    if not data.get("classes"):
        return f"analyze returned no classes: keys={sorted(data)}"
    return None


def handler(event, context):
    base = os.environ["CF_BASE_URL"].rstrip("/")
    failures = [f for f in (_check_health(base), _check_analyze(base)) if f]
    if failures:
        message = "DefectLens canary FAILED:\n- " + "\n- ".join(failures)
        _notify("DefectLens canary FAILED", message)
        return {"ok": False, "failures": failures}
    return {"ok": True}
