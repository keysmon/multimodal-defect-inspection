"""Lambda entrypoint: Mangum wiring + cloud env defaults, no heavy imports.

Run in a subprocess because importing the handler mutates os.environ (setdefault)
and must not pull torch/transformers at module load.
"""
import subprocess
import sys


def test_lambda_handler_wires_mangum_and_cloud_defaults_without_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, os\n"
            "import defectlens.serve.lambda_handler as lh\n"
            "assert callable(lh.handler), 'Mangum handler is not callable'\n"
            "assert os.environ['DEFECTLENS_NO_VLM'] == '1'\n"
            "assert os.environ['DEFECTLENS_DESCRIBER'] == 'bedrock'\n"
            "assert os.environ['CARD_VECTORS_PATH'].endswith('card_vectors.npz')\n"
            "assert os.environ['AUDIO_BANK_DIR'].endswith('audio_bank')\n"
            "assert 'torch' not in sys.modules, 'torch imported at module level'\n"
            "assert 'transformers' not in sys.modules, 'transformers imported at module level'\n"
            "assert 'boto3' not in sys.modules, 'boto3 imported at module level'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_lambda_handler_respects_preexisting_env():
    """setdefault must not clobber a real value (e.g. DESCRIBER=local)."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os\n"
            "os.environ['DEFECTLENS_DESCRIBER'] = 'local'\n"
            "import defectlens.serve.lambda_handler\n"
            "assert os.environ['DEFECTLENS_DESCRIBER'] == 'local'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_lambda_handler_dispatches_worker_events_to_run_worker():
    """A worker self-invocation ({'defectlens_job': ...}) runs the CPU worker;
    any other event goes to Mangum (HTTP). Patches both seams so the routing is
    tested without a real S3/model round-trip."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import defectlens.serve.lambda_handler as lh\n"
            "from defectlens.serve import async_jobs\n"
            "seen = {}\n"
            "async_jobs.run_worker = lambda app, event: (seen.__setitem__('worker', event), {'ok': True})[1]\n"
            "lh._mangum = lambda event, context: (seen.__setitem__('mangum', event), {'http': True})[1]\n"
            # worker event -> run_worker, not Mangum
            "r1 = lh.handler({'defectlens_job': {'job_id': 'j1'}}, None)\n"
            "assert r1 == {'ok': True}, r1\n"
            "assert seen.get('worker') == {'defectlens_job': {'job_id': 'j1'}}\n"
            "assert 'mangum' not in seen\n"
            # http event -> Mangum, not run_worker
            "seen.clear()\n"
            "r2 = lh.handler({'version': '2.0', 'routeKey': 'GET /health'}, None)\n"
            "assert r2 == {'http': True}, r2\n"
            "assert seen.get('mangum') is not None\n"
            "assert 'worker' not in seen\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
