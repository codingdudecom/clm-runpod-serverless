"""Supervise the private encoder and queue worker; never accept jobs before ready."""
import base64
import json
import logging
import math
import os
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LOG = logging.getLogger(__name__)
CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")


def resolve_encoder(versions, cache_root=CACHE_ROOT):
    repo = cache_root / ("models--" + versions["encoder_repo"].replace("/", "--"))
    snapshot = repo / "snapshots" / versions["encoder_revision"]
    if not (snapshot / "config.json").is_file():
        raise RuntimeError(
            f"Pinned encoder snapshot missing: {snapshot}. Set Runpod cached model to "
            f"{versions['encoder_repo']}. Cache must contain revision {versions['encoder_revision']}; "
            "do not silently use a different revision. See README.md cache troubleshooting."
        )
    if not any(snapshot.glob("*.safetensors")):
        raise RuntimeError(f"No safetensors weights found in {snapshot}")
    return snapshot


def encoder_command(snapshot):
    return ["vllm", "serve", str(snapshot), "--served-model-name", "qwen3-8b",
            "--runner", "pooling", "--convert", "embed", "--dtype", "bfloat16",
            "--pooler-config", '{"pooling_type":"LAST","normalize":true}',
            "--host", "127.0.0.1", "--port", "8090", "--enforce-eager",
            "--max-model-len", os.environ.get("MAX_MODEL_LEN", "2048"),
            "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"),
            "--max-num-seqs", os.environ.get("MAX_NUM_SEQS", "8")]


def probe_embedding():
    payload = {"model": "qwen3-8b", "input": ["CLM readiness check"],
               "encoding_format": "base64", "truncate_prompt_tokens": 2048}
    request = urllib.request.Request("http://127.0.0.1:8090/v1/embeddings",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.load(response)
    vector = result["data"][0]["embedding"]
    if isinstance(vector, str):
        raw = base64.b64decode(vector, validate=True)
        if len(raw) != 4096 * 4:
            raise RuntimeError("Readiness embedding must contain 4096 float32 values")
        vector = struct.unpack("<4096f", raw)
    if len(vector) != 4096 or not all(math.isfinite(v) for v in vector):
        raise RuntimeError("Readiness embedding must contain 4096 finite values")
    return True


def wait_ready(encoder, timeout, stopping, probe=probe_embedding):
    deadline = time.monotonic() + timeout
    last_error = "encoder has not responded"
    next_log = 0
    while time.monotonic() < deadline:
        if stopping():
            raise InterruptedError("Shutdown requested during startup")
        if encoder.poll() is not None:
            raise RuntimeError(f"Encoder exited during startup (code {encoder.returncode})")
        try:
            if probe():
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = str(exc)
        if time.monotonic() >= next_log:
            LOG.info("Waiting for encoder: %s", last_error)
            next_log = time.monotonic() + 15
        time.sleep(1)
    raise RuntimeError(f"Encoder readiness timed out after {timeout}s: {last_error}")


def stop_children(children):
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15
    for child in children:
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def main():
    logging.basicConfig(level=logging.INFO)
    children = []
    stopping = False

    def on_signal(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        versions = json.loads(Path(__file__).with_name("versions.json").read_text())
        snapshot = resolve_encoder(versions)
        started = time.monotonic()
        LOG.info("Starting encoder revision=%s pooling=LAST dtype=bfloat16", versions["encoder_revision"])
        encoder = subprocess.Popen(encoder_command(snapshot), start_new_session=True)
        children.append(encoder)
        wait_ready(encoder, int(os.environ.get("STARTUP_TIMEOUT_SECONDS", "900")), lambda: stopping)
        LOG.info("Encoder ready startup_seconds=%.2f dimension=4096", time.monotonic() - started)
        if stopping:
            return 0
        worker = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("handler.py"))],
                                  start_new_session=True)
        children.append(worker)
        while not stopping:
            if encoder.poll() is not None:
                raise RuntimeError(f"Encoder exited unexpectedly (code {encoder.returncode})")
            if worker.poll() is not None:
                return worker.returncode
            time.sleep(0.5)
        return 0
    except InterruptedError:
        return 0
    except Exception:
        LOG.exception("Worker startup/supervision failed")
        return 1
    finally:
        stop_children(children)


if __name__ == "__main__":
    sys.exit(main())
