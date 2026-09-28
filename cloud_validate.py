"""Optional acceptance checks inside the GPU worker, before it takes jobs."""
import json
import logging
import math
import subprocess
import time
from pathlib import Path

LOG = logging.getLogger(__name__)


def assert_close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (actual, expected)
        for key in expected:
            assert_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_close(left, right)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def validate(engine):
    from handler import Worker, build_engine

    started = time.monotonic()
    reference = build_engine()
    worker = Worker(engine)
    # Usage reflects cache misses; compare inference values separately.
    for filename in ("rank.json", "system_one.json"):
        job = json.loads((Path(__file__).parent / "examples" / filename).read_text())
        data = job["input"]
        actual = worker(job)
        if data["operation"] == "rank":
            expected = reference.rank(data["state"], data["candidates"])
            assert_close(actual, expected)
            assert math.isclose(sum(item["prob"] for item in actual), 1.0, abs_tol=1e-5)
            assert actual[0]["candidate"] == "The Moon's gravitational pull."
        else:
            expected = reference.answer(data["state"], data["questions"])
            assert_close(actual["answers"], expected["answers"])
            repeated = worker(job)
            assert repeated["usage"]["input_tokens"] == 0, "Repeated request missed bounded caches"
    # This string greatly exceeds the configured token limit. Verify upstream truncation.
    long_input = "hello " * 10000
    vectors, tokens = engine.embedder.embed([long_input])
    assert vectors.shape == (1, 4096)
    assert tokens <= engine.embedder.max_tokens, (tokens, engine.embedder.max_tokens)
    report = {"parity": "passed", "typed_questions": "passed", "cache_reuse": "passed",
              "long_input_truncation": "passed", "seconds": round(time.monotonic() - started, 3)}
    try:
        report["gpu_memory_snapshot"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        report["gpu_memory_snapshot"] = "unavailable"
    LOG.info("CLOUD_VALIDATION %s", json.dumps(report))
    return report
