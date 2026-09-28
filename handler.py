"""Runpod queue adapter. Inference stays in the upstream CLM Engine."""
import logging
import math
import os
import threading
import time
from pathlib import Path

LOG = logging.getLogger(__name__)


def validate_input(job):
    if not isinstance(job, dict) or not isinstance(job.get("input"), dict):
        raise ValueError("input must be a JSON object")
    data = job["input"]
    operation = data.get("operation")
    if operation not in ("rank", "system_one"):
        raise ValueError("operation must be 'rank' or 'system_one'")
    if "state" not in data or not isinstance(data["state"], (str, dict, list)):
        raise ValueError("state is required and must be a string, object, or array")
    temperature = data.get("temperature", 1.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a number in (0, 100]")
    if not math.isfinite(temperature) or not 0 < temperature <= 100:
        raise ValueError("temperature must be a number in (0, 100]")
    if operation == "rank":
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates must be a non-empty array of non-empty strings")
        if any(not isinstance(c, str) or not c.strip() for c in candidates):
            raise ValueError("candidates must be a non-empty array of non-empty strings")
        if data.get("instructions") is not None and not isinstance(data["instructions"], str):
            raise ValueError("instructions must be a string")
    else:
        questions = data.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object")
        for name, question in questions.items():
            if not isinstance(name, str) or not name or not isinstance(question, dict):
                raise ValueError("each question must be an object with a non-empty string ID")
            kind = question.get("type")
            if kind not in ("choice", "noul", "score"):
                raise ValueError(f"question {name!r}: type must be choice, noul, or score")
            criteria = question.get("criteria")
            if kind == "choice" and (not isinstance(criteria, dict) or not criteria):
                raise ValueError(f"question {name!r}: choice requires a non-empty criteria object")
            if kind == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
                raise ValueError(f"question {name!r}: score requires at least two criteria levels")
            if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
                raise ValueError(f"question {name!r}: noul criteria must be an object")
    return data, float(temperature)


def build_engine():
    from clm import Engine
    from clm.embedder import Embedder

    checkpoint = Path(os.environ.get("CLM_CKPT", "/opt/clm/CLM_v0.1-8B.pt"))
    if not checkpoint.is_file():
        raise RuntimeError(f"Required CLM head is missing: {checkpoint}")
    embedder = Embedder(
        url="http://127.0.0.1:8090/v1/embeddings", model="qwen3-8b",
        max_tokens=int(os.environ.get("MAX_MODEL_LEN", "2048")),
        cache_size=int(os.environ.get("EMBEDDING_CACHE_SIZE", "4096")), batch=8,
    )
    engine = Engine(embedder=embedder, checkpoint=str(checkpoint), device="cpu",
                    action_cache=os.environ.get("CLM_ACTION_CACHE", "64MiB"))
    if not engine.has("clm-latest"):
        raise RuntimeError("CLM projection heads failed to initialize")
    return engine


class Worker:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()

    def __call__(self, job):
        try:
            data, temperature = validate_input(job)
            started = time.monotonic()
            with self.lock:
                if data["operation"] == "rank":
                    result = self.engine.rank(data["state"], data["candidates"],
                                              instructions=data.get("instructions"),
                                              temperature=temperature)
                else:
                    result = self.engine.answer(data["state"], data["questions"],
                                                temperature=temperature)
            LOG.info("operation=%s duration_ms=%.1f", data["operation"],
                     (time.monotonic() - started) * 1000)
            return result
        except ValueError as exc:
            # Raising marks the Runpod job FAILED rather than COMPLETED with an error payload.
            raise ValueError(f"Invalid request: {exc}") from exc
        except Exception:
            LOG.exception("Inference failed")
            raise RuntimeError("CLM inference failed; inspect worker logs") from None


def main():
    import runpod

    logging.basicConfig(level=logging.INFO)
    worker = Worker(build_engine())
    if os.environ.get("VALIDATE_ON_STARTUP", "0") == "1":
        from cloud_validate import validate
        validate(worker.engine)
    runpod.serverless.start({"handler": worker})


if __name__ == "__main__":
    main()
