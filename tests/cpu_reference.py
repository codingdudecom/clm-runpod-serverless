"""Cloud CI only: real published CLM heads, deterministic synthetic encoder vectors.

This verifies head loading, adapter parity and caching, not Qwen or model quality.
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    from clm import Engine
    from huggingface_hub import hf_hub_download
    from cloud_validate import assert_close
    from handler import Worker

    versions = json.loads(Path("versions.json").read_text())
    with tempfile.TemporaryDirectory() as temp:
        path = hf_hub_download(versions["head_repo"], versions["head_filename"],
                              revision=versions["head_revision"], local_dir=temp)
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == versions["head_sha256"]

        class SyntheticEmbedder:
            def embed(self, texts):
                rows = []
                for text in texts:
                    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
                    row = np.random.default_rng(seed).normal(size=4096).astype(np.float32)
                    rows.append(row / np.linalg.norm(row))
                return np.stack(rows), len(texts)

        reference = Engine(embedder=SyntheticEmbedder(), checkpoint=path, device="cpu", action_cache="0")
        cached = Engine(embedder=SyntheticEmbedder(), checkpoint=path, device="cpu", action_cache="4MiB")
        worker = Worker(cached)
        assert cached.has("clm-latest")
        for filename in ("rank.json", "system_one.json"):
            job = json.loads(Path("examples", filename).read_text())
            data = job["input"]
            if data["operation"] == "rank":
                assert_close(worker(job), reference.rank(data["state"], data["candidates"]))
            else:
                actual = worker(job)
                expected = reference.answer(data["state"], data["questions"])
                assert_close(actual["answers"], expected["answers"])
                assert worker(job)["usage"]["input_tokens"] == 0
        print("Published CLM checkpoint loading, adapter parity and bounded-cache reuse passed (synthetic encoder).")


if __name__ == "__main__":
    main()
