# Validation status

Prepared on 2026-09-28. The user will upload and deploy manually; no endpoint was created.
No local model inference or Docker build was performed and no dependencies were installed.
Task-created temporary reference downloads and the empty test environment were removed.

## Completed locally, using only Python's standard library

- 16 automated tests passed: ranking result forwarding, all three typed-question
  formats, malformed requests and temperatures, upstream validation errors, runtime
  errors, exact cache revision lookup, encoder settings, embedding dimension check,
  base64/nonfinite embedding validation, encoder death, startup cancellation/timeout,
  real child-process shutdown, asynchronous client polling, job failure and timeout handling.
- Exact encoder/head repository commits and official image manifest were resolved.
- The public CLM wheel and head SHA-256 values were resolved; downloads were removed.
- vLLM 0.11.0 source was checked for PoolerConfig LAST/normalize fields and its
  pinned PyTorch 2.8.0 dependency.

## Pending cloud checks

1. GitHub **CPU checks**: repeat standard-library tests.
2. GitHub **Published CLM heads on cloud CPU**: real head loading and numeric adapter
   parity/cache reuse with synthetic encoder vectors. This has not been run yet.
3. Runpod image build and `pip check`: dependency resolution and container compatibility.
4. Runpod `VALIDATE_ON_STARTUP=1`: actual Qwen embeddings, 4,096 dimensions,
   worker/upstream parity, Choice/Noul/Score, repeated-cache reuse, and long-input truncation.
5. Confirm invalid job status, successful cold request after zero workers, and idle shutdown.
6. Measure cold/warm latency, peak VRAM and actual billed cost.

**Measured GPU latency, peak memory and cost are unavailable until deployment.**
The base image digest is a verified registry manifest, not a GPU-tested image.
