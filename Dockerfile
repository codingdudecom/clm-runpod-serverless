FROM --platform=linux/amd64 vllm/vllm-openai:v0.11.0@sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a

WORKDIR /app
# Satisfy PyGObject's missing dependency inherited from the upstream image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libcairo2-dev pkg-config python3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt constraints.txt versions.json ./
RUN python3 -m pip install --no-cache-dir -c constraints.txt -r requirements.txt \
    && python3 -m pip check
COPY download_head.py ./
RUN python3 download_head.py
COPY handler.py startup.py cloud_validate.py ./
COPY examples/rank.json examples/system_one.json ./examples/
ENV PYTHONUNBUFFERED=1 \
    CLM_CKPT=/opt/clm/CLM_v0.1-8B.pt \
    CLM_DEVICE=cpu \
    CLM_ACTION_CACHE=64MiB \
    MAX_MODEL_LEN=2048 \
    GPU_MEMORY_UTILIZATION=0.85 \
    MAX_NUM_SEQS=8 \
    EMBEDDING_CACHE_SIZE=4096 \
    STARTUP_TIMEOUT_SECONDS=900
# Replace the upstream image's vLLM entrypoint with our supervisor.
ENTRYPOINT ["python3", "-u", "/app/startup.py"]
