# CLM on Runpod Serverless

A queue worker for **candidate ranking** and **Choice / Noul / Score decisions** using
`Contrastive-LM/CLM-v0.1-8B` projection heads and a `Qwen/Qwen3-8B` encoder.
CLM scores supplied candidates; it does not generate text.

**No local installation, Docker build, or model download is needed.** Upload these
small source files to GitHub, let Runpod build the image, and run the examples in
AWS CloudShell or another cloud terminal.

## 1. Upload to GitHub

1. Create a repository named **I** in your GitHub account.
2. Extract the supplied source ZIP and upload the contents using GitHub's
   **Add file → Upload files**. Put `Dockerfile` directly at the repository root,
   alongside `handler.py`, `startup.py`, `versions.json`, and the other files.
   Do not put them inside an extra `CLM/` directory.
3. Include `examples/`, `tests/`, and `.github/workflows/`. On macOS, press
   **Command–Shift–Period** in Finder to show hidden `.github` and ignore files.
   If GitHub's uploader skips hidden directories, add the three workflow files
   using **Add file → Create new file** with their full paths.
4. The **CPU checks** workflow runs without installing dependencies. For a real
   projection-head loading check, manually run **Published CLM heads on cloud CPU**
   from the Actions tab. It installs CPU libraries only on a disposable GitHub
   runner, and downloads only the head checkpoint. It does not test the Qwen GPU encoder.

## 2. Build and deploy on Runpod

1. In Runpod **Settings → Connections**, connect GitHub and grant access only
   to this repository.
2. In **Serverless → New Endpoint → Import Git Repository**, select the repository,
   `main` branch, and root `Dockerfile`. Select **Queue**, not Load Balancer.
3. Configure these settings:

| Setting | Initial value |
| --- | --- |
| GPU count per worker | 1 |
| GPU | RTX 4090 / 24 GB where available |
| Active/minimum workers | 0 |
| Maximum workers | 1 |
| Idle timeout | 10 seconds |
| Execution timeout | 900 seconds |
| Cached model / Model field | `Qwen/Qwen3-8B` |
| Container disk | 30 GB |
| Environment variable | `VALIDATE_ON_STARTUP=1` for first acceptance run |

4. Review the displayed GPU rate, then deploy. No Docker Hub account or AWS build
   infrastructure is required for this route. The build downloads the small CLM
   head checkpoint; Runpod caches the encoder separately.
5. Inspect **Builds** and worker logs. The first request activates a GPU worker;
   use the asynchronous API below because cold starts can take several minutes.
6. Confirm `Encoder ready ... dimension=4096` and `CLOUD_VALIDATION` with passing
   parity, typed-question, cache, and truncation checks. A failed acceptance check
   prevents that worker from accepting jobs.
7. After acceptance, remove `VALIDATE_ON_STARTUP` or set it to `0` to avoid repeating
   those checks at every cold start. This is a normal endpoint configuration update.

No HF token is required for these public repositories. If your setup needs one,
put a read-only token in Runpod's secret environment variables as `HF_TOKEN`.
Do not include tokens in GitHub files, Docker build arguments, or public environment values.
The client uses your **Runpod API key**, stored only in the client environment.

## 3. Send requests without installing anything locally

Open AWS CloudShell and clone your repository there, or paste the JSON directly
into the curl request. For a private repository, use an authenticated cloud GitHub
session; do not embed credentials in clone URLs.

```sh
git clone https://github.com/YOUR_USERNAME/clm-runpod-serverless.git
cd clm-runpod-serverless
export RUNPOD_ENDPOINT_ID='YOUR_ENDPOINT_ID'
read -s -p 'Runpod API key: ' RUNPOD_API_KEY
export RUNPOD_API_KEY
```

### Python: submit and poll automatically

The client needs only Python's standard library, already available in CloudShell:

```sh
python3 examples/client.py examples/rank.json
python3 examples/client.py examples/system_one.json
```

It waits up to 20 minutes and prints the completed Runpod job, including `output`,
`delayTime`, and `executionTime`. A polling timeout does not cancel the job; check
its status before submitting a duplicate.

### curl: asynchronous API

```sh
curl --fail-with-body -sS \
  "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/rank.json
```

Copy the returned `id`, then poll until `COMPLETED` or `FAILED`:

```sh
export JOB_ID='ID_FROM_PREVIOUS_RESPONSE'
curl --fail-with-body -sS \
  "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/$JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

## API contract

All jobs have the shape `{"input": {...}}`.

| Field | Behavior |
| --- | --- |
| `operation` | Required: `rank` or `system_one` |
| `state` | Required: string, object, or array, rendered by upstream CLM |
| `temperature` | Optional number in `(0, 100]`; default `1.0` |
| `candidates` | Required for `rank`: non-empty list of non-empty strings |
| `instructions` | Optional string for `rank` |
| `questions` | Required for `system_one`: non-empty object of typed questions |

`rank` returns a list under Runpod's `output`, sorted best first, with `rank`,
`candidate`, and `prob`. `system_one` returns the upstream `model`, `answers`, and
`usage` object. The upstream name `clm-latest` is an API alias; the actual checkpoint
and encoder revisions are fixed in `versions.json`.

Question types are lowercase `choice`, `noul`, and `score`. Choice has a `criteria`
object mapping labels to descriptions; Score has an ordered list of at least two
levels; Noul has instructions and optional `true` / `false` criteria. See
`examples/system_one.json` for all three.

Invalid requests raise readable errors so Runpod marks the job `FAILED`. Inference
errors also fail the job, with the underlying cause in worker logs. One job is
processed at a time per worker.

The encoder truncates **each rendered state/question text and candidate text to
the final 2,048 tokens** by default. Long states can lose their beginning. To use
longer inputs, increase `MAX_MODEL_LEN` (it controls both vLLM and the CLM embedder),
then repeat GPU memory and truncation validation. Probabilities are relative to
the supplied candidate set, not calibrated standalone confidence scores.

## Request examples

Paste any example into Runpod's **Requests** editor and click **Run**.

### Rank answers or actions

Returns candidates sorted best first, with probabilities.

```json
{
  "input": {
    "operation": "rank",
    "state": "What causes tides on Earth?",
    "instructions": "Rank the answers by scientific accuracy.",
    "candidates": [
      "The Moon's gravitational pull.",
      "Photosynthesis in plants.",
      "Because the Earth is round."
    ],
    "temperature": 1.0
  }
}
```

### Choice: select a category

Map category IDs to descriptions in `criteria`.

```json
{
  "input": {
    "operation": "system_one",
    "state": "I was charged twice for my subscription. Please refund the duplicate payment.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
          "billing": "Payments, invoices, charges, and refunds",
          "technical": "Software bugs, errors, and outages",
          "sales": "Product information and purchasing"
        }
      }
    }
  }
}
```

### Noul: assess a yes/no-style question

```json
{
  "input": {
    "operation": "system_one",
    "state": "Our production website is offline and customers cannot complete purchases.",
    "questions": {
      "urgent": {
        "type": "noul",
        "instructions": "Does this issue require urgent attention?"
      }
    }
  }
}
```

### Score: assess an ordered scale

List `criteria` from low to high.

```json
{
  "input": {
    "operation": "system_one",
    "state": "This is the third time I have contacted support. I am extremely angry that this still hasn't been fixed!",
    "questions": {
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": [
          "Calm",
          "Slightly frustrated",
          "Frustrated",
          "Very angry"
        ]
      }
    }
  }
}
```

### Combine question types

Ask several questions about the same state in one job.

```json
{
  "input": {
    "operation": "system_one",
    "state": "My invoice was charged twice. I need my money back today, and nobody is answering my emails!",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Charges, invoices, and refunds",
          "technical": "Bugs and outages"
        }
      },
      "urgency": {
        "type": "noul",
        "instructions": "Does this need urgent attention?"
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      }
    },
    "temperature": 1.0
  }
}
```

`temperature` is optional (default `1.0`): lower values sharpen probabilities;
higher values flatten them. CLM scores possibilities; it does not generate text.

## Cold starts, memory, and cost

- BF16 encoder weights occupy roughly 16 GB before runtime overhead. 24 GB is the
  initial configuration, not a measured fit guarantee. If initialization runs
  out of GPU memory, use a 48 GB GPU without quantizing or changing the encoder.
- Startup uses explicit LAST pooling, BF16, eager execution, 85% GPU utilization,
  eight encoder sequences, CPU projection heads, a 4,096-entry embedding LRU,
  and a 64 MiB projection arena. Both caches disappear when a worker shuts down.
- Workers scale to zero; cached weights avoid repeat downloads, but loading and
  initializing the encoder still takes time. Worker initialization and idle
  time can be billable. Maximum workers `1` limits parallel capacity, not total spend.
- In CloudShell, run `python3 examples/benchmark.py` for four billable test jobs
  (initial, two warm, after idle). It saves request timing and worker IDs to
  `measurements.json`; verify zero workers in the console before labeling a request cold.
- Record GPU peak memory from the console monitoring graph across initialization
  and requests. `CLOUD_VALIDATION` logs only a memory snapshot, not peak memory.
  Obtain actual cost from Runpod billing; execution-time totals omit initialization
  and idle costs. Do not increase capacity until these measurements are acceptable.

## Cache and startup troubleshooting

The worker resolves this exact encoder snapshot:

```text
/runpod-volume/huggingface-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/
```

It deliberately does not choose an arbitrary cached snapshot or download the full
encoder on startup. If Runpod's cache lacks this revision, inspect the cache in a
Runpod cloud environment. Either populate the pinned revision in a network volume
at that path using Hugging Face's `snapshot_download(..., revision=..., cache_dir=...)`,
or update `versions.json` to the available upstream revision and rerun all acceptance
checks. A network volume introduces storage charges; the normal cached-model route
does not require one. Each endpoint supports one cached model, so cache Qwen, not
the small CLM head repository.

Readiness requires a successful embedding with 4,096 finite values. Startup times
out after 900 seconds, detects encoder death, and terminates both child processes
on shutdown. Failures log stack traces to stdout/stderr. Persistent failures should
be fixed before resubmitting jobs to avoid paying for repeated failed starts.

## Optional Docker Hub publishing

If Runpod's GitHub builder fails, add GitHub Actions secrets `DOCKERHUB_USERNAME`
and `DOCKERHUB_TOKEN` (a Docker Hub access token). Manually run **Publish optional
Docker Hub image**, choosing a version tag such as `v0.1.0`. It builds `linux/amd64`
on a GitHub runner and pushes `YOUR_USERNAME/clm-runpod-serverless:v0.1.0`.
Then use **Import from Docker Registry** in Runpod with the same endpoint settings.
Attach registry credentials in Runpod if the image is private. No local Docker is used.

## Validation and versions

See `VALIDATION.md` for checks actually run and checks still requiring cloud execution.
The base image digest, primary dependencies, encoder revision, and head checksum
are recorded in `versions.json`. The inherited vLLM/PyTorch stack is constrained
and `pip check` runs at build time. Additional dependencies required by the Runpod
SDK are resolved during the cloud build; this is not a fully frozen transitive lockfile.

## References

- [CLM model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
- [CLM upstream implementation](https://github.com/Contrastive-LM/CLM)
- [Runpod GitHub builds](https://docs.runpod.io/serverless/workers/github-integration)
- [Runpod model caching](https://docs.runpod.io/serverless/endpoints/model-caching)
