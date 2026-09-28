"""Record completed cold/warm job timings; this script creates billable jobs."""
import argparse
import json
import time
from pathlib import Path
from client import submit_and_wait


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default="examples/system_one.json")
    parser.add_argument("--output", default="measurements.json")
    parser.add_argument("--idle-wait", type=int, default=30)
    args = parser.parse_args()
    payload = json.loads(Path(args.request).read_text())
    records = []
    for phase in ["initial", "warm_1", "warm_2", "after_idle"]:
        if phase == "after_idle":
            # Platform shutdown can take longer than configured idle timeout.
            time.sleep(args.idle_wait)
        started = time.monotonic()
        result = submit_and_wait(payload)
        records.append({"phase": phase, "wall_seconds": time.monotonic() - started,
                        "delay_ms": result.get("delayTime"), "execution_ms": result.get("executionTime"),
                        "worker_id": result.get("workerId"), "usage": result.get("output", {}).get("usage")
                        if isinstance(result.get("output"), dict) else None})
    report = {"requests": records,
              "note": "Verify worker shutdown in console. Initial/after_idle labels do not prove cold starts. "
                      "Check Runpod billing for cost; request execution time omits billed initialization and idle time."}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
