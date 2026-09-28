"""Asynchronous Runpod client with bounded polling. Uses only Python's stdlib."""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request


def request(method, url, key, body=None):
    req = urllib.request.Request(url, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Runpod HTTP {exc.code}: {exc.read().decode()[:500]}") from None


def submit_and_wait(payload, timeout=1200, interval=2):
    key = os.environ["RUNPOD_API_KEY"]
    endpoint = os.environ["RUNPOD_ENDPOINT_ID"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", endpoint):
        raise ValueError("Invalid RUNPOD_ENDPOINT_ID")
    base = f"https://api.runpod.ai/v2/{endpoint}"
    submitted = request("POST", base + "/run", key, payload)
    job_id = submitted["id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Invalid job ID returned by Runpod")
    deadline = time.monotonic() + timeout
    result = submitted
    while result.get("status") not in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Job {job_id} still pending. It has not been cancelled; check Runpod before resubmitting.")
        time.sleep(interval)
        result = request("GET", base + "/status/" + job_id, key)
    if result["status"] != "COMPLETED":
        raise RuntimeError(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_file")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    with open(args.request_file) as stream:
        payload = json.load(stream)
    print(json.dumps(submit_and_wait(payload, timeout=args.timeout), indent=2))


if __name__ == "__main__":
    main()
