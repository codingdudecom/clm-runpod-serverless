"""Download only the small, pinned head checkpoint while building the image."""
import json
from pathlib import Path


def main():
    from huggingface_hub import hf_hub_download

    versions = json.loads(Path(__file__).with_name("versions.json").read_text())
    directory = Path("/opt/clm")
    path = hf_hub_download(
        repo_id=versions["head_repo"], filename=versions["head_filename"],
        revision=versions["head_revision"], local_dir=str(directory),
    )
    import hashlib
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if versions.get("head_sha256") and digest != versions["head_sha256"]:
        raise RuntimeError("CLM checkpoint checksum mismatch")
    print(json.dumps({"head_revision": versions["head_revision"], "sha256": digest}))


if __name__ == "__main__":
    main()
