"""Minimal Kaggle API wrapper that supports the new KGAT_ Bearer token format.

Usage:
  python scripts/kaggle_api.py push
  python scripts/kaggle_api.py status
  python scripts/kaggle_api.py output
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import urllib.request
import urllib.error
import zipfile
import tempfile
import shutil


def _load_creds() -> tuple[str, str]:
    creds_file = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    d = json.loads(creds_file.read_text())
    return d["username"], d["key"]


def _request(method: str, path: str, body: dict | None = None) -> dict:
    username, token = _load_creds()
    url = f"https://www.kaggle.com/api/v1{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)


def _request_binary(path: str, dest: pathlib.Path) -> None:
    username, token = _load_creds()
    url = f"https://www.kaggle.com/api/v1{path}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def cmd_push() -> None:
    username, _ = _load_creds()
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    kernel_dir = repo_root / "kaggle"
    meta_path  = kernel_dir / "kernel-metadata.json"
    script_path = kernel_dir / "run_experiment.py"

    meta = json.loads(meta_path.read_text())
    kernel_id = meta["id"]          # godspeed28/safety-circuits
    slug = kernel_id.split("/")[1]

    source = script_path.read_text()

    # Correct field names per Kaggle API spec (camelCase, no `id` for new kernels).
    # machineShape=T4 is required to actually provision a GPU on free Kaggle.
    payload = {
        "newTitle": meta.get("title", slug),
        "sourceCode": source,
        "language": meta.get("language", "python"),
        "kernelType": meta.get("kernel_type", "script"),
        "isPrivate": meta.get("is_private", True),
        "enableGpu": True,
        "enableInternet": True,
        "machineShape": "T4",
        "datasetDataSources": meta.get("dataset_sources", []),
        "competitionDataSources": meta.get("competition_sources", []),
        "kernelDataSources": meta.get("kernel_sources", []),
    }

    # Include kernelId if stored from a previous push (needed to update vs create).
    id_cache = repo_root / ".kaggle_kernel_id"
    if id_cache.exists():
        numeric_id = int(id_cache.read_text().strip())
        payload["id"] = numeric_id
        print(f"Updating existing kernel {kernel_id} (id={numeric_id}) ...")
    else:
        print(f"Creating new kernel {kernel_id} ...")

    result = _request("POST", "/kernels/push", body=payload)
    # Cache the numeric kernelId for future updates
    if result.get("kernelId"):
        id_cache.write_text(str(result["kernelId"]))
    print("Push result:", json.dumps(result, indent=2))


def cmd_status() -> None:
    username, _ = _load_creds()
    result = _request("GET", "/kernels/list?group=PROFILE&pageSize=5&search=safety-circuits")
    kernels = result if isinstance(result, list) else []
    if not kernels:
        print("No kernel found yet — it may still be provisioning.")
        return
    k = kernels[0]
    ref = k.get("ref", "?")
    version = k.get("currentVersionNumber", 0)
    last_run = k.get("lastRunTime", "?")
    gpu = k.get("enableGpu", False)
    print(f"Kernel:   {ref}")
    print(f"Version:  {version}")
    print(f"Last run: {last_run}")
    print(f"GPU:      {gpu}")


def cmd_output() -> None:
    username, _ = _load_creds()
    kernel_id = f"{username}/safety-circuits"
    out_dir = pathlib.Path(__file__).resolve().parents[1] / "results" / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = _request("GET", f"/kernels/{kernel_id}/output")
    files = result.get("files", [])
    if not files:
        print("No output files yet — kernel may still be running.")
        return
    for f in files:
        name = f["name"]
        url_path = f["url"]
        dest = out_dir / name
        print(f"Downloading {name} → {dest}")
        _request_binary(url_path.replace("https://www.kaggle.com/api/v1", ""), dest)
    print(f"\nAll files saved to {out_dir}/")


CMDS = {"push": cmd_push, "status": cmd_status, "output": cmd_output}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(f"Usage: python scripts/kaggle_api.py [{' | '.join(CMDS)}]")
        sys.exit(1)
    CMDS[sys.argv[1]]()
