"""Kaggle API wrapper using KGAT_ Bearer token format.

Workflow for updating the kernel:
  1. Edit code locally, push to GitHub:
       git push origin main
  2. Open the notebook in browser and click "Save & Run All":
       https://www.kaggle.com/code/godspeed28/safety-circuits-nb/edit
     (The kernel script git-pulls latest from GitHub on every run.)

NOTE: `push` is intentionally disabled. The Kaggle API overwrites the
notebook with an empty file when pushing via Bearer auth, deleting all code.
Browser-based save is the only reliable way to update the notebook.

Usage:
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
    print("ERROR: API push is disabled.")
    print()
    print("Pushing via the Kaggle API (Bearer auth) overwrites the notebook with an")
    print("empty file, deleting all code. Use the browser workflow instead:")
    print()
    print("  1. Push code changes to GitHub:")
    print("       git push origin main")
    print("  2. Open the notebook and click 'Save & Run All':")
    print("       https://www.kaggle.com/code/godspeed28/safety-circuits-nb/edit")
    print()
    print("The kernel script already git-pulls the latest GitHub code on every run,")
    print("so no manual code pasting is needed after the initial browser setup.")
    sys.exit(1)


def cmd_status() -> None:
    username, _ = _load_creds()
    result = _request("GET", "/kernels/list?group=PROFILE&pageSize=5&search=safety-circuits-nb")
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
    kernel_id = f"{username}/safety-circuits-nb"
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
ACTIVE_CMDS = ["status", "output"]

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(f"Usage: python scripts/kaggle_api.py [{' | '.join(ACTIVE_CMDS)}]")
        sys.exit(1)
    CMDS[sys.argv[1]]()
