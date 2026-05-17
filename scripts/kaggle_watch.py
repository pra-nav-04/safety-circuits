"""Poll Kaggle every 60s and print when the kernel finishes or produces output.

Usage: python3 scripts/kaggle_watch.py
Ctrl+C to stop.
"""

import json
import pathlib
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone


def _token() -> str:
    return json.loads((pathlib.Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]


def _get(path: str):
    req = urllib.request.Request(f"https://www.kaggle.com/api/v1{path}")
    req.add_header("Authorization", f"Bearer {_token()}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def check() -> None:
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")

    # last run time from list
    kernels = _get("/kernels/list?group=PROFILE&pageSize=5&search=safety-circuits-nb")
    last_run = kernels[0].get("lastRunTime", "?") if kernels else "?"

    # output files
    out = _get("/kernels/godspeed28/safety-circuits-nb/output")
    files = [f.get("name") for f in (out or {}).get("files", [])]

    if files:
        print(f"[{now}] ✓ DONE — output files: {files}")
        print("Run:  python3 scripts/kaggle_api.py output")
        return True

    print(f"[{now}] running... lastRunTime={last_run}  (no output files yet)")
    return False


if __name__ == "__main__":
    print("Watching godspeed28/safety-circuits — Ctrl+C to stop.\n")
    while True:
        try:
            done = check()
            if done:
                break
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(60)
