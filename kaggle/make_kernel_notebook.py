"""Convert run_experiment.py into a Kaggle-ready .ipynb (nbformat 4.4)."""
import json, pathlib, uuid

HERE = pathlib.Path(__file__).parent
script = (HERE / "run_experiment.py").read_text()

# nbformat 4.4: source is a list of lines (each ending in \n except the last);
# cells need an `id` field (8-char hex).
lines = script.splitlines(keepends=True)

nb = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "outputs": [],
            "source": lines
        }
    ]
}

out = HERE / "kernel.ipynb"
out.write_text(json.dumps(nb))
# quick sanity check
assert json.loads(out.read_text())["cells"][0]["source"] == lines
print(f"Written {out}  ({out.stat().st_size} bytes, {len(lines)} source lines)")
