"""Convert run_experiment.py into a Kaggle-ready .ipynb and update the push payload."""
import json, pathlib

HERE = pathlib.Path(__file__).parent
script = (HERE / "run_experiment.py").read_text()

nb = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": script
        }
    ]
}

out = HERE / "kernel.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Written {out}")
