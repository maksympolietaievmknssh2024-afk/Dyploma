#!/usr/bin/env python3
"""
Visualize evaluation results for object understanding using utils.visualization.

Reads evaluation/evaluation_results.json and saves summary + per-case figures.
If the file does not exist, prints a helpful message and exits.
"""

import os
import json

from utils.visualization import visualize_evaluation_results


def main():
    results_path = os.path.join("evaluation", "evaluation_results.json")
    if not os.path.exists(results_path):
        print(f"No evaluation results found at {results_path}.\n"
              f"Run: run_python.bat evaluate.py --model_path <checkpoint> --test_file data\\test_cases.json --output_dir evaluation")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    out_dir = os.path.dirname(results_path)
    visualize_evaluation_results(results, out_dir)


if __name__ == "__main__":
    main()