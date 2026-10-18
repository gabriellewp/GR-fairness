"""Validate complete sweep shards and recompute the global validation frontier."""

import argparse
import json
from pathlib import Path

from .tune_beauty_reranking import CUTOFFS, task_plan, write_outputs


def merge(task_dir, output_dir):
    directories = sorted(Path(task_dir).glob("task_*"))
    if not directories:
        raise ValueError("No tuning tasks found")
    manifests, rows_by_task = {}, {}
    common = None
    controls = {}
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        index = manifest["task_index"]
        if type(index) is not int or index in manifests:
            raise ValueError("Task indices must be unique integers")
        identity = {key: value for key, value in manifest.items()
                    if key not in {"task_index", "evaluated_cutoffs", "resources"}}
        if common is None:
            common = identity
        elif identity != common:
            raise ValueError("Tasks have different inputs, grids, sources, or settings")
        plan = task_plan(manifest["method"], manifest["strengths"], manifest["floors"])
        if plan != manifest["task_plan"] or len(plan) != manifest["task_count"]:
            raise ValueError("Task plan does not match its grid")
        if not 0 <= index < len(plan):
            raise ValueError("Task index outside the sweep grid")
        task = plan[index]
        cutoffs = list(CUTOFFS) if manifest["method"] == "rr" else [task["k"]]
        if manifest["evaluated_cutoffs"] != cutoffs:
            raise ValueError("Task evaluated the wrong cutoffs")
        parameters = task if manifest["method"] == "rr" else {"lambda": task["lambda"]}
        rows = json.loads((directory / "validation_rows.json").read_text())
        expected = {(k, json.dumps(p, sort_keys=True)) for k in cutoffs for p in ({}, parameters)}
        seen = set()
        trials = []
        for row in rows:
            key = (row["k"], row["parameters"])
            if key not in expected or key in seen or row["method"] != manifest["method"] or row["requests"] != manifest["request_count"]:
                raise ValueError("Task rows are duplicate, unexpected, or misaligned")
            seen.add(key)
            if row["parameters"] == "{}":
                if row["k"] in controls and controls[row["k"]] != row:
                    raise ValueError("Control metrics differ between tasks")
                controls[row["k"]] = row
            else:
                trials.append(row)
        if seen != expected:
            raise ValueError("Task has missing result rows")
        manifests[index] = manifest
        rows_by_task[index] = trials
    if set(manifests) != set(range(common["task_count"])):
        raise ValueError("Sweep is incomplete: missing task indices")
    if set(controls) != set(CUTOFFS):
        raise ValueError("Sweep is missing control cutoffs")
    rows = [controls[k] for k in CUTOFFS]
    rows.extend(row for index in range(common["task_count"]) for row in rows_by_task[index])
    manifest = {**common, "task_index": None, "evaluated_cutoffs": list(CUTOFFS),
                "task_resources": {str(index): manifests[index]["resources"] for index in sorted(manifests)},
                "merged_tasks": [str(directory.resolve()) for directory in directories]}
    return write_outputs(output_dir, rows, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(merge(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
