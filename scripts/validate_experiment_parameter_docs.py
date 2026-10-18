"""Validate saved settings, upstream comparison, and documentation references."""

import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_labels import method_label


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(path.open()))


def main():
    folder = ROOT / "docs/experiment_parameters_20261014"
    p = json.loads((folder / "parameter_inventory.json").read_text())
    records = p["records"]
    lookup = {r["id"]: r for r in records}
    assert len(lookup) == len(records) == 220
    index = rows(folder / "run_index.csv")
    comparison = rows(ROOT / "docs/rq2_comparison_draft_20261011/comparison_results.csv")
    assert len(index) == 188 and len(comparison) == 564
    assert {(r["dataset"], r["tokenizer"], r["source_method"]) for r in index} == {(r["dataset"], r["tokenizer"], r["method"]) for r in comparison}
    for r in index:
        assert r["parameter_record"] in lookup
        assert r["method"] == method_label(r["source_method"])
    for r in comparison:
        assert r["display_method"] == method_label(r["method"])
    for path, digest in p["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in p["current_implementation_sha256"].items():
        assert sha(ROOT / path) == digest, path
    checks = Counter()
    for r in records:
        q = r["parameters"]
        if r["family"] == "local_continuation":
            run = q["run"]
            assert run["optimizer_audit"]["resumed"]
            assert run["identity"]["max_updates"] == 32 and run["identity"]["validation_interval"] == 8
            if run.get("rw_statistics"):
                s = run["rw_statistics"]
                for g, w in s["weights"].items():
                    expected = s["history_shares"][g] / s["target_shares"][g] if s["target_shares"][g] else 0
                    assert math.isclose(w, expected, rel_tol=1e-13)
                checks["rw_vectors"] += 1
        if r["family"] == "reranking":
            applied, method = q["applied_parameters"], q["applied_method"]
            if method == "rr":
                correction = q["rr_frozen_correction"]
                ks = correction["cutoffs"]
                assert ks == [1, 5, 10, 20]
                raw = {g: sum(k / sum(ks) * correction["validation_fairness"][str(k)]["GU"][g] for k in ks) for g in correction["punishment"]}
                scale = max(map(abs, raw.values()))
                for g, w in correction["punishment"].items():
                    assert math.isclose(w, raw[g] / scale if scale else 0, rel_tol=1e-13)
                assert applied["alpha"] == correction["alpha"] and applied["floor"] == correction["epsilon"]
                checks["rr_vectors"] += 1
            if method in ("cpfair", "pmmf"):
                assert applied["lambda"] == (1 if method == "cpfair" else 30)
        if r["checkpoint"]:
            assert (ROOT / r["checkpoint"]["path"]).exists(), r["id"]
    for name in ("run_index.csv", "parameter_values.csv", "group_weights.csv", "checkpoint_index.csv", "reranking_search_spaces.csv"):
        checks[name] = len(rows(folder / name))
        assert checks[name] > 0
    upstream = ROOT / "docs/original_source_parameter_comparison_20261014"
    manifest = json.loads((upstream / "source_manifest.json").read_text())
    assert len(manifest["sources"]) == 20
    for ref in manifest["sources"]:
        assert sha(upstream / ref["path"]) == ref["snapshot_sha256"]
    assert manifest["local_inventory_sha256"] == sha(folder / "parameter_inventory.json")
    upstream_rows = rows(upstream / "parameter_comparison.csv")
    assert len(upstream_rows) == manifest["comparison_rows"] == 5784
    for r in upstream_rows:
        assert r["local_record"] in lookup
    def field(family, parameter, method=None):
        return [r for r in upstream_rows if r["family"] == family and r["parameter"] == parameter
                and (method is None or r["method"] == method)]
    for key, expected in (("optim.optimizer.lr", .001), ("optim.optimizer.weight_decay", .0001)):
        subset = field("native_generator", key)
        assert len(subset) == 12
        assert all(json.loads(r["upstream_value"]) == json.loads(r["local_value"]) == expected for r in subset)
    subset = field("classic", "n_layers", "SASRec")
    assert len(subset) == 4 and all(json.loads(r["upstream_value"]) == 1 and json.loads(r["local_value"]) == 2 for r in subset)
    subset = field("fairdiverse_training", "exposure_sample_num", "GRID + FairDual (training)")
    assert len(subset) == 12 and all(json.loads(r["upstream_value"]) == 700 and json.loads(r["local_value"]) == 200 for r in subset)
    checks["upstream_highlight_claims"] = 4
    documents = [ROOT / "docs/EXPERIMENT_PARAMETERS_AND_WEIGHTS.md", ROOT / "docs/ORIGINAL_SOURCE_PARAMETER_COMPARISON.md", folder / "selected_settings.md"]
    for path in documents:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#")[0]).exists(), (path, target)
    scripts = ["document_experiment_parameters.py", "compare_original_source_parameters.py", "update_experiment_terminology.py", "experiment_labels.py", "validate_experiment_parameter_docs.py", "build_tokenizer_reranking_tables.py"]
    for name in scripts:
        ast.parse((ROOT / "scripts" / name).read_text())
    for name in ("build_grid_update_october_2026_v1.py", "build_new_methods_deck.py", "grid_update_audit.py", "rq2_comparison_draft_20261011/assemble_tables.py"):
        ast.parse((ROOT / "docs" / name).read_text())
    checks.update(logical_result_rows=188, cutoff_result_rows=564, configuration_records=220,
                  hashed_configuration_sources=len(p["source_sha256"]), hashed_current_implementations=len(p["current_implementation_sha256"]),
                  upstream_snapshots=20, upstream_parameter_comparisons=len(upstream_rows))
    report = dict(status="passed", checks=dict(checks), scope="Configuration coverage, RW/RR formulas, strengths, checkpoint existence, hashes, readable labels, upstream snapshots, Python syntax, and documentation links. Neural tensors are not loaded or rehashed.")
    (folder / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
