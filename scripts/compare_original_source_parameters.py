"""Compare saved experiment settings with pinned released source defaults.

Source snapshots are retrieved separately. This generator is offline and does
not infer an upstream experiment's effective settings from an example default.
"""

import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml

from scripts.paper.archive_paths import optional_source_root, optional_output_root
ROOT = optional_source_root(Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))
from scripts.experiment_labels import method_label

OUT = optional_output_root(ROOT / "docs/original_source_parameter_comparison_20261014")
MISSING = object()


def flatten(value, prefix=""):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from flatten(child, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, value


def status(source, local):
    if local is MISSING:
        return "not present at the same configuration path"
    if source == "???":
        return "upstream requires a runtime override"
    if isinstance(source, str) and "${" in source:
        return "same interpolation expression" if source == local else "interpolation requires contextual resolution"
    if source == local:
        return "same saved value"
    if isinstance(source, (str, float, int)) and isinstance(local, (str, float, int)):
        try:
            if float(source) == float(local):
                return "same numeric scalar"
        except ValueError:
            pass
    return "different saved value"


def main():
    manifest_path = OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    refs = {r["id"]: r for r in manifest["sources"]}
    snapshots = {}
    for key, ref in refs.items():
        path = OUT / ref["path"]
        raw = path.read_bytes()
        ref["snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
        snapshots[key] = yaml.safe_load(raw) if path.suffix == ".yaml" else json.loads(raw)
    inventory_path = ROOT / "docs/experiment_parameters_20261014/parameter_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    rows = []

    def emit(record, key, upstream, local, source_id, basis="saved source default"):
        ref = refs[source_id]
        rows.append(dict(family=record["family"], dataset=record["dataset"], tokenizer=record["tokenizer"],
                         method=method_label(record["method"]), parameter=key,
                         upstream_value=json.dumps(upstream, ensure_ascii=False),
                         local_value="<not present>" if local is MISSING else json.dumps(local, ensure_ascii=False),
                         comparison=status(upstream, local), basis=basis, upstream_source=ref["url"],
                         upstream_revision=ref["revision"], local_record=record["id"]))

    def compare(record, source_id, local, prefix=""):
        local_flat = dict(flatten(local))
        for key, value in flatten(snapshots[source_id]):
            emit(record, prefix + key, value, local_flat.get(key, MISSING), source_id)

    for record in inventory["records"]:
        family, q = record["family"], record["parameters"]
        if family == "native_generator":
            compare(record, "grid_generator", q)
        elif family == "native_tokenizer":
            compare(record, "grid_" + record["tokenizer"], q)
        elif family == "native_embedding":
            compare(record, "grid_embedding", q["embedding"])
        elif family == "classic":
            compare(record, "fd_SASRec" if record["method"] == "SASRec" else "fd_gru4rec", q["training"])
            compare(record, "fd_global", q["training"], prefix="generic_training_default.")
        elif family == "fairdiverse_training":
            name = q["rank_model"]
            compare(record, "fd_" + name, q)
            compare(record, "fd_global", q, prefix="generic_training_default.")
        elif family == "reranking" and q["applied_method"] in ("cpfair", "pmmf"):
            name = "CPFair" if q["applied_method"] == "cpfair" else "PMMF"
            local = {**q["applied_parameters"], **{k: q["backend"][k] for k in ("learning_rate", "gamma")}}
            compare(record, "fd_" + name, local)
        elif family == "local_continuation" and q["run"]["mode"] == "sprint":
            settings = q["run"]["sprint_settings"]
            local = {"sprint_attention_weight": settings["attention_weight"], "sprint_ffn_weight": settings["ffn_weight"],
                     "sprint_emb_weight": settings["embedding_weight"], "sprint_attention_temperature": settings["temperature"],
                     "spectral_norm_iters": settings["spectral_norm_iterations"]}
            compare(record, "genrec_sprint_defaults", local)
        elif family == "local_continuation" and q["run"]["mode"] == "rw":
            original = snapshots["ifair_popularity_defaults"]
            local = {"learning_rate": q["run"]["identity"]["learning_rate"], "popularity_groups": 5,
                     "weight_definition": "history group share / target group share"}
            for key in local:
                emit(record, key, original[key], local[key], "ifair_popularity_defaults", "function default / active code definition")
            emit(record, "training_budget", f"{original['num_epochs']} epochs", f"{q['run']['identity']['max_updates']} updates",
                 "ifair_popularity_defaults", "different units; an adaptation, not an equivalent training budget")
            emit(record, "loss_reduction", original["released_loss_reduction"], "mean(per_example_weight * per_example_sid_loss)",
                 "ifair_popularity_defaults", "active released expression versus local adaptation")
    assert len(refs) == 20
    assert len({r["local_record"] for r in rows if r["family"] == "classic"}) == 8
    assert len({r["local_record"] for r in rows if r["family"] == "fairdiverse_training"}) == 84
    with (OUT / "parameter_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest["local_inventory_sha256"] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    manifest["comparison_rows"] = len(rows)
    manifest["interpretation"] = "Raw saved defaults and matching configuration paths; interpolation and runtime overrides are explicit. These are not claims of matched paper reproduction."
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"source_snapshots": len(refs), "parameter_comparisons": len(rows),
                      "local_records_compared": len({r['local_record'] for r in rows})}, indent=2))


if __name__ == "__main__":
    main()
