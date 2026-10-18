"""Recompute popularity calibration from saved RVQ/RQ-VAE test rankings.

No models are loaded and no settings are selected. Candidate requests are
streamed to avoid loading the full score payload; one saved reranking is read
at a time. Each case writes a small reusable calibration record.
"""
import argparse
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path

import ijson

from build_tokenizer_reranking_tables import ROOT, OUT, COMPARISON, DATASETS, TOKENIZERS

CUTOFFS = (5, 10, 20)


def reference(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest())


def extract(path, prefix):
    with path.open("rb") as stream:
        return next(ijson.items(stream, prefix, use_float=True))


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-11):
        raise ValueError(f"Saved metric mismatch: {actual} != {expected}")


def accumulator():
    return {k: dict(popularity=Counter(), brand=Counter(), hits=0, ndcg=0.) for k in CUTOFFS}


def add(stats, k, ranking, target, popularity, brand):
    selected = ranking[:k]
    assert len(selected) == k and len(set(selected)) == k
    row = stats[k]
    row["popularity"].update(popularity[item] for item in selected)
    row["brand"].update(brand[item] for item in selected)
    try:
        position = selected.index(target)
    except ValueError:
        return
    row["hits"] += 1
    row["ndcg"] += 1 / math.log2(position + 2)


def summarize(stats, history_counts, popularity, brand, requests):
    groups = sorted(set(popularity.values()))
    gh = {g: history_counts[g] / sum(history_counts.values()) for g in groups}
    result = {}
    for k, row in stats.items():
        gp = {g: row["popularity"][g] / (requests * k) for g in groups}
        gu = {g: gp[g] - gh[g] for g in groups}
        counts = sorted(row["brand"][g] for g in set(brand.values()))
        n, total = len(counts), sum(counts)
        assert total == requests * k
        result[str(k)] = dict(
            GH=gh, GP=gp, GU=gu, MGU=sum(abs(v) for v in gu.values()) / len(groups),
            DGU=max(gu.values()) - min(gu.values()), recall=row["hits"] / requests,
            ndcg=row["ndcg"] / requests,
            brand_gini=sum((2*i-n-1)*v for i, v in enumerate(counts, 1)) / (n*total),
            entropy_bits=-sum((v/total)*math.log2(v/total) for v in counts if v))
    return result


def case(task):
    dataset, tokenizer = task
    dest = OUT / "calibration" / f"{dataset}_{tokenizer}.json"
    if dest.exists():
        saved = json.loads(dest.read_text())
        assert saved["comparison"] == reference(COMPARISON)
        print(f"Reusing {dataset}/{tokenizer}", flush=True)
        return str(dest)
    expected = {(r["method"], int(r["k"])): r for r in csv.DictReader(COMPARISON.open())
                if r["dataset"] == dataset and r["tokenizer"] == tokenizer}
    selected_folder = (ROOT / expected["GRID + CPFair", 5]["source"]).parent
    selection_path = selected_folder / "selection.json"
    selection = json.loads(selection_path.read_text())
    candidate_path = Path(selection["test_candidates"]["path"])
    candidate_ref = reference(candidate_path)
    assert candidate_ref == selection["test_candidates"]
    metadata = extract(candidate_path, "metadata")
    assert metadata["seed"] == 42 and metadata["split"]["name"] == "test"
    assert metadata["dataset"] == dataset and metadata["tokenizer"] == tokenizer
    assert reference(Path(metadata["split"]["source"]["path"])) == metadata["split"]["source"]
    maps = extract(candidate_path, "item_groups")
    popularity = {int(i): g for i, g in maps["popularity"].items()}
    brand = {int(i): g for i, g in maps["brand"].items()}
    stats = accumulator()
    requests = {}
    ascending = metadata["score_convention"] == "negative_log_probability"
    with candidate_path.open("rb") as stream:
        for row in ijson.items(stream, "requests.item", use_float=True):
            rid = row["request_id"]
            assert rid not in requests and len(row["targets"]) == 1
            candidates, scores = row["candidate_ids"], row["scores"]
            assert len(candidates) == len(scores) and len(set(candidates)) == len(candidates)
            assert all(math.isfinite(v) for v in scores)
            ranking = [candidates[i] for i in sorted(range(len(scores)), key=scores.__getitem__, reverse=not ascending)]
            target = row["targets"][0]
            requests[rid] = (array("i", candidates), target)
            for k in CUTOFFS:
                add(stats, k, ranking, target, popularity, brand)
    n = len(requests)
    assert n == json.loads((ROOT / "recommendation/processed_dataset" / dataset / "dataset_stats.json").read_text())["user_num"]
    print(f"{dataset}/{tokenizer}: read {n} candidate requests", flush=True)
    history_path = candidate_path.parent / "test_histories.json"
    history_counts, seen = Counter(), set()
    with history_path.open("rb") as stream:
        for rid, history in ijson.kvitems(stream, "", use_float=True):
            assert rid in requests and rid not in seen
            seen.add(rid)
            history_counts.update(popularity[i] for i in history)
    assert len(seen) == n
    del seen
    results = {"control": summarize(stats, history_counts, popularity, brand, n)}
    refs = [reference(selection_path), candidate_ref, reference(history_path)]
    canonical_sources = json.loads((COMPARISON.parent / "provenance.json").read_text())["sources"]
    for method, display in (("rr", "RR"), ("cpfair", "CPFair"), ("pmmf", "P-MMF")):
        report_path = ROOT / expected["GRID + " + display, 5]["source"]
        report_ref = reference(report_path)
        assert report_ref["sha256"] == canonical_sources[str(report_path.relative_to(ROOT))]
        refs.append(report_ref)
        with report_path.open() as stream:
            report = json.load(stream)
        assert report["control_metrics"]["request_count"] == n
        stats = accumulator()
        if method == "rr":
            ranks = report["rankings"]
            assert len(ranks) == n and ranks.keys() == requests.keys()
            digest = hashlib.sha256(json.dumps(sorted(popularity.items()), separators=(",", ":")).encode()).hexdigest()
            assert digest == report["frozen_correction"]["group_mapping_sha256"]
            for rid, ranking in ranks.items():
                candidates, target = requests[rid]
                assert len(set(ranking)) == len(ranking) and set(ranking).issubset(candidates)
                for k in CUTOFFS:
                    add(stats, k, ranking, target, popularity, brand)
        else:
            assert report["input_metadata"] == metadata
            for k in CUTOFFS:
                ranks = report["rankings_by_k"][str(k)]
                assert len(ranks) == n and ranks.keys() == requests.keys()
                for rid, ranking in ranks.items():
                    candidates, target = requests[rid]
                    assert len(ranking) == k and set(ranking).issubset(candidates)
                    add(stats, k, ranking, target, popularity, brand)
        results[method] = summarize(stats, history_counts, popularity, brand, n)
        if method == "rr":
            for field, key in (("control_ifairlrs_fairness", "control"), ("ifairlrs_fairness", "rr")):
                for k in CUTOFFS:
                    for metric in ("MGU", "DGU"):
                        close(results[key][str(k)][metric], report[field][str(k)][metric])
                    for metric in ("GH", "GP", "GU"):
                        for group, value in results[key][str(k)][metric].items():
                            close(value, report[field][str(k)][metric][group])
        print(f"{dataset}/{tokenizer}: verified {display} calibration and rankings", flush=True)
        del report, ranks
    for method, values in results.items():
        label = "GRID: saved reranking reference" if method == "control" else "GRID + " + {"rr": "RR", "cpfair": "CPFair", "pmmf": "P-MMF"}[method]
        for k in CUTOFFS:
            canonical = expected[label, k]
            for metric, column in (("recall", "recall"), ("ndcg", "ndcg"),
                                   ("brand_gini", "brand_gini"), ("entropy_bits", "brand_entropy_bits")):
                close(values[str(k)][metric], float(canonical[column]))
    saved = dict(dataset=dataset, tokenizer=tokenizer, request_count=n, seed=42, comparison=reference(COMPARISON),
                 sources=refs, metadata=metadata, results=results,
                 definition="GH: pooled test-history popularity shares; GP: unweighted top-K popularity exposure; GU=GP-GH; MGU=mean(abs(GU)); DGU=max(GU)-min(GU)",
                 checks=["Candidate and saved-report fingerprints", "Full request counts and exact ranking request IDs",
                         "Unique eligible ranked items and full lists", "RR MGU/DGU and group shares reproduced",
                         "All Recall/NDCG/brand Gini/entropy values reproduced against canonical comparison"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as stream:
        json.dump(saved, stream, indent=2)
        stream.write("\n")
    return str(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = [(dataset, tok) for tok in TOKENIZERS for dataset in DATASETS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(case, tasks):
            print("Saved", result, flush=True)
