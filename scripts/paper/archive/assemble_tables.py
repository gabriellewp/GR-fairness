"""Build four comparison tables with explicit placeholders for unaudited cells."""
import csv
import hashlib
import json
import math
from pathlib import Path
import yaml
from scripts.experiment_labels import method_label

from scripts.paper.archive_paths import source_root, output_root
ROOT = source_root()
OUTPUT = output_root()
OUT = OUTPUT
DATASETS = {"amazon-beauty": "Beauty", "amazon-sports": "Sports", "amazon-toys": "Toys", "amazon-pet-supplies": "Pet Supplies"}
TOKENS = {"rkmeans": "RKMeans", "rvq": "RVQ", "rqvae": "RQ-VAE"}
METHODS = ["APR", "FairDual", "IPS", "Minmax-SGD", "SDRO", "FOCF", "Reg"]
LOCAL_METHODS = {"rw": "GRID + IFairLRS RW (training)", "sprint": "GRID + SPRINT (training)"}
SOURCES = {}
DATA = {}
REPORTED = {}
RECORDED_CLASSICS = {}
MISSING_REASONS = {}
LOCAL_BRANCHES = {}
RERANK_BEAMS = {}
FAIRNESS_COMPLETIONS = []


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def count_metrics(counts):
    values = sorted(counts.values())
    total, n = sum(values), len(values)
    return (sum((2*i-n-1)*v for i, v in enumerate(values, 1))/(n*total),
            -sum(v/total*math.log2(v/total) for v in values if v))


def add(ds, tok, method, metrics, source):
    assert len(metrics) == 3
    DATA[ds, tok, method] = (metrics, str(source.relative_to(ROOT)))


classic = {}
classic_paths = sorted((ROOT / "output/rq2_classic_evaluation/356042").glob("case_*/test_full/results.json"))
classic_paths += sorted((ROOT / "output/rq2_classic_evaluation").glob("pet_sasrec_*/test_full/results.json"))
for p in classic_paths:
    d = read(p)
    assert d["full_split"] and d["split"] == "test"
    ds, model = d["case"]["dataset"], d["case"]["model"]
    classic.setdefault(ds, d)
    vals = []
    for k in (5, 10, 20):
        m = d["metrics"]
        vals.append((m[f"recall@{k}"], m[f"ndcg@{k}"], m[f"gini@{k}"], m[f"entropy_bits@{k}"]))
    add(ds, "", model, vals, p)

# The completed matched Pet SASRec run has a full saved evaluation, but its
# common-protocol smoke evaluation failed. Keep the recorded scores explicitly
# marked, and let a completed common-protocol result take precedence later.
pet_key = ("amazon-pet-supplies", "", "SASRec")
pet_inventory = ROOT / "output/rq2_classic_evaluation/pet_sasrec_358498/audit/inventory.json"
if pet_key not in DATA and pet_inventory.exists():
    from additional_baselines.candidates import file_reference

    inventory = read(pet_inventory)
    case = next(c for c in inventory["cases"]
                if (c["dataset"], "", c["model"]) == pet_key)
    ref = classic[case["dataset"]]
    for pinned in [case["checkpoint"], case["config_reference"], case["saved_result"],
                   *case["inputs"].values()]:
        assert file_reference(pinned["path"]) == pinned, f"Pinned input changed: {pinned['path']}"
        SOURCES[str(Path(pinned["path"]).relative_to(ROOT))] = pinned["sha256"]
    config_path = Path(case["config_reference"]["path"])
    c = yaml.safe_load(config_path.read_text())
    assert c == case["config"]
    assert c["seed"] == 42 and c["history_length"] == 40 and c["padding_item_id"] == -1
    assert not c["fair-rank"] and c["topk"] == [5, 10, 20] and c["fairness_type"] == "Exposure"
    assert c["sample_num"] == c["item_num"] and c["user_num"] == ref["request_count"]
    assert c["group_num"] == ref["metrics"]["group_count@5"]
    assert case["inputs"]["test"] == ref["case"]["inputs"]["test"]
    assert case["inputs"]["groups"] == ref["case"]["inputs"]["groups"]
    result_path = Path(case["saved_result"]["path"])
    m = read(result_path)
    summary_path = result_path.parent / "user_fairness/user_fairness_summary.json"
    summary = read(summary_path)
    assert summary["request_count"] == summary["user_count"] == ref["request_count"]
    assert summary["mapped_users_without_evaluation"] == 0
    assert summary["activity_mapping"] == case["inputs"]["activity"]
    assert summary["evaluation"]["split"] == "test" and summary["evaluation"]["model"] == "SASRec"
    vals = []
    for k in (5, 10, 20):
        assert round(summary["statistics"]["overall"][f"recall@{k}"], c["decimals"]) == m[f"hr@{k}"]
        assert round(summary["statistics"]["overall"][f"ndcg@{k}"], c["decimals"]) == m[f"ndcg@{k}"]
        assert 0 <= m[f"ndcg@{k}"] <= m[f"hr@{k}"] <= 1
        assert 0 <= m[f"gini@{k}"] <= 1 and 0 <= m[f"entropy@{k}"] <= math.log2(c["group_num"])
        vals.append((m[f"hr@{k}"], m[f"ndcg@{k}"], m[f"gini@{k}"], m[f"entropy@{k}"]))
    add(*pet_key, vals, result_path)
    RECORDED_CLASSICS[pet_key] = dict(
        config=str(config_path.relative_to(ROOT)), source=str(result_path.relative_to(ROOT)),
        audit=str(pet_inventory.relative_to(ROOT)), summary=str(summary_path.relative_to(ROOT)),
        checkpoint=case["checkpoint"], request_count=summary["request_count"],
        provenance="Pinned saved result/config/checkpoint and inputs checked; full-test user summary matches rounded Recall/NDCG",
        note="Recorded FairDiverse evaluation in bits; common-protocol evaluation failed its batched-versus-single-user score check. Legacy sigmoid/tie handling differs from raw-logit ranking.")

fresh_paths = list((ROOT / "output/rq1_grid_agreement/356038").glob("case_*/test_full/results.json"))
fresh_paths += list((ROOT / "output/rq1_grid_agreement/356026").glob("case_*/test_full/results.json"))
# Pet RKMeans is scheduled separately; prefer its latest full evaluation.
fresh_paths += sorted((ROOT / "output/rq1_grid_agreement").glob("*/case_9/test_full/results.json"),
                      key=lambda p: (p.stat().st_mtime_ns, str(p)))
for p in fresh_paths:
    d = read(p)
    assert d["full_split"]
    ds, tok = d["case"]["dataset"], d["case"]["tokenizer"]
    ref = classic[ds]
    assert d["input_metadata"]["split"]["source"] == ref["case"]["inputs"]["test"]
    assert d["metrics"]["request_count"] == ref["request_count"]
    expected_groups = set(json.loads(Path(ref["case"]["inputs"]["groups"]["path"]).read_text()).values())
    expected_groups = {str(g) for g in expected_groups}
    vals = []
    for k in (5, 10, 20):
        m = d["metrics"]["at_k"][str(k)]
        b = m["groups"]["brand"]
        assert set(b["exposure_counts"]) == expected_groups
        gini, entropy = count_metrics(b["exposure_counts"])
        assert math.isclose(gini, b["gini"], abs_tol=1e-12)
        assert math.isclose(entropy*math.log(2), b["entropy"], abs_tol=1e-12)
        vals.append((m["recall"], m["ndcg"], gini, entropy))
    add(ds, tok, "GRID: original (fresh evaluation)", vals, p)

# Populate completed fairness-training runs, keeping recorded evaluations
# distinguishable from the newly harmonized reevaluations above.
tsv_path = ROOT / "wandb_current_baseline_results.tsv"
raw = tsv_path.read_bytes()
SOURCES[str(tsv_path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
wandb_rows = list(csv.DictReader(raw.decode().splitlines(), delimiter="\t"))
wandb_index = {}
for row in wandb_rows:
    if (row["model"] == "GRID" and row["embedding_model"] == "google/flan-t5-xl"
            and row["metadata_text_mode"] == "complete_metadata" and row["rank_model"]):
        key = row["dataset"], row["grid_tokenizer"], row["rank_model"]
        assert key not in wandb_index, f"Ambiguous W&B training run: {key}"
        wandb_index[key] = row

matched_wandb = set()
excluded_training_runs = []
for config_path in sorted((ROOT / "fairdiverse/recommendation/log").glob("*/config.yaml")):
    result_path = config_path.with_name("test_result.json")
    if not result_path.exists():
        continue
    config_raw = config_path.read_bytes()
    c = yaml.safe_load(config_raw)
    if not (c.get("model") == "GRID" and c.get("metadata_text_mode") == "complete_metadata"
            and c.get("embedding_model") == "google/flan-t5-xl" and c.get("fair-rank")):
        continue
    ds, tok, ranker = c["dataset"], c["grid_tokenizer"], c["rank_model"]
    method = "Minmax-SGD" if ranker == "Minmax_SGD" else ranker
    if ds not in DATASETS or tok not in TOKENS or method not in METHODS:
        continue
    assert c["seed"] == 42 and c["fairness_type"] == "Exposure" and not c["grid_native_metrics"]
    assert c["user_num"] == classic[ds]["request_count"]
    assert c["group_num"] == classic[ds]["metrics"]["group_count@5"]
    assert c["topk"] == [5, 10, 20] and c["grid_data_source"] == "embedded_tfrecords"
    name = "GRID + " + method + " (training)"
    # Preserve the table's W&B-matched recorded training cohort. Additional
    # runs with different generation settings must not replace these rows.
    wandb = wandb_index.get((ds, tok, ranker))
    if wandb and c["log_name"] != wandb["run_name"]:
        excluded_training_runs.append(dict(
            source=str(result_path.relative_to(ROOT)),
            reason="Additional training run; table retains the W&B-matched recorded cohort",
            retained_run=wandb["run_name"]))
        continue
    assert (ds, tok, name) not in DATA, f"Multiple saved training evaluations: {ds, tok, method}"
    m = read(result_path)
    wandb = wandb_index.get((ds, tok, ranker))
    if wandb:
        assert wandb["state"] == "finished"
        assert c["log_name"] == wandb["run_name"]
        for k in (5, 10, 20):
            for metric in ("hr", "ndcg", "gini", "entropy"):
                assert float(wandb[f"test/{metric}@{k}"]) == m[f"{metric}@{k}"]
        matched_wandb.add((ds, tok, ranker))
    vals = []
    for k in (5, 10, 20):
        assert 0 <= m[f"ndcg@{k}"] <= m[f"hr@{k}"] <= 1
        assert 0 <= m[f"gini@{k}"] <= 1
        vals.append((m[f"hr@{k}"], m[f"ndcg@{k}"], m[f"gini@{k}"], m[f"entropy@{k}"]))
    SOURCES[str(config_path.relative_to(ROOT))] = hashlib.sha256(config_raw).hexdigest()
    add(ds, tok, name, vals, result_path)
    REPORTED[ds, tok, name] = dict(config=str(config_path.relative_to(ROOT)),
        source=str(result_path.relative_to(ROOT)), wandb_url=wandb["url"] if wandb else "",
        provenance="TSV metrics exactly matched to saved test result" if wandb else "Saved test result/config; absent from supplied TSV snapshot",
        note="Recorded FairDiverse evaluation, rounded to four decimals; not a fresh common-protocol reevaluation. HR equals Recall under single-target leave-one-out evaluation.")
assert matched_wandb == set(wandb_index), "Some matching W&B runs were not found on disk"

# Recover validation-selected standard fine-tuning controls from their existing artifacts.
# These use the local continuation evaluation, independently of the W&B TSV.
from additional_baselines.prepare_rerank_candidates import TOKENIZER_SUITES, SUITES

ordinary_sources = dict(TOKENIZER_SUITES)
ordinary_sources.update({(ds, "rkmeans"): suite for ds, suite in SUITES.items() if suite})
ordinary_sources["amazon-beauty", "rkmeans"] = "beauty_native_control_lr1e3"
ordinary_verified = []
for (ds, tok), suite in ordinary_sources.items():
    folder = ROOT / "additional_baselines/artifacts" / suite
    if suite != "beauty_native_control_lr1e3":
        folder /= "ordinary"
    p = folder / "test_results.json"
    if not p.exists():
        continue
    d = read(p)
    continuation = read(folder / "continuation.json")
    trace = read(folder / "validation_trace.json")
    metadata = d["input_metadata"]
    ref = classic[ds]
    assert metadata["seed"] == 42 and d["mode"] == "ordinary"
    assert metadata["split"]["name"] == "test"
    assert metadata["split"]["source"] == ref["case"]["inputs"]["test"]
    assert d["metrics"]["request_count"] == ref["request_count"]
    selected = max(trace, key=lambda row: (row["metrics"]["at_k"]["5"]["recall"], -row["update"]))
    assert d["selected_update"] == continuation["selected_update"] == selected["update"]
    assert metadata["checkpoint"] == continuation["selected_checkpoint"]
    # Historical update-zero exports require separate verification.
    assert d["selected_update"] > 0
    checkpoint = Path(metadata["checkpoint"]["path"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metadata["checkpoint"]["sha256"]
    assert tok in Path(metadata["sid_mapping"]["path"]).parts
    expected_groups = {str(g) for g in json.loads(Path(ref["case"]["inputs"]["groups"]["path"]).read_text()).values()}
    vals = []
    for k in (5, 10, 20):
        m = d["metrics"]["at_k"][str(k)]
        b = m["groups"]["brand"]
        assert set(b["exposure_counts"]) == expected_groups
        gini, entropy = count_metrics(b["exposure_counts"])
        assert math.isclose(gini, b["gini"], abs_tol=1e-12)
        assert math.isclose(entropy * math.log(2), b["entropy"], abs_tol=1e-12)
        vals.append((m["recall"], m["ndcg"], gini, entropy))
    add(ds, tok, "GRID: ordinary continuation", vals, p)
    ordinary_verified.append(dict(dataset=ds, tokenizer=tok, source=str(p.relative_to(ROOT)),
                                  selected_update=d["selected_update"], checkpoint=metadata["checkpoint"]))

p = ROOT / "docs/rkmeans_reranking_analysis_20261011/test_results.csv"
raw = p.read_bytes()
SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
saved = list(csv.DictReader(raw.decode().splitlines()))
for ds, label in DATASETS.items():
    for method in ("Untreated", "RR", "CPFair", "P-MMF"):
        group = sorted((r for r in saved if r["dataset"] == label and r["method"] == method), key=lambda r:int(r["k"]))
        if not group:
            continue
        assert [int(r["k"]) for r in group] == [5, 10, 20]
        assert all(int(r["requests"]) == classic[ds]["request_count"] for r in group)
        vals = [(float(r["recall"]), float(r["ndcg"]), float(r["brand_exposure_gini"]),
                 float(r["brand_exposure_entropy"])/math.log(2)) for r in group]
        name = "GRID: saved reranking reference" if method == "Untreated" else "GRID + " + method
        add(ds, "rkmeans", name, vals, p)

# New gap-filling pipelines and the pending Pet RKMeans tests are imported
# directly when complete; their local results are not in the W&B TSV.
def local_values(metrics, metadata, ds, tok):
    ref = classic[ds]
    assert metadata["seed"] == 42 and tok in Path(metadata["sid_mapping"]["path"]).parts
    assert metadata["split"] == {"name": "test", "source": ref["case"]["inputs"]["test"]}
    assert metrics["request_count"] == ref["request_count"]
    expected = {str(g) for g in json.loads(Path(ref["case"]["inputs"]["groups"]["path"]).read_text()).values()}
    vals = []
    for k in (5, 10, 20):
        m = metrics["at_k"][str(k)]
        b = m["groups"]["brand"]
        assert set(b["exposure_counts"]) == expected
        gini, entropy = count_metrics(b["exposure_counts"])
        assert math.isclose(gini, b["gini"], abs_tol=1e-12)
        assert math.isclose(entropy * math.log(2), b["entropy"], abs_tol=1e-12)
        vals.append((m["recall"], m["ndcg"], gini, entropy))
    return vals


# Include the matched local continuation methods alongside their ordinary
# controls. Historical Pet RVQ update-zero output evaluated the wrong weights;
# use the completed pinned-source evaluation for that selected checkpoint.
for (ds, tok), suite in ordinary_sources.items():
    control_folder = ROOT / "additional_baselines/artifacts" / suite
    if suite != "beauty_native_control_lr1e3":
        control_folder /= "ordinary"
    control = read(control_folder / "continuation.json")
    completion_path = control_folder.parent / "fairness_completion.json"
    completion = read(completion_path) if completion_path.exists() else None
    if completion:
        assert completion["dataset"] == ds and completion["tokenizer"] == tok
        assert completion["control_improvement_required"] is False
        FAIRNESS_COMPLETIONS.append(completion)
    for mode, method in LOCAL_METHODS.items():
        folder = (ROOT / "additional_baselines/artifacts" / f"beauty_native_{mode}_lr1e3"
                  if ds == "amazon-beauty" and tok == "rkmeans" else control_folder.parent / mode)
        p = folder / "test_results.json"
        key = ds, tok, method
        if not p.exists():
            if completion and mode in completion["jobs"]:
                MISSING_REASONS[key] = "pending: matched branch without standard fine-tuning improvement prerequisite"
            elif control["selected_update"] == 0:
                MISSING_REASONS[key] = "not run: ordinary-control validation gate"
            continue
        continuation = read(folder / "continuation.json")
        trace = read(folder / "validation_trace.json")
        run_config = read(folder / "run_config.json")
        assert run_config["mode"] == continuation["mode"] == mode
        assert continuation["identity"] == control["identity"]
        selected = max(trace, key=lambda r: (r["metrics"]["at_k"]["5"]["recall"], -r["update"]))
        assert selected["update"] == continuation["selected_update"]
        selected_original = continuation["selected_update"] == 0
        corrected = selected_original and not continuation.get("selected_weights_restored", False)
        if corrected:
            source = ROOT / DATA[ds, tok, "GRID: original (fresh evaluation)"][1]
            d = read(source)
            assert d["full_split"] and d["common_rank_metric_formula_agreement"]
        else:
            source = p
            d = read(p)
            assert d["mode"] == mode and d["selected_update"] == continuation["selected_update"]
        metadata = d["input_metadata"]
        assert metadata["checkpoint"] == continuation["selected_checkpoint"]
        assert hashlib.sha256(Path(metadata["checkpoint"]["path"]).read_bytes()).hexdigest() == metadata["checkpoint"]["sha256"]
        add(ds, tok, method, local_values(d["metrics"], metadata, ds, tok), source)
        LOCAL_BRANCHES[key] = dict(dataset=ds, tokenizer=tok, method=method,
            selected_update=continuation["selected_update"], source=str(source.relative_to(ROOT)),
            continuation=str((folder / "continuation.json").relative_to(ROOT)),
            checkpoint=metadata["checkpoint"], corrected_update_zero=corrected,
            selected_update_zero=selected_original,
            control_improvement_required=run_config.get("control_improvement_required", True),
            group=run_config["group"], ordinary_control=str((control_folder / "continuation.json").relative_to(ROOT)),
            note="RW is the saved IFairLRS adaptation to GRID; branch/control share training plan and source checkpoint.")

artifacts = ROOT / "additional_baselines/artifacts"
selected_folders = sorted(artifacts.glob("*_rerank_candidates_*_selected_tests"))
selected_folders += sorted(artifacts.glob("pet-supplies_rkmeans_validation_selected_test_*"))
for folder in selected_folders:
    if not all((folder / f"{m}_test_results.json").exists() for m in ("rr", "cpfair", "pmmf")):
        continue
    selection = read(folder / "selection.json")
    ds = selection["dataset"]
    cp = read(folder / "cpfair_test_results.json")
    metadata = cp["input_metadata"]
    tok = metadata["tokenizer"]
    assert tok in TOKENS and ds in DATASETS
    RERANK_BEAMS[ds, tok] = metadata["generation"]["beam_width"]
    add(ds, tok, "GRID: saved reranking reference", local_values(cp["control_metrics"], metadata, ds, tok), folder / "cpfair_test_results.json")
    for method, label in (("rr", "RR"), ("cpfair", "CPFair"), ("pmmf", "P-MMF")):
        p = folder / f"{method}_test_results.json"
        d = cp if method == "cpfair" else read(p)
        assert d["control_metrics"] == cp["control_metrics"]
        if method == "rr":
            vm = d["frozen_correction"]["validation_metadata"]
            for key in ("checkpoint", "sid_mapping", "seed", "candidate_membership", "score_convention"):
                assert vm[key] == metadata[key]
            assert d["frozen_correction"]["alpha"] == selection["rr_alpha"]
            assert d["frozen_correction"]["epsilon"] == selection["rr_floor"]
        else:
            assert d["input_metadata"] == metadata
            assert d["fairness_lambda"] == selection[method + "_lambda"]
        metrics = dict(d["metrics"])
        if "request_count" not in metrics:
            # CPFair/P-MMF save per-cutoff metrics and the explicit rankings;
            # their request count is recorded only in the control report.
            rankings = d["rankings_by_k"]
            assert set(rankings) == {"5", "10", "20"}
            count = cp["control_metrics"]["request_count"]
            assert all(len(ranked) == count for ranked in rankings.values())
            assert all(ranked.keys() == rankings["5"].keys() for ranked in rankings.values())
            metrics["request_count"] = count
        add(ds, tok, "GRID + " + label, local_values(metrics, metadata, ds, tok), p)

for p in sorted(artifacts.glob("*_rerank_candidates_*/source_test_results.json")):
    if not p.with_name("complete.json").exists():
        continue
    d = read(p)
    ds, tok = d["dataset"], d["tokenizer"]
    assert d["full_split"]
    vals = local_values(d["metrics"], d["input_metadata"], ds, tok)
    if (ds, tok, "GRID: original (fresh evaluation)") not in DATA:
        add(ds, tok, "GRID: original (fresh evaluation)", vals, p)
    # Candidate preparation records the full-test untreated scores even when
    # the downstream selected reranker tests have not completed.
    if (ds, tok, "GRID: saved reranking reference") not in DATA:
        add(ds, tok, "GRID: saved reranking reference", vals, p)
        RERANK_BEAMS[ds, tok] = d["input_metadata"]["generation"]["beam_width"]
    if ds == "amazon-pet-supplies" and tok == "rqvae":
        folder = artifacts / TOKENIZER_SUITES[ds, tok] / "ordinary"
        continuation = read(folder / "continuation.json")
        trace = read(folder / "validation_trace.json")
        selected = max(trace, key=lambda r: (r["metrics"]["at_k"]["5"]["recall"], -r["update"]))
        assert selected["update"] == continuation["selected_update"] == 0
        assert continuation["selected_checkpoint"] == d["input_metadata"]["checkpoint"]
        add(ds, tok, "GRID: ordinary continuation", vals, p)
        ordinary_verified.append(dict(dataset=ds, tokenizer=tok, selected_update=0,
                                      source=str(p.relative_to(ROOT)), note="Validation selected original checkpoint; no retraining needed"))

rows, tex = [], [r"% Requires booktabs and graphicx. Entropy is in bits throughout."]
recorded_evaluations = {**REPORTED, **RECORDED_CLASSICS}
for ds, label in DATASETS.items():
    tex += [r"\begin{table*}[p]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{"+label+r": comparison of recommendation effectiveness and brand exposure. Each metric cell lists $K=5/10/20$. Entropy is in bits; seed 42. The original GRID row uses fresh evaluation; the saved reference matches the candidates used for reranking. All training variants are evaluated without reranking unless RR, CPFair, or P-MMF is named. Historical IFairLRS RW and SPRINT rows select checkpoints by full-validation Recall@5. Tables~\ref{tab:joint-fairness-outcomes} and~\ref{tab:joint-fairness-tests} report the later joint-fairness-selected experiment and matched ordinary controls. $\dagger$ denotes recorded FairDiverse training-run evaluations, checked against saved configurations/results and W\&B where available; these have not been freshly reevaluated under the common protocol. Their HR equals Recall for single-target evaluation. A dash indicates a result not yet verified for this comparison. " +
        ((r"Pet SASRec uses its completed recorded full-test evaluation; its common-protocol reevaluation failed the score-consistency check. "
          if pet_key in RECORDED_CLASSICS else "") +
         r"Local branches marked $^{0}$ selected the original checkpoint at update zero. Pet RVQ SPRINT uses its corrected original-checkpoint evaluation. " +
         ("Pet RQ-VAE RW and SPRINT use the matched control without requiring standard fine-tuning validation improvement; their settings and checkpoint selection remain unchanged. "
          if any(c["dataset"] == ds and c["tokenizer"] == "rqvae" for c in FAIRNESS_COMPLETIONS)
          else "Pet RQ-VAE RW and SPRINT were not run because the standard fine-tuning control failed the validation-improvement gate. ")
         if ds == "amazon-pet-supplies" else "") +
        (f"Sports RVQ saved reranking reference and reranker rows use beam width {RERANK_BEAMS[ds, 'rvq']}; original and training rows retain their recorded generation settings. "
         if ds == "amazon-sports" and RERANK_BEAMS.get((ds, "rvq"), 100) != 100 else "") + "}",
        r"\label{tab:comparison-"+ds.removeprefix("amazon-")+"}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcccc}", r"\toprule",
        r"Model / variant & Tokenizer & Recall $\uparrow$ & NDCG $\uparrow$ & Brand Gini $\downarrow$ & Brand entropy $\uparrow$ \\", r"\midrule"]
    blocks = [("", ["SASRec", "GRU4Rec"])]
    blocks += [(tok, ["GRID: original (fresh evaluation)", "GRID: ordinary continuation"]+list(LOCAL_METHODS.values())+
                ["GRID + " + m + " (training)" for m in METHODS]+
                ["GRID: saved reranking reference", "GRID + RR", "GRID + CPFair", "GRID + P-MMF"]) for tok in TOKENS]
    for tok, methods in blocks:
        for method in methods:
            entry = DATA.get((ds, tok, method))
            cells = ["--"]*4
            if entry:
                metrics, source = entry
                cells = [" / ".join(f"{v[i]:.4f}" for v in metrics) for i in range(4)]
                for k, values in zip((5, 10, 20), metrics):
                    rows.append(dict(dataset=ds, tokenizer=tok, method=method, display_method=method_label(method), k=k, recall=values[0], ndcg=values[1],
                        brand_gini=values[2], brand_entropy_bits=values[3],
                        status="recorded FairDiverse evaluation; config/source checked" if (ds, tok, method) in recorded_evaluations else
                               "verified; corrected validation-selected update-zero source" if LOCAL_BRANCHES.get((ds, tok, method), {}).get("corrected_update_zero") else
                               "verified; validation selected original checkpoint" if LOCAL_BRANCHES.get((ds, tok, method), {}).get("selected_update_zero") else "verified",
                        source=source, wandb_url=recorded_evaluations.get((ds, tok, method), {}).get("wandb_url", "")))
            else:
                for k in (5, 10, 20):
                    rows.append(dict(dataset=ds, tokenizer=tok, method=method, display_method=method_label(method), k=k, recall="", ndcg="",
                        brand_gini="", brand_entropy_bits="", status=MISSING_REASONS.get((ds, tok, method), "not verified for this comparison"), source="", wandb_url=""))
            label_method = method_label(method)
            if (ds, tok, method) in recorded_evaluations:
                label_method += r"$^{\dagger}$"
            if LOCAL_BRANCHES.get((ds, tok, method), {}).get("selected_update_zero"):
                label_method += r"$^{0}$"
            tex.append(" & ".join([label_method, TOKENS.get(tok, "--")]+cells)+r" \\")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex += [r"\end{tabular}%", "}", r"\end{table*}", ""]
from scripts.build_grid_baseline_comparison import write_baseline_comparison
tex[1:1] = [write_baseline_comparison(rows)]
from scripts.build_joint_fairness_comparison import write_joint_fairness_comparison
tex += [write_joint_fairness_comparison(OUT)]
(OUT / "comparison_tables.tex").write_text("\n".join(tex))
with (OUT / "comparison_results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
(OUT / "provenance.json").write_text(json.dumps(dict(sources=SOURCES,
    entropy_unit="bits; converted natural-log GRID entropy by dividing by ln(2)",
    note="Seven-method completed training runs are included as recorded evaluations, with explicit protocol/rounding caveats. Saved standard fine-tuning controls have matching test fingerprints, request counts, brand groups and validation-selected checkpoint hashes. A placeholder does not establish that an experiment is missing.",
    ordinary_continuations=ordinary_verified,
    local_training_branches=list(LOCAL_BRANCHES.values()),
    fairness_completions=FAIRNESS_COMPLETIONS,
    gated_branches=[dict(dataset=ds, tokenizer=tok, method=method, reason=reason) for (ds,tok,method),reason in MISSING_REASONS.items() if reason.startswith("not run:")],
    pending_branches=[dict(dataset=ds, tokenizer=tok, method=method, reason=reason) for (ds,tok,method),reason in MISSING_REASONS.items() if reason.startswith("pending:")],
    excluded_results=[dict(source="additional_baselines/artifacts/pet_rvq_complete_metadata_continuation_20261004/sprint/test_results.json",
        reason="Historical update-zero report used final trained weights; replaced with pinned-source full-test evaluation")],
    excluded_training_runs=excluded_training_runs,
    training_runs=[dict(dataset=ds, tokenizer=tok, method=method, **record) for (ds,tok,method),record in REPORTED.items()],
    recorded_classic_runs=[dict(dataset=ds, tokenizer=tok, method=method, **record) for (ds,tok,method),record in RECORDED_CLASSICS.items()],
    wandb_training_runs_matched=len(matched_wandb), recorded_training_combinations=len(REPORTED),
    verification="Fresh GRID and classic full-test request counts, test-split fingerprints and GRID catalog brand-group IDs agree. Fresh GRID Gini/entropy recomputed from exposure counts.",
    caveats=["One seed", "Fresh and saved original GRID evaluations are distinct evaluation artifacts, not different trained models",
        "Beauty saved candidates differ from fresh generation for three requests", "Classics rank the full catalog; GRID results depend on SID generation coverage",
        "Raw entropy across datasets has different group-count ceilings",
        "Recorded FairDiverse training metrics use the original evaluation path and four-decimal rounding; exact historical split membership, float/tie handling, and returned candidates have not been freshly reconciled"]), indent=2)+"\n")
print(f"Saved four comparison tables: {sum(r['recall']!='' for r in rows)} populated cutoff rows; {sum(r['recall']=='' for r in rows)} placeholder cutoff rows.")
print(f"Included {len(REPORTED)} seven-method training combinations; {len(matched_wandb)} exactly match supplied W&B records.")
print(f"Included {len(LOCAL_BRANCHES)} verified RW/SPRINT combinations; {len(MISSING_REASONS)} unavailable branches.")
