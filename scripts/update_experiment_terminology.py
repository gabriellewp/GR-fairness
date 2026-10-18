"""Refresh reader-facing terminology while preserving archived IDs and paths."""

import csv
from pathlib import Path
import re
import sys
import tokenize
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_labels import method_label

PHRASES = [
    (r"\bordinary continuations?\b", "standard fine-tuning"),
    (r"\bordinary[- ]control\b", "standard fine-tuning control"),
    (r"\bordinary validation\b", "standard fine-tuning validation"),
    (r"\bordinary[- ]improvement\b", "standard fine-tuning improvement"),
    (r"\bordinary selected\b", "standard fine-tuning selected"),
    (r"\bordinary ordering\b", "original ordering"),
    (r"\bordinary/RR\b", "untreated/RR"),
    (r"\bordinary T5\b", "standard T5"),
    (r"\bordinary GRID\b", "standard GRID"),
    (r"\bordinary beam\b", "standard beam"),
    (r"\bcontinuation modes\b", "fine-tuning modes"),
    (r"\bcontinuation checkpoints\b", "fine-tuning checkpoints"),
    (r"\bcontinuation updates\b", "fine-tuning updates"),
    (r"\bcontinuation report\b", "fine-tuning report"),
    (r"\bcontinuation implementation\b", "fine-tuning implementation"),
    (r"\bcontinuation runner\b", "fine-tuning runner"),
    (r"\blocal continuations?\b", "fairness-aware fine-tuning"),
    (r"\bnative continuations?\b", "fairness-aware fine-tuning"),
    (r"\bcontinuations?\b", "fairness-aware fine-tuning"),
]


def phrases(text):
    for pattern, replacement in PHRASES:
        text = re.sub(pattern, lambda m: replacement[0].upper() + replacement[1:]
                      if m.group()[0].isupper() else replacement, text, flags=re.I)
    return text


def reader_text(text):
    text = text.replace("ordinary/RW/SPRINT", "standard fine-tuning/RW/SPRINT")
    text = text.replace("ordinary/RW", "standard fine-tuning/RW")
    text = text.replace("ordinary/RR", "untreated/RR")
    text = text.replace("L_{\\rm ordinary}", "L_{\\rm standard}")
    protected = []

    def hide(match):
        protected.append(match.group())
        return f"ZZPROTECTED{len(protected)-1}ZZ"

    # Commands, identifiers, link destinations, and artifact paths stay usable.
    text = re.sub(r"```[\s\S]*?```|`[^`\n]+`|\\(?:source|code|input|include|label)\{[^}]*\}|(?<=\]\()[^)]+(?=\))|\S*[/_]\S*", hide, text)
    text = phrases(text)
    text = re.sub(r"\bordinary\b", lambda m: "Standard fine-tuning" if m.group()[0].isupper() else "standard fine-tuning", text, flags=re.I)
    for i, value in enumerate(protected):
        text = text.replace(f"ZZPROTECTED{i}ZZ", value)
    return text


def update_python(path):
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for token in tokenize.generate_tokens(StringIO(text).readline):
        if token.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        value = token.string
        # These strings form the established result/provenance keys.
        if any(value in (repr(x), '"' + x + '"') for x in
               ("GRID: ordinary continuation", "GRID + IFairLRS RW (training)", "GRID + SPRINT (training)")):
            continue
        # Replace phrases in prose, never bare mode names such as "ordinary".
        new = value
        for old, label in (("Ordinary continuation", "Standard fine-tuning"),
                           ("ordinary continuation", "standard fine-tuning"),
                           ("ordinary control", "standard fine-tuning control"),
                           ("ordinary validation", "standard fine-tuning validation"),
                           ("ordinary-improvement", "standard fine-tuning improvement"),
                           ("local GRID continuations", "GRID fairness-aware fine-tuning runs"),
                           ("local ordinary/RW/SPRINT continuation", "standard and fairness-aware fine-tuning"),
                           ("native continuations", "fairness-aware fine-tuning runs"),
                           ("these continuations", "these fine-tuning runs"),
                           ("ordinary/RW/SPRINT", "standard fine-tuning/RW/SPRINT"),
                           ("ordinary/RW", "standard fine-tuning/RW"),
                           ("ordinary peak", "standard fine-tuning peak"),
                           ("over ordinary", "over standard fine-tuning"),
                           ("matched ordinary pilot", "matched standard fine-tuning pilot"),
                           ("Ordinary (epoch pilot)", "Standard fine-tuning (epoch pilot)"),
                           ("Ordinary pilot", "Standard fine-tuning pilot"),
                           ("native continuation results", "fairness-aware fine-tuning results"),
                           ("no continuation test", "no fine-tuning test")):
            new = new.replace(old, label)
        if new != value:
            edits.append((offsets[token.start[0]-1]+token.start[1], offsets[token.end[0]-1]+token.end[1], new))
    for start, end, value in reversed(edits):
        text = text[:start]+value+text[end:]
    return text


def main():
    paths = [p for p in (ROOT / "docs").rglob("*") if p.suffix in (".md", ".tex") and
             "original_source_parameter_comparison_20261014" not in p.parts]
    paths += list((ROOT / "additional_baselines").glob("*.md"))
    changed = []
    for path in paths:
        original = path.read_text()
        updated = reader_text(original)
        if path.name == "comparison_tables.tex":
            for old in ("GRID: ordinary continuation", "GRID + IFairLRS RW (training)", "GRID + SPRINT (training)"):
                updated = updated.replace(old, method_label(old))
        if updated != original:
            path.write_text(updated)
            changed.append(str(path.relative_to(ROOT)))
    generators = [ROOT / "docs" / p for p in ("build_grid_update_october_2026_v1.py", "build_new_methods_deck.py", "grid_update_audit.py", "rq2_comparison_draft_20261011/assemble_tables.py")]
    generators += [ROOT / "scripts/build_tokenizer_reranking_tables.py"]
    for path in generators:
        original = path.read_text()
        updated = update_python(path)
        if updated != original:
            path.write_text(updated)
            changed.append(str(path.relative_to(ROOT)))
    path = ROOT / "docs/rq2_comparison_draft_20261011/comparison_results.csv"
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        row["display_method"] = method_label(row["method"])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Updated reader-facing terminology in", len(changed), "files; retained archived paths and run keys.")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
