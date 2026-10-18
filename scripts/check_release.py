"""Validate a source release without displaying credential values."""
import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

SKIP = {".git", ".venv", ".cache", "__pycache__", ".pytest_cache", "artifacts",
        "processed_dataset", "wandb", "logs", "log", "output", "outputs", "build", "dist"}
PATTERNS = {
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "literal service credential": re.compile(r"(?:WANDB_API_KEY|HF_TOKEN|HF_KEY|AWS_SECRET_ACCESS_KEY)\s*[:=]\s*[\"']?[A-Za-z0-9_+/=-]{24,}"),
}


def check(root):
    errors, counts = [], {"python": 0, "json": 0, "shell": 0, "markdown": 0}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (not path.is_file() or relative.parts[0] == "data"
                or relative.parts[:2] == ("GRID", "data")
                or any(part in SKIP or part.endswith(".egg-info") for part in relative.parts)):
            continue
        if path.suffix not in {".py", ".json", ".sh", ".md", ".yaml", ".yml", ".toml", ".txt", ".example", ".csv", ".tex"}:
            continue
        source = path.read_text(encoding="utf-8")
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(source):
                line = source.count("\n", 0, match.start())+1
                errors.append(f"{relative}:{line}: {name} (value suppressed)")
        try:
            if path.suffix == ".py":
                ast.parse(source, filename=str(relative)); counts["python"] += 1
            elif path.suffix == ".json":
                json.loads(source); counts["json"] += 1
            elif path.suffix == ".sh":
                result = subprocess.run(["bash", "-n", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode: errors.append(f"{relative}: invalid shell syntax")
                counts["shell"] += 1
            elif path.suffix == ".md":
                counts["markdown"] += 1
                for target in re.findall(r"\]\(([^)]+)\)", source):
                    if "://" in target or target.startswith(("#", "mailto:")): continue
                    local = target.split("#",1)[0]
                    if local and not (path.parent / local).exists(): errors.append(f"{relative}: broken link {local}")
        except (SyntaxError, ValueError) as exc:
            errors.append(f"{relative}: invalid syntax: {type(exc).__name__}")
    return counts, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    counts, errors = check(args.root.resolve())
    print(json.dumps({"checked": counts, "errors": errors}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
