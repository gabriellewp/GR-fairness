# Development and validation

Use the bundled GRID source directory and the documented source environment:

```bash
python scripts/check_release.py
USE_TF=0 python -m pytest tests additional_baselines/tests -q
python scripts/smoke_test.py --output-dir artifacts/local-smoke
bash scripts/slurm/submit_experiments.sh \
  --config_file configs/reproduction/experiments_config_amazon-beauty_grid_tokenizers.json \
  --max_concurrent 2 --dry-run
```

`check_release.py` validates Python/JSON/shell syntax, scans the current source tree
for token/private-key patterns without printing values, and checks documentation
links. It excludes generated runtime directories. It checks the source checkout,
not historical commits. Behavioral tests cover candidate identity, metrics,
reranking/solver parity, selection restoration, wide beams, batching, padding and
data mapping. Slurm tests use a fake scheduler; validation does not submit jobs.

Before release, validate the package build and its included YAML/JSON assets,
run the local artifact example, and smoke the real GPU workflow on your cluster.
Full model training and cluster jobs are intentionally separate from CPU unit
checks. Record any dependency-specific or hardware-specific limits.

Use new ignored output folders for experiments. The release curation inventory
explains file removals; the research branch preserves the original material.
