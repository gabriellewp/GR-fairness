# Clean source snapshot

This repository contains the current generative-fairness implementation and its
portable execution workflows. It includes GRID source, classic recommenders,
CPFair/P-MMF reranking, IFairLRS RR/RW, SPRINT, validation-only continuation tuning,
batched generation parity checks, deadline pilots and paired activity analyses.

Both Python packages and their configuration assets are installable from this
source checkout. GRID is an ordinary directory, so the same code is available in
a Git clone and an anonymous repository ZIP. Its upstream license and all third-party
notice content are preserved; the notices are split into ordered parts below 8 MB.

Raw/processed datasets, embeddings, checkpoints, candidate exports, logs, per-user
results and dated research reports are generated or obtained separately. Exact
historical configuration recipes are retained under `configs/reproduction/`;
replace artifact paths with your own matching files before execution. New-run
configuration generation is documented in `experiments.md`.

Slurm launchers use explicit roots/interpreters and inherit authentication from
the environment. No launcher installs packages or activates a personal environment.
This snapshot has independent Git history and includes no imported research commits.

Validation results and hardware limits are recorded in `release-validation.json`.
Full GPU training, real Slurm scheduling, online services and dataset/model downloads
must be verified in the execution environment; CPU source checks do not certify those.
