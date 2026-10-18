# Reranking implementations

CPFair and P-MMF adapters call the maintained FairDiverse implementations on
explicit candidate artifacts. Catalog/group mappings, request order, horizon,
weights, and score conventions are validated before execution. Solver and dense
memory guards remain explicit. Fast P-MMF lookup/exact variants preserve their
backend identity in output metadata. See the parent
[fairness guide](../../docs/fairness.md) for commands and selection policy.
