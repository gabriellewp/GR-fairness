"""Recommendation package exports.

Use lazy attribute loading so importing submodules like
`fairdiverse.recommendation.process_dataset` does not eagerly import trainer
dependencies (e.g., GRID/transformers) during preprocessing-only runs.
"""

__all__ = ["RecTrainer", "RecReRanker"]


def __getattr__(name):
	if name == "RecTrainer":
		from .trainer import RecTrainer
		return RecTrainer
	if name == "RecReRanker":
		from .reranker import RecReRanker
		return RecReRanker
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
