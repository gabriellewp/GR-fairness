"""Load the actual local classes without importing unrelated ranker backends."""

from functools import lru_cache
import importlib
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2] / "fairdiverse" / "recommendation"


@lru_cache(maxsize=2)
def load_class(name):
    if name not in {"CPFair", "PMMF"}:
        raise ValueError("Only CPFair and PMMF are supported by this adapter")
    # The public rerank_model/__init__.py eagerly imports TaxRank, RAIF, LLM,
    # etc. Private package paths let normal relative imports load only the
    # selected module, its real Abstract_Reranker, and its real utils module.
    # No source is copied, rewritten, or monkey-patched.
    prefix = "additional_baselines._legacy_recommendation"
    for package, path in ((prefix, ROOT), (prefix + ".rerank_model", ROOT / "rerank_model")):
        if package not in sys.modules:
            module = ModuleType(package)
            module.__path__ = [str(path)]
            module.__package__ = package
            sys.modules[package] = module
    try:
        module = importlib.import_module(f"{prefix}.rerank_model.{name}")
    except ModuleNotFoundError as exc:
        if exc.name == "cvxpy":
            raise RuntimeError("FairDiverse PMMF requires CVXPY and a QP solver; install additional_baselines/requirements-rerank.txt") from exc
        raise
    return getattr(module, name)
