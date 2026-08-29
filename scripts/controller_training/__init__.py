"""Deterministic controller evaluation and optimization helpers.

This package is intentionally script-local. It builds on the simulator's
internal race primitives without adding a public runtime API or a controller
dependency on training artifacts.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cem import Candidate, CEMConfig, CEMOptimizer, ParameterSpace, ParameterSpec
    from .evaluator import (
        ParameterizedControllerFactory,
        SoloEvaluator,
        SoloTrialResult,
        controller_factory_from_module,
        run_solo_trial,
    )
    from .genetic import GAConfig, GeneticOptimizer
    from .records import ExperimentJournal, ExperimentRecord
    from .seeds import SeedManifest, generate_seed_manifest, load_seed_manifest, write_seed_manifest

__all__ = [
    "CEMConfig",
    "CEMOptimizer",
    "Candidate",
    "ExperimentJournal",
    "ExperimentRecord",
    "GAConfig",
    "GeneticOptimizer",
    "ParameterSpace",
    "ParameterSpec",
    "ParameterizedControllerFactory",
    "SeedManifest",
    "SoloEvaluator",
    "SoloTrialResult",
    "controller_factory_from_module",
    "generate_seed_manifest",
    "load_seed_manifest",
    "run_solo_trial",
    "write_seed_manifest",
]


def __getattr__(name: str) -> object:
    """Load evaluator exports lazily so its ``python -m`` CLI stays warning-free."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = {
        "CEMConfig": "cem",
        "CEMOptimizer": "cem",
        "Candidate": "cem",
        "ParameterSpace": "cem",
        "ParameterSpec": "cem",
        "GAConfig": "genetic",
        "GeneticOptimizer": "genetic",
        "ExperimentJournal": "records",
        "ExperimentRecord": "records",
        "SeedManifest": "seeds",
        "generate_seed_manifest": "seeds",
        "load_seed_manifest": "seeds",
        "write_seed_manifest": "seeds",
    }.get(name, "evaluator")
    module = import_module(f".{module_name}", __name__)
    return getattr(module, name)
