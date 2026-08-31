"""Print a controller parameter block from a search checkpoint, ready to paste into source.

Tuning output lives under ignored ``artifacts/``; final parameters are baked into
``src/controllers/`` so the shipped controller never reads that directory at
runtime. This tool formats the complete block and never edits source.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path

from controllers.minimum_viable import MINIMUM_VIABLE_PARAMETERS
from controllers.preview_controller import ControllerParameters

DEFAULT_CHECKPOINT = Path("artifacts/controller-search/faster-line/checkpoint.json")


def searched_parameters(checkpoint: Path) -> tuple[dict[str, float], dict[str, float], int]:
    """Return searched and fixed checkpoint values plus its generation."""
    record = json.loads(checkpoint.read_text(encoding="utf-8"))
    vector = record["best_parameter_vector"]
    ordered = {name: float(vector[name]) for name in record["parameter_names"]}
    context = {name: float(value) for name, value in record.get("checkpoint_context", {}).items()}
    return ordered, context, int(record["generation"])


def bake_block(
    *,
    checkpoint: Path,
    base: ControllerParameters,
    preset_base: ControllerParameters,
) -> str:
    """Render the ``replace(...)`` block for every field that differs from ``base``.

    The searched vector is applied to the *preset's* base parameters, not to the
    currently baked ones. A preset may enable behavior outside its search space -
    `faster-line` turns on ``phase_aware_racing_line`` - and the tuned line
    offsets are inert without it, so that flag has to survive the bake.
    """
    searched, fixed, generation = searched_parameters(checkpoint)
    baked = replace(preset_base, **fixed, **searched)
    lines = [
        f"# Searched values from {checkpoint.as_posix()}, generation {generation}.",
        "RACE_FASTER_PARAMETERS = replace(",
        "    MINIMUM_VIABLE_PARAMETERS,",
    ]
    hand_set: list[str] = []
    for field in fields(ControllerParameters):
        value = getattr(baked, field.name)
        if value == getattr(base, field.name):
            continue
        line = f"    {field.name}={value!r},"
        if field.name in searched:
            lines.append(line)
        else:
            hand_set.append(line)
    if hand_set:
        lines.append("    # Carried over; not in this search space.")
        lines.extend(hand_set)
    lines.append(")")
    return "\n".join(lines)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--preset",
        default="faster-line",
        choices=(
            "minimum",
            "faster",
            "faster-line",
            "faster-line-v2",
            "faster-line-v3",
            "faster-line-v4",
            "faster-line-v5",
            "faster-line-v6",
            "faster-line-v7",
            "faster-line-v8",
            "faster-line-v9",
            "faster-line-v10",
            "faster-line-v11",
            "faster-line-v12",
            "faster-line-v13",
            "faster-line-v14",
            "faster-line-v15",
            "faster-line-v16",
            "faster-line-v17",
            "faster-line-v18",
            "faster-line-v19",
            "faster-line-v20",
            "faster-line-v21",
            "faster-line-v22",
            "faster-line-v23",
        ),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    from scripts.controller_training.search import _preset_configuration

    preset_base, _ = _preset_configuration(str(args.preset))
    print(
        bake_block(
            checkpoint=checkpoint,
            base=MINIMUM_VIABLE_PARAMETERS,
            preset_base=preset_base,
        )
    )


if __name__ == "__main__":
    main()
