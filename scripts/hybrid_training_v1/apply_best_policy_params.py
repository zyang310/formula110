#!/usr/bin/env python3
"""Apply tuner best_params JSON to controllers.hybrid_track_policy_data.DEFAULT_GAINS."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "artifacts" / "policies" / "first_tune.json"
DEFAULT_TARGET = PROJECT_ROOT / "src" / "controllers" / "hybrid_track_policy_data.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"tuner JSON path (default: {DEFAULT_SOURCE.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"policy data Python file (default: {DEFAULT_TARGET.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the replacement block without editing")
    return parser.parse_args()


def apply_best_params(source: Path, target: Path, *, dry_run: bool = False) -> dict[str, float]:
    best_params = read_best_params(source)
    target_text = target.read_text(encoding="utf-8")
    existing_gains = read_default_gains(target_text)
    unknown = sorted(set(best_params) - set(existing_gains))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"best_params includes keys not present in DEFAULT_GAINS: {names}")
    merged = {name: best_params.get(name, value) for name, value in existing_gains.items()}
    replacement = render_default_gains(merged)
    updated_text = replace_default_gains_block(target_text, replacement)
    if dry_run:
        print(replacement)
    else:
        target.write_text(updated_text, encoding="utf-8")
    return {name: float(best_params[name]) for name in best_params}


def read_best_params(source: Path) -> dict[str, float]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    best_params = payload.get("best_params") if isinstance(payload, dict) else None
    if not isinstance(best_params, dict):
        raise ValueError(f"{source} does not contain an object field named best_params")
    parsed: dict[str, float] = {}
    for name, value in best_params.items():
        if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("best_params must map string names to numeric values")
        parsed[name] = float(value)
    if not parsed:
        raise ValueError("best_params is empty")
    return parsed


def read_default_gains(source_text: str) -> dict[str, float]:
    tree = ast.parse(source_text)
    for node in tree.body:
        target_name = _assignment_name(node)
        if target_name != "DEFAULT_GAINS":
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, dict):
            raise ValueError("DEFAULT_GAINS is not a literal dictionary")
        gains: dict[str, float] = {}
        for key, raw_value in value.items():
            if not isinstance(key, str) or isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
                raise ValueError("DEFAULT_GAINS must map string names to numeric values")
            gains[key] = float(raw_value)
        return gains
    raise ValueError("DEFAULT_GAINS assignment not found")


def replace_default_gains_block(source_text: str, replacement: str) -> str:
    tree = ast.parse(source_text)
    for node in tree.body:
        if _assignment_name(node) == "DEFAULT_GAINS":
            lines = source_text.splitlines()
            start = node.lineno - 1
            end = node.end_lineno
            return "\n".join((*lines[:start], replacement, *lines[end:])) + "\n"
    raise ValueError("DEFAULT_GAINS assignment not found")


def render_default_gains(gains: dict[str, float]) -> str:
    lines = ["DEFAULT_GAINS: dict[str, float] = {"]
    for name, value in gains.items():
        lines.append(f'    "{name}": {_format_float(value)},')
    lines.append("}")
    return "\n".join(lines)


def _assignment_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _format_float(value: float) -> str:
    text = f"{value:.8g}"
    return text if "." in text else f"{text}.0"


def main() -> None:
    args = parse_args()
    try:
        applied = apply_best_params(args.source, args.target, dry_run=bool(args.dry_run))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    action = "would apply" if args.dry_run else "applied"
    print(f"{action} {len(applied)} tuned parameter(s) from {args.source}")


if __name__ == "__main__":
    main()
