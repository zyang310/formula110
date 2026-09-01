#!/usr/bin/env python3
"""Export one targeted controller with its submission manifest and runtime files."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
CONTROLLERS_ROOT = SOURCE_ROOT / "controllers"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "formula110-student-controllers.zip"
MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
PROJECT_FILE_NAMES = ("pyproject.toml", "uv.lock")
SUBMISSION_MANIFEST_NAME = "formula110-submission.json"


def controller_module_name(value: str) -> str:
    """Validate a dotted module name inside the controllers package."""
    if value.endswith(".py") or MODULE_NAME_PATTERN.fullmatch(value) is None or not value.startswith("controllers."):
        raise argparse.ArgumentTypeError(f"expected a dotted controllers module name, got {value!r}")
    return value


def module_source(module_name: str) -> Path:
    """Resolve one controller module to its project source file."""
    return SOURCE_ROOT.joinpath(*module_name.split(".")).with_suffix(".py")


def selected_sources(controller_module: str) -> tuple[Path, ...]:
    """Validate one target module and return every controller runtime file."""
    try:
        controller_module_name(controller_module)
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error)) from error
    target = module_source(controller_module)
    if not target.is_file():
        raise FileNotFoundError(f"controller source file not found: {target.relative_to(PROJECT_ROOT)}")

    sources = tuple(
        sorted(
            path
            for path in CONTROLLERS_ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and path.name != "py.typed"
        )
    )

    missing = tuple(path for path in sources if not path.is_file())
    if missing:
        names = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(f"controller source file not found: {names}")
    return sources


def export_controller(controller_module: str, output: Path) -> Path:
    """Write the controller package, dependency metadata, and manifest to a zip."""
    sources = selected_sources(controller_module)
    project_files = tuple(PROJECT_ROOT / name for name in PROJECT_FILE_NAMES if (PROJECT_ROOT / name).is_file())
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    if pyproject_path not in project_files:
        raise FileNotFoundError(f"project dependency file not found: {pyproject_path}")
    resolved_output = output.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(resolved_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sources:
            archive.write(source, arcname=source.relative_to(SOURCE_ROOT))
        for project_file in project_files:
            archive.write(project_file, arcname=project_file.name)
        archive.writestr(
            SUBMISSION_MANIFEST_NAME,
            json.dumps(
                {"schema_version": 1, "controller_module": controller_module},
                indent=2,
            )
            + "\n",
        )
    return resolved_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "controller_module",
        type=controller_module_name,
        help="one controller module to grade, such as controllers.race_faster",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"archive path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        output = export_controller(str(args.controller_module), args.output)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(output)


if __name__ == "__main__":
    main()
