from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from racing import RobotCommand, RobotSensors

PROJECT_ROOT = Path(__file__).parents[1]
TRUSTED_RESULT_PREFIX = "FORMULA110_RESULT="
TRIAL_METRIC_KEYS = (
    "seed",
    "elapsed_seconds",
    "raw_distance_m",
    "partial_laps",
    "lap_count",
    "damage",
    "survived",
    "wall_contact_seconds",
    "max_speed_mps",
    "first_lap_time_seconds",
    "best_lap_time_seconds",
)


def load_evaluator() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "controller_training" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("formula110_solo_evaluator", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVALUATOR = load_evaluator()


def test_solo_trial_result_is_versioned_strict_json() -> None:
    result = EVALUATOR.SoloTrialResult(
        seed=110,
        elapsed_seconds=30.0,
        raw_distance_m=190.0,
        partial_laps=1.05,
        lap_count=1,
        damage=0.0,
        survived=True,
        wall_contact_seconds=0.0,
        max_speed_mps=11.0,
        first_lap_time_seconds=28.5,
        best_lap_time_seconds=28.5,
    )

    record = result.to_dict()

    assert record["schema_version"] == EVALUATOR.SOLO_TRIAL_RESULT_SCHEMA_VERSION
    assert record["record_type"] == EVALUATOR.SOLO_TRIAL_RESULT_RECORD_TYPE
    assert record["ok"] is True
    assert json.loads(json.dumps(record, allow_nan=False)) == record


@pytest.mark.parametrize("duration_seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_solo_evaluator_rejects_invalid_duration_before_starting_panda(duration_seconds: float) -> None:
    evaluator = EVALUATOR.SoloEvaluator()

    with pytest.raises(ValueError, match="finite and positive"):
        evaluator.run_trial(
            controller_factory=lambda: _NeverCalledController(),
            seed=110,
            duration_seconds=duration_seconds,
        )


def test_solo_evaluator_matches_trusted_gradescope_race_worker() -> None:
    duration_seconds = 0.5
    module_name = "controllers.minimum_viable"
    module_file = PROJECT_ROOT / "src" / "controllers" / "minimum_viable.py"

    local_result = _run_json_command(
        [
            sys.executable,
            "-m",
            "scripts.controller_training.evaluator",
            module_name,
            "--seed",
            "110",
            "--seconds",
            str(duration_seconds),
        ]
    )
    worker_environment = {
        **os.environ,
        "FORMULA110_LOCAL_CONTROL": "1",
        "FORMULA110_CONTROL_WORKER": str(PROJECT_ROOT / "autograder" / "gradescope" / "control_worker.py"),
    }
    trusted_result = _run_json_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "autograder" / "gradescope" / "race_worker.py"),
            "--submission",
            str(PROJECT_ROOT),
            "--module-file",
            str(module_file),
            "--seed",
            "110",
            "--seconds",
            str(duration_seconds),
        ],
        environment=worker_environment,
        line_prefix=TRUSTED_RESULT_PREFIX,
    )

    assert trusted_result["ok"] is True
    assert {key: local_result[key] for key in TRIAL_METRIC_KEYS} == {
        key: trusted_result[key] for key in TRIAL_METRIC_KEYS
    }


class _NeverCalledController:
    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        raise AssertionError(f"controller should not have been called: {sensors!r}")


def _run_json_command(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    line_prefix: str = "",
) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert completed.returncode == 0, completed.stderr
    matching_lines = [line for line in completed.stdout.splitlines() if line.startswith(line_prefix)]
    assert matching_lines, completed.stdout
    return json.loads(matching_lines[-1][len(line_prefix) :])
