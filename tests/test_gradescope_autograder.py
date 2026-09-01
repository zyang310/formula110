from __future__ import annotations

import argparse
import importlib.util
import json
import stat
import subprocess
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def load_builder() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "build_gradescope_autograder.py"
    spec = importlib.util.spec_from_file_location("build_gradescope_autograder", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_exporter() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "export_student_controllers.py"
    spec = importlib.util.spec_from_file_location("export_student_controllers", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_grader() -> ModuleType:
    path = Path(__file__).parents[1] / "autograder" / "gradescope" / "grade.py"
    spec = importlib.util.spec_from_file_location("formula110_gradescope_grade", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_race_worker() -> ModuleType:
    path = Path(__file__).parents[1] / "autograder" / "gradescope" / "race_worker.py"
    spec = importlib.util.spec_from_file_location("formula110_race_worker", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_gradescope_archive_has_required_root_files_and_config(tmp_path: Path) -> None:
    builder = load_builder()
    output = tmp_path / "autograder.zip"

    built = builder.build_archive("controllers.minimum", "controllers.improved", output)

    assert built == output.resolve()
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {
            "setup.sh",
            "run_autograder",
            "grade.py",
            "race_worker.py",
            "control_worker.py",
            "config.json",
        } <= names
        assert "trusted/racing/student/api.py" in names
        assert all(not name.startswith("formula110-autograder/") for name in names)
        config = json.loads(archive.read("config.json"))
        assert config["modules"] == {
            "minimum_viable": "controllers.minimum",
            "improved": "controllers.improved",
        }
        assert config["seeds"] == [110, 2026]
        assert config["duration_seconds"] == 30.0
        assert sum(config["rubric"].values()) == 100.0
        assert sum(value for key, value in config["rubric"].items() if key.startswith("minimum_")) == 65.0
        assert sum(value for key, value in config["rubric"].items() if key.startswith("improved_")) == 35.0
        for executable in ("setup.sh", "run_autograder", "race_worker.py", "control_worker.py"):
            mode = archive.getinfo(executable).external_attr >> 16
            assert mode & stat.S_IXUSR
        setup = archive.read("setup.sh").decode()
        assert "UV_PYTHON_INSTALL_DIR=/opt/formula110-python" in setup
        assert "formula110-runtime.pth" in setup


def test_build_gradescope_archive_rejects_same_module(tmp_path: Path) -> None:
    builder = load_builder()

    with pytest.raises(ValueError, match="must differ"):
        builder.build_archive("controllers.same", "controllers.same", tmp_path / "autograder.zip")


def test_controller_worker_declares_cpu_only_512_mib_runtime() -> None:
    worker = load_race_worker()

    assert worker.CONTROLLER_MEMORY_LIMIT_BYTES == 512 * 1024 * 1024
    assert worker.CPU_ONLY_ENVIRONMENT["FORMULA110_DEVICE"] == "cpu"
    assert worker.CPU_ONLY_ENVIRONMENT["CUDA_VISIBLE_DEVICES"] == ""
    assert worker.CPU_ONLY_ENVIRONMENT["JAX_PLATFORMS"] == "cpu"


@pytest.mark.parametrize("value", ["controller-name", "controllers/foo", "controllers.foo.py", ""])
def test_module_name_rejects_non_dotted_names(value: str) -> None:
    builder = load_builder()

    with pytest.raises(argparse.ArgumentTypeError):
        builder.module_name(value)


def test_export_selected_student_controller_uses_package_layout(tmp_path: Path) -> None:
    exporter = load_exporter()
    output = tmp_path / "submission.zip"

    built = exporter.export_controller("controllers.crash_fast", output)

    assert built == output.resolve()
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {
            "controllers/__init__.py",
            "controllers/crash_fast.py",
            "formula110-submission.json",
            "pyproject.toml",
            "uv.lock",
        } <= names
        manifest = json.loads(archive.read("formula110-submission.json"))
        assert manifest == {"schema_version": 1, "controller_module": "controllers.crash_fast"}


@pytest.mark.parametrize("arguments", [[], ["controllers.one", "controllers.two"]])
def test_export_cli_requires_exactly_one_controller(arguments: list[str]) -> None:
    exporter = load_exporter()

    with pytest.raises(SystemExit):
        exporter.build_parser().parse_args(arguments)


def test_export_excludes_python_cache_and_type_marker(tmp_path: Path) -> None:
    exporter = load_exporter()
    output = tmp_path / "submission.zip"

    exporter.export_controller("controllers.crash_fast", output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "controllers/crash_fast.py" in names
        assert not any("__pycache__" in name or name.endswith(".pyc") or name.endswith("/py.typed") for name in names)


def test_export_selected_controller_includes_complete_runtime_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = load_exporter()
    source_root = tmp_path / "src"
    controllers_root = source_root / "controllers"
    controllers_root.mkdir(parents=True)
    (controllers_root / "__init__.py").write_text("", encoding="utf-8")
    (controllers_root / "main.py").write_text("from controllers.helper import VALUE\n", encoding="utf-8")
    (controllers_root / "helper.py").write_text("VALUE: int = 1\n", encoding="utf-8")
    (controllers_root / "weights.bin").write_bytes(b"checkpoint")
    (controllers_root / "py.typed").write_text("", encoding="utf-8")
    cache = controllers_root / "__pycache__"
    cache.mkdir()
    (cache / "main.cpython-311.pyc").write_bytes(b"cache")
    output = tmp_path / "submission.zip"
    monkeypatch.setattr(exporter, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(exporter, "CONTROLLERS_ROOT", controllers_root)

    exporter.export_controller("controllers.main", output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {
            "controllers/__init__.py",
            "controllers/main.py",
            "controllers/helper.py",
            "controllers/weights.bin",
            "formula110-submission.json",
            "pyproject.toml",
            "uv.lock",
        } <= names
        assert "controllers/py.typed" not in names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_export_student_controllers_reports_missing_module(tmp_path: Path) -> None:
    exporter = load_exporter()

    with pytest.raises(FileNotFoundError, match="not found"):
        exporter.export_controller("controllers.does_not_exist", tmp_path / "submission.zip")


def test_minimum_only_submission_still_earns_all_minimum_points(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = load_builder()
    grader = load_grader()
    config = builder.build_config("controllers.minimum", "controllers.improved")
    minimum_file = Path("/submission/controllers/minimum.py")

    monkeypatch.setattr(grader, "read_config", lambda: config)

    def locate_module(name: str) -> tuple[Path | None, str]:
        if name == "controllers.minimum":
            return minimum_file, "found controllers/minimum.py"
        return None, "expected controllers/improved.py"

    def static_checks(module_file: Path | None, check_id: str) -> dict[str, tuple[bool, str]]:
        return {
            "pyright": (module_file is not None, "passed" if module_file is not None else "missing"),
            "ruff_lint": (module_file is not None, "passed" if module_file is not None else "missing"),
            "ruff_format": (module_file is not None, "passed" if module_file is not None else "missing"),
        }

    def validate_control(module_file: Path | None, function_name: str) -> dict[str, object]:
        return {"ok": module_file is not None, "error": "module file is missing"}

    monkeypatch.setattr(grader, "locate_module", locate_module)
    monkeypatch.setattr(grader, "static_checks", static_checks)
    monkeypatch.setattr(grader, "validate_control", validate_control)

    def passing_trials(
        module_file: Path,
        function_name: str,
        seeds: list[int],
        duration_seconds: float,
        timeout_seconds: float,
    ) -> list[dict[str, object]]:
        return [
            {
                "ok": True,
                "seed": seed,
                "raw_distance_m": 200.0,
                "partial_laps": 1.1,
                "lap_count": 1,
                "damage": 0.0,
                "survived": True,
                "wall_contact_seconds": 0.0,
                "max_speed_mps": 10.0,
                "first_lap_time_seconds": 30.0,
                "best_lap_time_seconds": 30.0,
            }
            for seed in seeds
        ]

    monkeypatch.setattr(grader, "run_trials", passing_trials)

    results = grader.grade()

    assert sum(float(test["score"]) for test in results["tests"]) == 65.0
    assert all(test["status"] == "passed" for test in results["tests"] if "Minimum" in test["name"])
    assert all(test["status"] == "failed" for test in results["tests"] if "Improved" in test["name"])
    assert results["leaderboard"] == []


def test_pyright_receives_submission_file_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grader = load_grader()
    module_file = tmp_path / "controller.py"
    module_file.write_text("value: int = 1\n", encoding="utf-8")
    results_path = tmp_path / "results" / "results.json"
    results_path.parent.mkdir()
    commands: list[list[str]] = []

    def record_command(command: list[str], *, timeout_seconds: float) -> tuple[bool, str]:
        commands.append(command)
        return True, "passed"

    monkeypatch.setattr(grader, "RESULTS_PATH", results_path)
    monkeypatch.setattr(grader, "run_command", record_command)

    checks = grader.static_checks(module_file, "minimum")

    assert all(passed for passed, _output in checks.values())
    pyright_command = commands[0]
    assert "--pythonpath" in pyright_command
    assert pyright_command[-1] == str(module_file)
    config = json.loads((results_path.parent / "pyrightconfig-minimum.json").read_text(encoding="utf-8"))
    assert "include" not in config


def test_controller_startup_diagnostic_includes_child_exit_and_stderr() -> None:
    module = load_race_worker()
    client = module.ControllerClient(
        submission=Path("/submission"), module_file=Path("/submission/controller.py"), function_name="control"
    )
    client.process = subprocess.Popen(
        ["bash", "-c", "echo interpreter-permission-denied >&2; exit 126"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    client.process.wait(timeout=2.0)

    message = client._unexpected_exit_message()

    assert "exit 126" in message
    assert "interpreter-permission-denied" in message
