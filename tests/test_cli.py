from __future__ import annotations

import json
from pathlib import Path

import pytest

from racing.game import cli
from racing.game.cli import build_argument_parser
from racing.game.config import CameraView, GameConfig, HeadToHeadViewerConfig
from racing.graphics.colors import (
    DEFAULT_CHALLENGER_TEAM_COLOR,
    DEFAULT_FORMULA_TEAM_COLOR,
    DEFAULT_INCUMBENT_TEAM_COLOR,
)
from racing.track.world import (
    TRACK_ID_BLUE_RIDGE_LOOP,
    TRACK_ID_CAROLINA_LOOP,
    TRACK_ID_DOGWOOD_LOOP,
    TRACK_ID_HARBOR_LOOP,
    TRACK_ID_MUGELLO_SHORT,
    TRACK_ID_PINE_SWITCHBACKS,
    TRACK_ID_STADIUM_LOOP,
)


class _FakeApp:
    def run(self) -> None:
        pass


class _FakeHeadlessResult:
    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, "summary": {"winner": "challenger"}}


def test_parser_has_student_controller_playable_mode() -> None:
    args = build_argument_parser().parse_args(["--student-module", "student_driver.py", "--camera", "follow"])

    assert args.student_module == "student_driver.py"
    assert args.camera == "follow"
    assert args.command is None


def test_parser_accepts_human_gameplay_recording_path() -> None:
    args = build_argument_parser().parse_args(["--record-human", "artifacts/human.jsonl"])

    assert args.record_human == Path("artifacts/human.jsonl")


def test_parser_uses_unc_default_formula_color() -> None:
    args = build_argument_parser().parse_args([])

    assert args.team_color == DEFAULT_FORMULA_TEAM_COLOR


def test_parser_defaults_to_drone_camera() -> None:
    parser = build_argument_parser()
    playable_args = parser.parse_args([])
    head_to_head_args = parser.parse_args(["h2h"])

    assert playable_args.camera == CameraView.DRONE.value
    assert head_to_head_args.camera == CameraView.DRONE.value


def test_parser_selects_tracks_in_both_racing_modes() -> None:
    parser = build_argument_parser()
    playable_args = parser.parse_args(["--track", TRACK_ID_CAROLINA_LOOP])
    head_to_head_args = parser.parse_args(["h2h", "--track", TRACK_ID_CAROLINA_LOOP])

    assert playable_args.track == TRACK_ID_CAROLINA_LOOP
    assert head_to_head_args.track == TRACK_ID_CAROLINA_LOOP


def test_parser_accepts_each_generated_track() -> None:
    parser = build_argument_parser()

    for track_id in (
        TRACK_ID_STADIUM_LOOP,
        TRACK_ID_HARBOR_LOOP,
        TRACK_ID_DOGWOOD_LOOP,
        TRACK_ID_BLUE_RIDGE_LOOP,
        TRACK_ID_PINE_SWITCHBACKS,
    ):
        assert parser.parse_args(["--track", track_id]).track == track_id


def test_parser_defaults_to_mugello_short_track() -> None:
    parser = build_argument_parser()

    assert parser.parse_args([]).track == TRACK_ID_MUGELLO_SHORT
    assert parser.parse_args(["h2h"]).track == TRACK_ID_MUGELLO_SHORT


def test_head_to_head_parser_defaults_to_thirty_second_race() -> None:
    args = build_argument_parser().parse_args(["h2h"])

    assert args.round_seconds == 30.0


def test_parser_shares_seed_option_between_playable_and_head_to_head_modes() -> None:
    parser = build_argument_parser()

    playable_args = parser.parse_args(["--seed", "271"])
    head_to_head_args = parser.parse_args(["h2h", "--seed", "272"])
    global_head_to_head_args = parser.parse_args(["--seed", "273", "h2h"])

    assert playable_args.seed == 271
    assert head_to_head_args.seed == 272
    assert global_head_to_head_args.seed == 273


def test_parser_rejects_removed_vehicle_flag() -> None:
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(["--vehicle", "formula"])


def test_parser_rejects_old_head_to_head_module_flags() -> None:
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(["h2h", "--challenger-student-module", "driver.py"])


def test_parser_accepts_student_head_to_head_without_vehicle_flag() -> None:
    args = build_argument_parser().parse_args(
        [
            "h2h",
            "--challenger-module",
            "driver_a.py",
            "--incumbent-module",
            "driver_b.py",
            "--races",
            "3",
        ]
    )

    assert args.command == "h2h"
    assert args.challenger_module == "driver_a.py"
    assert args.incumbent_module == "driver_b.py"
    assert args.races == 3
    assert args.json is False
    assert args.challenger_team_color == DEFAULT_CHALLENGER_TEAM_COLOR
    assert args.incumbent_team_color == DEFAULT_INCUMBENT_TEAM_COLOR


def test_parser_accepts_machine_readable_head_to_head_results() -> None:
    args = build_argument_parser().parse_args(
        [
            "h2h",
            "--challenger-module",
            "driver_a.py",
            "--incumbent-module",
            "driver_b.py",
            "--json",
        ]
    )

    assert args.json is True


def test_headless_cli_prints_json_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    controller_path = tmp_path / "controller.py"
    controller_path.write_text(
        "from racing import RobotCommand\ndef control(sensors):\n    return RobotCommand()\n",
        encoding="utf-8",
    )

    captured_arguments: dict[str, object] = {}

    def fake_run_headless_head_to_head(**arguments: object) -> _FakeHeadlessResult:
        captured_arguments.update(arguments)
        return _FakeHeadlessResult()

    monkeypatch.setattr(cli, "run_headless_head_to_head", fake_run_headless_head_to_head)

    cli.main(
        [
            "h2h",
            "--challenger-module",
            str(controller_path),
            "--incumbent-module",
            str(controller_path),
            "--seed",
            "271",
            "--track",
            TRACK_ID_CAROLINA_LOOP,
            "--json",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output == {"schema_version": 1, "summary": {"winner": "challenger"}}
    assert captured_arguments["random_seed"] == 271
    assert captured_arguments["track_id"] == TRACK_ID_CAROLINA_LOOP


def test_cli_passes_human_recording_path_to_playable_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording_path = tmp_path / "human.jsonl"
    captured_config: GameConfig | None = None

    def fake_create_app(config: GameConfig) -> _FakeApp:
        nonlocal captured_config
        captured_config = config
        return _FakeApp()

    monkeypatch.setattr(cli, "create_app", fake_create_app)

    cli.main(["--record-human", str(recording_path)])

    assert captured_config is not None
    assert captured_config.human_recording_path == recording_path


def test_cli_passes_seed_to_playable_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_config: GameConfig | None = None

    def fake_create_app(config: GameConfig) -> _FakeApp:
        nonlocal captured_config
        captured_config = config
        return _FakeApp()

    monkeypatch.setattr(cli, "create_app", fake_create_app)

    cli.main(["--seed", "271"])

    assert captured_config is not None
    assert captured_config.random_seed == 271


def test_cli_passes_track_to_playable_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_config: GameConfig | None = None

    def fake_create_app(config: GameConfig) -> _FakeApp:
        nonlocal captured_config
        captured_config = config
        return _FakeApp()

    monkeypatch.setattr(cli, "create_app", fake_create_app)

    cli.main(["--track", TRACK_ID_CAROLINA_LOOP])

    assert captured_config is not None
    assert captured_config.track_id == TRACK_ID_CAROLINA_LOOP


def test_cli_rejects_recording_an_automated_controller(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--student-module",
                "controllers.crash_fast",
                "--record-human",
                str(tmp_path / "human.jsonl"),
            ]
        )


def test_cli_rejects_human_recording_in_head_to_head_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--record-human", str(tmp_path / "human.jsonl"), "h2h"])


def test_playable_student_color_overrides_cli_team_color(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "orange_driver.py"
    module_path.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        "\n"
        "RACING_COLOR = '#ff8000'\n"
        "\n"
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        "    return RobotCommand(throttle=0.4)\n",
        encoding="utf-8",
    )
    captured_config: GameConfig | None = None

    def fake_create_app(config: GameConfig) -> _FakeApp:
        nonlocal captured_config
        captured_config = config
        return _FakeApp()

    monkeypatch.setattr(cli, "create_app", fake_create_app)

    cli.main(["--student-module", str(module_path), "--team-color", "#0000ff"])

    assert captured_config is not None
    assert captured_config.team_color == (1.0, 128 / 255, 0.0, 1.0)


def test_watched_head_to_head_student_colors_override_team_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    challenger_path = tmp_path / "orange_challenger.py"
    challenger_path.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        "\n"
        "RACING_COLOR = '#ff8000'\n"
        "\n"
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        "    return RobotCommand(throttle=0.4)\n",
        encoding="utf-8",
    )
    incumbent_path = tmp_path / "green_incumbent.py"
    incumbent_path.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        "\n"
        "RACING_COLOR = (0.0, 1.0, 0.0)\n"
        "\n"
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        "    return RobotCommand(throttle=0.2)\n",
        encoding="utf-8",
    )
    captured_config: HeadToHeadViewerConfig | None = None

    def fake_create_head_to_head_viewer_app(config: HeadToHeadViewerConfig) -> _FakeApp:
        nonlocal captured_config
        captured_config = config
        return _FakeApp()

    monkeypatch.setattr(cli, "create_head_to_head_viewer_app", fake_create_head_to_head_viewer_app)

    cli.main(
        [
            "h2h",
            "--watch",
            "--challenger-module",
            str(challenger_path),
            "--incumbent-module",
            str(incumbent_path),
            "--seed",
            "271",
            "--challenger-team-color",
            "#0000ff",
            "--incumbent-team-color",
            "#ff0000",
        ]
    )

    assert captured_config is not None
    assert captured_config.random_seed == 271
    assert captured_config.challenger_team_color == (1.0, 128 / 255, 0.0, 1.0)
    assert captured_config.incumbent_team_color == (0.0, 1.0, 0.0, 1.0)
