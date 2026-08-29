from __future__ import annotations

import importlib.util
import json
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from controllers.preview_controller import ControllerParameters, PreviewController
from controllers.race_faster import RACE_FASTER_PARAMETERS
from racing import RobotSensors

PROJECT_ROOT = Path(__file__).parents[1]
TOOLS_ROOT = PROJECT_ROOT / "scripts" / "controller_training"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_tool(name: str) -> ModuleType:
    path = TOOLS_ROOT / f"{name}.py"
    module_name = f"formula110_controller_training_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_generated_seed_manifest_matches_logged_suites_and_never_overlaps() -> None:
    seeds = load_tool("seeds")

    manifest = seeds.generate_seed_manifest()

    assert manifest.official == (110, 2026)
    assert manifest.training[:3] == (30991, 89384, 37399)
    assert manifest.training[-1] == 1718
    assert manifest.validation == (82361, 16872, 41256, 8681, 60604, 19331, 37089, 75222, 88117, 90661, 76542, 56221)
    assert manifest.final_soak[:3] == (38605, 37849, 60758)
    assert manifest.final_soak[-1] == 10221
    combined = (*manifest.official, *manifest.training, *manifest.validation, *manifest.final_soak)
    assert len(combined) == len(set(combined))


def test_seed_manifest_write_is_idempotent_and_strict_json(tmp_path: Path) -> None:
    seeds = load_tool("seeds")
    path = tmp_path / "seed-manifest.json"

    seeds.write_seed_manifest(path)
    first = path.read_text(encoding="utf-8")
    seeds.write_seed_manifest(path)

    assert path.read_text(encoding="utf-8") == first
    assert seeds.load_seed_manifest(path) == seeds.generate_seed_manifest()
    json.dumps(json.loads(first), allow_nan=False)


def test_cem_checkpoint_resume_samples_same_next_generation(tmp_path: Path) -> None:
    cem = load_tool("cem")
    space = cem.ParameterSpace(
        (
            cem.ParameterSpec("speed", 1.0, 20.0, 8.0, 2.0),
            cem.ParameterSpec("steer", 0.0, 1.0, 0.5, 0.2),
        )
    )
    config = cem.CEMConfig(population_size=6, elite_count=2, generations=3, optimizer_seed=1234)
    uninterrupted = cem.CEMOptimizer(space=space, config=config)
    population = uninterrupted.sample_population()
    elites = population[:2]
    uninterrupted.update(elites)
    checkpoint = tmp_path / "checkpoint.json"
    uninterrupted.save_checkpoint(checkpoint, best_candidate=elites[0], metrics={"safe": 2})

    resumed = cem.CEMOptimizer.from_checkpoint(space=space, config=config, path=checkpoint)

    assert resumed.generation == 1
    assert resumed.mean == uninterrupted.mean
    assert resumed.deviation == uninterrupted.deviation
    assert resumed.sample_population() == uninterrupted.sample_population()
    record = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert record["optimizer_seed"] == 1234
    assert record["generation"] == 1
    assert record["metrics"] == {"safe": 2}


def test_ga_checkpoint_resume_samples_same_next_generation(tmp_path: Path) -> None:
    cem = load_tool("cem")
    genetic = load_tool("genetic")
    space = cem.ParameterSpace(
        (
            cem.ParameterSpec("speed", 1.0, 20.0, 8.0, 2.0),
            cem.ParameterSpec("steer", 0.0, 1.0, 0.5, 0.2),
        )
    )
    config = genetic.GAConfig(population_size=6, elite_count=2, generations=3, optimizer_seed=1234)
    uninterrupted = genetic.GeneticOptimizer(space=space, config=config, checkpoint_context={"fixed": 0.5})
    population = uninterrupted.sample_population()
    uninterrupted.update_from_ranking(
        ranked_population=population,
        ranked_elites=population[:2],
        generation_best_score=(1.0, 2.0),
    )
    checkpoint = tmp_path / "checkpoint.json"
    uninterrupted.save_checkpoint(checkpoint, best_candidate=population[0], metrics={"safe": 2})

    resumed = genetic.GeneticOptimizer.from_checkpoint(
        space=space,
        config=config,
        path=checkpoint,
        checkpoint_context={"fixed": 0.5},
    )

    assert resumed.generation == 1
    assert resumed.sample_population() == uninterrupted.sample_population()


def test_ga_mutation_can_move_a_gene_outside_the_elite_span() -> None:
    cem = load_tool("cem")
    genetic = load_tool("genetic")
    space = cem.ParameterSpace((cem.ParameterSpec("gene", 0.0, 1.0, 0.5, 0.1),))
    config = genetic.GAConfig(
        population_size=6,
        elite_count=2,
        generations=2,
        optimizer_seed=4,
        mutation_probability=1.0,
    )
    optimizer = genetic.GeneticOptimizer(space=space, config=config)
    identical = tuple(cem.Candidate(index=index, values=(0.5,)) for index in range(6))
    optimizer.update_from_ranking(
        ranked_population=identical,
        ranked_elites=identical[:2],
        generation_best_score=(1.0,),
    )

    next_population = optimizer.sample_population()

    assert any(abs(candidate.values[0] - 0.5) > 1e-9 for candidate in next_population[2:])


def test_ga_elitism_preserves_the_best_vectors_and_bounds() -> None:
    cem = load_tool("cem")
    genetic = load_tool("genetic")
    space = cem.ParameterSpace((cem.ParameterSpec("gene", 0.0, 1.0, 0.5, 0.1),))
    config = genetic.GAConfig(population_size=6, elite_count=2, generations=2, optimizer_seed=8)
    optimizer = genetic.GeneticOptimizer(space=space, config=config)
    ranked = tuple(cem.Candidate(index=index, values=(1.0 - index * 0.1,)) for index in range(6))
    optimizer.update_from_ranking(
        ranked_population=ranked,
        ranked_elites=ranked[:2],
        generation_best_score=(1.0,),
    )

    next_population = optimizer.sample_population()

    assert tuple(candidate.values for candidate in next_population[:2]) == tuple(
        candidate.values for candidate in ranked[:2]
    )
    assert all(0.0 <= candidate.values[0] <= 1.0 for candidate in next_population)


def test_ga_stagnation_inflates_and_resets_mutation_scale() -> None:
    cem = load_tool("cem")
    genetic = load_tool("genetic")
    space = cem.ParameterSpace((cem.ParameterSpec("gene", 0.0, 1.0, 0.5, 0.1),))
    config = genetic.GAConfig(
        population_size=4,
        elite_count=1,
        generations=5,
        optimizer_seed=9,
        stagnation_patience=2,
    )
    optimizer = genetic.GeneticOptimizer(space=space, config=config)
    ranked = tuple(cem.Candidate(index=index, values=(0.5,)) for index in range(4))
    for score in ((1.0,), (1.0,), (1.0,)):
        optimizer.update_from_ranking(
            ranked_population=ranked,
            ranked_elites=ranked[:1],
            generation_best_score=score,
        )
    assert optimizer.current_mutation_scale == pytest.approx(0.18)

    optimizer.update_from_ranking(
        ranked_population=ranked,
        ranked_elites=ranked[:1],
        generation_best_score=(2.0,),
    )

    assert optimizer.current_mutation_scale == config.mutation_scale
    assert optimizer.stagnation_counter == 0


def test_cem_checkpoint_rejects_a_ga_checkpoint(tmp_path: Path) -> None:
    cem = load_tool("cem")
    genetic = load_tool("genetic")
    space = cem.ParameterSpace((cem.ParameterSpec("gene", 0.0, 1.0, 0.5, 0.1),))
    ga_config = genetic.GAConfig(population_size=4, elite_count=1, generations=2, optimizer_seed=10)
    optimizer = genetic.GeneticOptimizer(space=space, config=ga_config)
    population = optimizer.sample_population()
    optimizer.update_from_ranking(
        ranked_population=population,
        ranked_elites=population[:1],
        generation_best_score=(1.0,),
    )
    checkpoint = tmp_path / "ga.json"
    optimizer.save_checkpoint(checkpoint, best_candidate=population[0], metrics={})

    with pytest.raises(ValueError, match="different optimizer"):
        cem.CEMOptimizer.from_checkpoint(
            space=space,
            config=cem.CEMConfig(population_size=4, elite_count=1, generations=2, optimizer_seed=10),
            path=checkpoint,
        )


def test_experiment_journal_is_append_only_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    records = load_tool("records")
    journal = records.ExperimentJournal(tmp_path / "experiments.jsonl")
    record = records.ExperimentRecord(
        experiment_id="T-001",
        date="2026-08-28",
        controller="candidate",
        artifact="checkpoint.json",
        seeds=(1, 2),
        metrics={"distance_m": 10.0},
        decision="retain",
        next_step="continue",
    )

    journal.append(record)

    assert journal.read_all() == (record,)
    with pytest.raises(ValueError, match="already exists"):
        journal.append(record)


def test_search_seed_rotation_and_minimum_ranking_are_safety_first() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    training = tuple(range(1, 29))

    assert search.rotating_training_seeds(training, 0) == (1, 2, 3, 4, 5, 6)
    assert search.rotating_training_seeds(training, 4) == (25, 26, 27, 28, 1, 2)
    fast_damage = evaluator.SoloTrialResult(
        seed=1,
        elapsed_seconds=30.0,
        raw_distance_m=250.0,
        partial_laps=1.3,
        lap_count=1,
        damage=0.01,
        survived=True,
        wall_contact_seconds=0.0,
        max_speed_mps=15.0,
        first_lap_time_seconds=23.0,
        best_lap_time_seconds=23.0,
    )
    slower_safe = evaluator.SoloTrialResult(
        seed=1,
        elapsed_seconds=30.0,
        raw_distance_m=190.0,
        partial_laps=1.05,
        lap_count=1,
        damage=0.0,
        survived=True,
        wall_contact_seconds=0.0,
        max_speed_mps=11.0,
        first_lap_time_seconds=29.0,
        best_lap_time_seconds=29.0,
    )

    assert search.minimum_score((slower_safe,)) > search.minimum_score((fast_damage,))


def test_improved_ranking_uses_distance_instead_of_peak_speed() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    faster_lap = evaluator.SoloTrialResult(
        seed=1,
        elapsed_seconds=30.0,
        raw_distance_m=300.0,
        partial_laps=1.6,
        lap_count=1,
        damage=0.0,
        survived=True,
        wall_contact_seconds=0.0,
        max_speed_mps=18.0,
        first_lap_time_seconds=18.0,
        best_lap_time_seconds=18.0,
    )
    higher_peak = evaluator.SoloTrialResult(
        seed=1,
        elapsed_seconds=30.0,
        raw_distance_m=290.0,
        partial_laps=1.5,
        lap_count=1,
        damage=0.0,
        survived=True,
        wall_contact_seconds=0.0,
        max_speed_mps=25.0,
        first_lap_time_seconds=19.0,
        best_lap_time_seconds=19.0,
    )

    baseline_distances = {1: 200.0}

    assert search.improved_score((faster_lap,), baseline_distances) > search.improved_score(
        (higher_peak,), baseline_distances
    )


def _improved_trial(
    evaluator: ModuleType,
    *,
    raw_distance_m: float,
    damage: float = 0.0,
    wall_contact_seconds: float = 0.0,
    lap_count: int = 1,
) -> Any:
    return evaluator.SoloTrialResult(
        seed=1,
        elapsed_seconds=30.0,
        raw_distance_m=raw_distance_m,
        partial_laps=raw_distance_m / 181.1,
        lap_count=lap_count,
        damage=damage,
        survived=True,
        wall_contact_seconds=wall_contact_seconds,
        max_speed_mps=20.0,
        first_lap_time_seconds=20.0,
        best_lap_time_seconds=20.0,
    )


def test_improved_score_prefers_a_faster_candidate_with_small_bounded_damage() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    faster_with_incident = _improved_trial(evaluator, raw_distance_m=494.0, damage=0.06, wall_contact_seconds=0.67)
    slower_and_clean = _improved_trial(evaluator, raw_distance_m=437.0)

    baseline_distances = {1: 350.0}

    assert search.improved_score((faster_with_incident,), baseline_distances) > search.improved_score(
        (slower_and_clean,), baseline_distances
    )


def test_improved_score_rejects_a_candidate_that_exceeds_the_incident_budget() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    reckless = _improved_trial(evaluator, raw_distance_m=520.0, damage=0.35)
    slower_and_clean = _improved_trial(evaluator, raw_distance_m=437.0)

    baseline_distances = {1: 350.0}

    assert search.improved_score((slower_and_clean,), baseline_distances) > search.improved_score(
        (reckless,), baseline_distances
    )


def test_improved_score_ranks_lap_completion_above_the_incident_budget() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    lapped_with_incident = _improved_trial(evaluator, raw_distance_m=200.0, damage=0.30, lap_count=1)
    clean_without_a_lap = _improved_trial(evaluator, raw_distance_m=170.0, lap_count=0)

    baseline_distances = {1: 160.0}

    assert search.improved_score((lapped_with_incident,), baseline_distances) > search.improved_score(
        (clean_without_a_lap,), baseline_distances
    )


def test_faster_line_preset_is_isolated_and_uses_planned_bounds() -> None:
    search = load_tool("search")

    existing_names = search.faster_parameter_space().names
    base, space = search._preset_configuration("faster-line")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}

    assert existing_names == (
        "heading_steer_gain",
        "center_steer_gain",
        "yaw_damping_gain",
        "racing_line_offset_ratio",
        "curvature_heading_degrees",
        "curvature_lateral_ratio",
        "straight_target_speed_mps",
        "corner_target_speed_mps",
        "steering_speed_reduction",
        "yaw_speed_reduction",
        "front_brake_start_m",
        "side_slow_start_m",
        "side_speed_floor",
        "brake_gain",
    )
    assert base.phase_aware_racing_line
    assert len(space.specs) == 15
    assert bounds == {
        "straight_target_speed_mps": (18.0, 26.0),
        "corner_target_speed_mps": (11.0, 17.0),
        "throttle_gain": (0.18, 0.90),
        "front_brake_start_m": (4.0, 14.0),
        "steering_speed_reduction": (0.0, 0.35),
        "yaw_speed_reduction": (0.0, 0.30),
        "curvature_heading_degrees": (45.0, 90.0),
        "curvature_lateral_ratio": (0.60, 1.40),
        "heading_steer_gain": (0.75, 1.30),
        "center_steer_gain": (0.10, 0.55),
        "wall_balance_gain": (0.0, 0.45),
        "steer_slew_per_tick": (0.06, 0.20),
        "racing_line_offset_ratio": (0.0, 0.45),
        "racing_line_entry_offset_ratio": (0.0, 0.45),
        "racing_line_exit_offset_ratio": (0.0, 0.45),
    }
    assert search.parse_args(["faster-line"]).preset == "faster-line"


def test_faster_line_v2_probe_and_ga_spaces_are_isolated() -> None:
    search = load_tool("search")

    probe_base, probe = search.preset_configuration("faster-line-v2-probe")
    ga_base, ga = search.preset_configuration("faster-line-v2")
    probe_bounds = {spec.name: (spec.minimum, spec.maximum) for spec in probe.specs}

    assert probe_base.pose_invariant_racing_line
    assert ga_base.pose_invariant_racing_line
    assert len(probe.specs) == 19
    assert len(ga.specs) == 17
    assert "preview_line_compensation" in probe.names
    assert "wall_balance_line_compensation" in probe.names
    assert "preview_line_compensation" not in ga.names
    assert "wall_balance_line_compensation" not in ga.names
    assert "curvature_heading_degrees" not in probe.names
    assert "steer_slew_per_tick" not in probe.names
    assert probe_bounds["center_steer_gain"] == (0.10, 0.90)
    assert probe_bounds["curvature_lateral_ratio"] == (0.20, 1.00)
    assert probe_bounds["line_target_slew_per_tick"] == (0.005, 0.10)
    # The phase signal is a local curvature that peaks near 0.16 on this track,
    # so a ceiling above that would search values that disable the line entirely.
    assert probe_bounds["line_turn_sensitivity"] == (0.010, 0.150)
    args = search.parse_args(["faster-line-v2", "--optimizer", "ga", "--objective", "improved-v2"])
    assert args.optimizer == "ga"
    assert args.objective == "improved-v2"


def test_faster_line_v2_ga_is_seeded_from_the_probe_winner(tmp_path: Path) -> None:
    search = load_tool("search")
    _, probe = search.preset_configuration("faster-line-v2-probe")
    vector = {spec.name: spec.initial for spec in probe.specs}
    vector.update(
        curvature_lateral_ratio=0.77,
        preview_line_compensation=0.83,
        wall_balance_line_compensation=0.79,
    )
    checkpoint = tmp_path / "probe.json"
    checkpoint.write_text(json.dumps({"best_parameter_vector": vector}), encoding="utf-8")

    base, ga = search.preset_configuration("faster-line-v2", seed_checkpoint=checkpoint)
    initial = dict(zip(ga.names, ga.initial_mean, strict=True))

    assert initial["curvature_lateral_ratio"] == 0.77
    assert base.preview_line_compensation == 0.83
    assert base.wall_balance_line_compensation == 0.79


def test_improved_score_v2_keeps_three_hard_tiers_and_one_robust_distance() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    results = tuple(
        evaluator.SoloTrialResult(
            seed=seed,
            elapsed_seconds=30.0,
            raw_distance_m=distance,
            partial_laps=distance / 181.1,
            lap_count=1,
            damage=0.0,
            survived=True,
            wall_contact_seconds=0.0,
            max_speed_mps=20.0,
            first_lap_time_seconds=10.0,
            best_lap_time_seconds=9.5,
        )
        for seed, distance in ((1, 500.0), (2, 550.0), (3, 600.0))
    )

    score = search.improved_score_v2(results, {1: 300.0, 2: 300.0, 3: 300.0})

    assert score[:3] == (3.0, 3.0, 3.0)
    assert len(score) == 4
    assert score[3] == pytest.approx(search.percentile((500.0, 550.0, 600.0), 0.10))


def test_diversity_metrics_record_rejection_tiers() -> None:
    search = load_tool("search")
    cem = load_tool("cem")
    space = cem.ParameterSpace((cem.ParameterSpec("gene", 0.0, 1.0, 0.5, 0.1),))
    evaluations = tuple(
        search.CandidateEvaluation(
            candidate=cem.Candidate(index=index, values=(1.0 - index * 0.2,)),
            results=(),
            score=score,
        )
        for index, score in enumerate(
            ((3.0, 3.0, 3.0, 1.0), (3.0, 3.0, 3.0, 0.9), (3.0, 3.0, 2.0, 5.0), (2.0, 9.0, 9.0, 9.0))
        )
    )

    metrics = search._diversity_metrics(
        space=space,
        ranked_batch=evaluations,
        ranked_elites=evaluations[:2],
        elite_count=2,
    )

    assert metrics["rejection_count_by_score_tier"] == [1, 0, 1, 0]
    assert metrics["rejection_count_tied_at_cutoff"] == 0


def test_trace_collector_emits_strict_tick_and_summary_records() -> None:
    trace = load_tool("trace")
    evaluator = load_tool("evaluator")
    stream = StringIO()
    collector = trace.TraceCollector(controller=PreviewController(ControllerParameters()), stream=stream)

    collector(RobotSensors())
    collector.write_summary(
        evaluator.SoloTrialResult(
            seed=110,
            elapsed_seconds=1.0 / 60.0,
            raw_distance_m=0.0,
            partial_laps=0.0,
            lap_count=0,
            damage=0.0,
            survived=True,
            wall_contact_seconds=0.0,
            max_speed_mps=0.0,
            first_lap_time_seconds=None,
            best_lap_time_seconds=None,
        )
    )

    records = tuple(json.loads(line) for line in stream.getvalue().splitlines())
    assert [record["record_type"] for record in records] == ["controller_trace_tick", "controller_trace_summary"]
    assert records[-1]["straight_mean_absolute_line_target_m"] == 0.0
    assert records[-1]["strong_line_target_tick_count"] == 0
    assert records[-1]["strong_line_target_mean_directional_offset_m"] == 0.0
    json.dumps(records, allow_nan=False)


def test_trace_loads_preview_controller_from_module_factory() -> None:
    trace = load_tool("trace")

    controller = trace._controller("controllers.race_faster", None, ())

    assert isinstance(controller, PreviewController)
    # Compare against the module's own block rather than a baked literal, so a
    # re-bake of the shipped vector does not fail a test about module loading.
    assert controller.parameters == RACE_FASTER_PARAMETERS


def test_v2_bake_includes_fixed_compensation_context(tmp_path: Path) -> None:
    bake = load_tool("bake")
    checkpoint = tmp_path / "ga.json"
    checkpoint.write_text(
        json.dumps(
            {
                "generation": 40,
                "parameter_names": ["line_turn_sensitivity"],
                "best_parameter_vector": {"line_turn_sensitivity": 0.08},
                "checkpoint_context": {
                    "preview_line_compensation": 0.91,
                    "wall_balance_line_compensation": 0.87,
                },
            }
        ),
        encoding="utf-8",
    )

    block = bake.bake_block(
        checkpoint=checkpoint,
        base=ControllerParameters(),
        preset_base=ControllerParameters(pose_invariant_racing_line=True),
    )

    assert "line_turn_sensitivity=0.08" in block
    assert "preview_line_compensation=0.91" in block
    assert "wall_balance_line_compensation=0.87" in block
    assert "pose_invariant_racing_line=True" in block


def test_generation_checkpoints_are_archived_immutably(tmp_path: Path) -> None:
    search = load_tool("search")
    artifact_root = tmp_path / "faster"
    latest = artifact_root / "checkpoint.json"
    latest.parent.mkdir(parents=True)
    latest.write_text('{"generation":20}\n', encoding="utf-8")

    archived = search.archive_generation_checkpoint(
        checkpoint_path=latest,
        artifact_root=artifact_root,
        generation=20,
    )

    assert archived == artifact_root / "generations" / "generation-020.json"
    assert archived.read_bytes() == latest.read_bytes()
    assert (
        search.archive_generation_checkpoint(
            checkpoint_path=latest,
            artifact_root=artifact_root,
            generation=20,
        )
        == archived
    )
    latest.write_text('{"generation":20,"different":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="different contents"):
        search.archive_generation_checkpoint(
            checkpoint_path=latest,
            artifact_root=artifact_root,
            generation=20,
        )


def test_every_completed_generation_gets_its_own_archive(tmp_path: Path) -> None:
    search = load_tool("search")
    artifact_root = tmp_path / "all-generations"
    checkpoint = artifact_root / "checkpoint.json"
    checkpoint.parent.mkdir(parents=True)

    for generation in range(1, 4):
        checkpoint.write_text(json.dumps({"generation": generation}) + "\n", encoding="utf-8")
        search.archive_generation_checkpoint(
            checkpoint_path=checkpoint,
            artifact_root=artifact_root,
            generation=generation,
        )

    archives = sorted((artifact_root / "generations").glob("generation-*.json"))
    assert [path.name for path in archives] == ["generation-001.json", "generation-002.json", "generation-003.json"]
    assert [json.loads(path.read_text(encoding="utf-8"))["generation"] for path in archives] == [1, 2, 3]


def test_faster_line_v3_unpins_the_v2_bounds_and_stays_inside_the_barrier() -> None:
    search = load_tool("search")

    base, space = search.preset_configuration("faster-line-v3")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}

    assert base.pose_invariant_racing_line
    assert base.pose_invariant_speed_curvature
    assert len(space.specs) == 16
    # Both compensations become inert once the speed scalar is pose-invariant.
    assert "curvature_offset_compensation" not in space.names
    assert "curvature_heading_compensation" not in space.names
    # Every bound the v2 winner finished pinned against must have moved.
    assert bounds["center_steer_gain"][0] == 0.0
    assert bounds["line_turn_sensitivity"][0] < 0.010
    assert bounds["curvature_lateral_ratio"] == (0.02, 0.30)
    assert bounds["maximum_racing_line_offset_ratio"][0] == 0.65

    # Half-track is 3.3 m, the collision hull half-width is 0.63 m, and the
    # barrier inner face sits at 4.7 m, so the body edge reaches the barrier at
    # about 4.07 m of offset. The searchable ceiling must stay well inside that.
    body_edge_m = bounds["maximum_racing_line_offset_ratio"][1] * 3.3 + 0.63
    assert body_edge_m < 4.7 - 0.9, body_edge_m
    for ratio in ("racing_line_offset_ratio", "racing_line_entry_offset_ratio", "racing_line_exit_offset_ratio"):
        assert bounds[ratio][1] >= bounds["maximum_racing_line_offset_ratio"][1]
