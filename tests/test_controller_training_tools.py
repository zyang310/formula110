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
    max_speed_mps: float = 20.0,
    survived: bool = True,
    best_lap_time_seconds: float = 20.0,
    first_lap_time_seconds: float | None = None,
) -> Any:
    return evaluator.SoloTrialResult(
        seed=1,
        elapsed_seconds=30.0,
        raw_distance_m=raw_distance_m,
        partial_laps=raw_distance_m / 181.1,
        lap_count=lap_count,
        damage=damage,
        survived=survived,
        wall_contact_seconds=wall_contact_seconds,
        max_speed_mps=max_speed_mps,
        first_lap_time_seconds=(best_lap_time_seconds if first_lap_time_seconds is None else first_lap_time_seconds),
        best_lap_time_seconds=best_lap_time_seconds,
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


def test_lap_time_score_is_robust_and_keeps_the_hard_safety_tiers() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    clean_fast = _improved_trial(
        evaluator,
        raw_distance_m=650.0,
        best_lap_time_seconds=7.80,
    )
    clean_slow = _improved_trial(
        evaluator,
        raw_distance_m=680.0,
        best_lap_time_seconds=7.95,
    )
    unsafe_fast = _improved_trial(
        evaluator,
        raw_distance_m=700.0,
        damage=0.30,
        best_lap_time_seconds=7.60,
    )
    fast_but_variable = (
        _improved_trial(evaluator, raw_distance_m=680.0, best_lap_time_seconds=7.70),
        _improved_trial(evaluator, raw_distance_m=680.0, best_lap_time_seconds=8.10),
    )
    consistently_fast = (
        _improved_trial(evaluator, raw_distance_m=670.0, best_lap_time_seconds=7.90),
        _improved_trial(evaluator, raw_distance_m=670.0, best_lap_time_seconds=7.90),
    )

    assert search.lap_time_score((clean_fast,), {}) > search.lap_time_score((clean_slow,), {})
    assert search.lap_time_score((clean_slow,), {}) > search.lap_time_score((unsafe_fast,), {})
    assert search.lap_time_score(consistently_fast, {}) > search.lap_time_score(fast_but_variable, {})


def test_lap_time_score_v2_ignores_float_jitter_before_ranking_distribution() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    jitter_winner = (
        _improved_trial(evaluator, raw_distance_m=680.0, best_lap_time_seconds=7.833333333332888),
        _improved_trial(evaluator, raw_distance_m=680.0, best_lap_time_seconds=7.833333333332888),
    )
    meaningful_winner = (
        _improved_trial(evaluator, raw_distance_m=680.0, best_lap_time_seconds=7.833333333333670),
        _improved_trial(evaluator, raw_distance_m=680.0, best_lap_time_seconds=7.816666666666222),
    )

    assert search.lap_time_score(jitter_winner, {}) > search.lap_time_score(meaningful_winner, {})
    assert search.lap_time_score_v2(meaningful_winner, {}) > search.lap_time_score_v2(jitter_winner, {})


def test_lap_time_score_v3_requires_three_laps_and_clean_trials_before_speed() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    clean_slow = _improved_trial(
        evaluator,
        raw_distance_m=680.0,
        lap_count=3,
        best_lap_time_seconds=7.90,
    )
    incident_fast = _improved_trial(
        evaluator,
        raw_distance_m=690.0,
        damage=0.02,
        wall_contact_seconds=0.15,
        lap_count=3,
        best_lap_time_seconds=7.70,
    )
    two_lap_fast = _improved_trial(
        evaluator,
        raw_distance_m=500.0,
        lap_count=2,
        best_lap_time_seconds=7.60,
    )

    assert search.lap_time_score_v3((clean_slow,), {}) > search.lap_time_score_v3((incident_fast,), {})
    assert search.lap_time_score_v3((incident_fast,), {}) > search.lap_time_score_v3((two_lap_fast,), {})


def test_lap_time_score_v4_uses_first_lap_after_repeated_lap_pace() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    clean_start = _improved_trial(
        evaluator,
        raw_distance_m=680.0,
        lap_count=3,
        best_lap_time_seconds=7.80,
        first_lap_time_seconds=8.70,
    )
    correction_start = _improved_trial(
        evaluator,
        raw_distance_m=680.0,
        lap_count=3,
        best_lap_time_seconds=7.80,
        first_lap_time_seconds=9.95,
    )
    slower_repeated_lap = _improved_trial(
        evaluator,
        raw_distance_m=690.0,
        lap_count=3,
        best_lap_time_seconds=7.82,
        first_lap_time_seconds=8.60,
    )

    assert search.lap_time_score_v4((clean_start,), {}) > search.lap_time_score_v4((correction_start,), {})
    assert search.lap_time_score_v4((clean_start,), {}) > search.lap_time_score_v4((slower_repeated_lap,), {})


def test_lap_time_score_v5_rejects_a_tiny_repeated_gain_with_a_slow_launch() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    balanced = _improved_trial(
        evaluator,
        raw_distance_m=684.0,
        lap_count=3,
        best_lap_time_seconds=7.80,
        first_lap_time_seconds=8.90,
    )
    slow_launch = _improved_trial(
        evaluator,
        raw_distance_m=674.0,
        lap_count=3,
        best_lap_time_seconds=7.77,
        first_lap_time_seconds=10.20,
    )

    assert search.lap_time_score_v4((slow_launch,), {}) > search.lap_time_score_v4((balanced,), {})
    assert search.lap_time_score_v5((balanced,), {}) > search.lap_time_score_v5((slow_launch,), {})


def test_lap_time_score_v6_ties_equal_physical_totals_in_integer_ticks() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    parent = _improved_trial(
        evaluator,
        raw_distance_m=684.0,
        lap_count=3,
        best_lap_time_seconds=466 / 60,
        first_lap_time_seconds=535 / 60,
    )
    decimal_artifact = _improved_trial(
        evaluator,
        raw_distance_m=681.0,
        lap_count=3,
        best_lap_time_seconds=470 / 60,
        first_lap_time_seconds=527 / 60,
    )

    # Both totals are exactly 1,467 simulator ticks. Six-decimal component
    # rounding reverses them; integer-tick ranking ties the total and then keeps
    # the parent's faster repeated lap.
    assert search.lap_time_score_v5((decimal_artifact,), {}) > search.lap_time_score_v5((parent,), {})
    parent_score = search.lap_time_score_v6((parent,), {})
    artifact_score = search.lap_time_score_v6((decimal_artifact,), {})
    assert parent_score[5] == artifact_score[5] == -1467.0
    assert parent_score > artifact_score


def test_lap_time_score_v7_weights_first_and_best_laps_equally() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    faster_best = _improved_trial(
        evaluator,
        raw_distance_m=690.0,
        lap_count=3,
        best_lap_time_seconds=450 / 60,
        first_lap_time_seconds=600 / 60,
    )
    balanced = _improved_trial(
        evaluator,
        raw_distance_m=680.0,
        lap_count=3,
        best_lap_time_seconds=465 / 60,
        first_lap_time_seconds=570 / 60,
    )

    # Both candidates tie at 1,500 ticks under first + 2*best.  Equal weighting
    # prefers the balanced candidate's 1,035 ticks over 1,050.
    assert search.lap_time_score_v6((faster_best,), {}) > search.lap_time_score_v6((balanced,), {})
    assert search.lap_time_score_v7((balanced,), {}) > search.lap_time_score_v7((faster_best,), {})


def test_lap_time_score_v8_refuses_to_trade_best_lap_for_first_lap() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    # The exact pair v21 generation 11 chose between: 513 + 457 and 512 + 458
    # are both 970 ticks, so v7's sum key ties and its next key, first lap,
    # picked the candidate whose best lap is a tick slower.
    incumbent = _improved_trial(
        evaluator,
        raw_distance_m=700.0,
        lap_count=3,
        best_lap_time_seconds=457 / 60,
        first_lap_time_seconds=513 / 60,
    )
    first_lap_trade = _improved_trial(
        evaluator,
        raw_distance_m=700.0,
        lap_count=3,
        best_lap_time_seconds=458 / 60,
        first_lap_time_seconds=512 / 60,
    )

    assert search.lap_time_score_v7((first_lap_trade,), {}) > search.lap_time_score_v7((incumbent,), {})
    assert search.lap_time_score_v8((incumbent,), {}) > search.lap_time_score_v8((first_lap_trade,), {})

    # With best lap held equal, v8 still takes the faster first lap.
    faster_first = _improved_trial(
        evaluator,
        raw_distance_m=700.0,
        lap_count=3,
        best_lap_time_seconds=457 / 60,
        first_lap_time_seconds=512 / 60,
    )
    assert search.lap_time_score_v8((faster_first,), {}) > search.lap_time_score_v8((incumbent,), {})


def test_lap_time_score_v9_allows_bounded_damage_but_rejects_excess() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    clean = _improved_trial(
        evaluator,
        raw_distance_m=700.0,
        lap_count=3,
        best_lap_time_seconds=457 / 60,
        first_lap_time_seconds=513 / 60,
    )
    faster_with_brush = _improved_trial(
        evaluator,
        raw_distance_m=705.0,
        damage=0.40,
        wall_contact_seconds=1.80,
        lap_count=3,
        best_lap_time_seconds=456 / 60,
        first_lap_time_seconds=512 / 60,
    )
    excessive = _improved_trial(
        evaluator,
        raw_distance_m=710.0,
        damage=0.51,
        lap_count=3,
        best_lap_time_seconds=455 / 60,
        first_lap_time_seconds=511 / 60,
    )

    assert search.lap_time_score_v8((clean,), {}) > search.lap_time_score_v8((faster_with_brush,), {})
    assert search.lap_time_score_v9((faster_with_brush,), {}) > search.lap_time_score_v9((clean,), {})
    assert search.lap_time_score_v9((clean,), {}) > search.lap_time_score_v9((excessive,), {})


def test_speed_max_score_allows_damage_but_rejects_elimination() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    safe = _improved_trial(evaluator, raw_distance_m=690.0, lap_count=3, max_speed_mps=30.0)
    damaged_fast = _improved_trial(
        evaluator,
        raw_distance_m=650.0,
        damage=0.90,
        wall_contact_seconds=8.0,
        lap_count=3,
        max_speed_mps=36.0,
    )
    eliminated = _improved_trial(
        evaluator,
        raw_distance_m=710.0,
        damage=1.0,
        survived=False,
        lap_count=3,
        max_speed_mps=40.0,
    )

    assert search.speed_max_score_v1((damaged_fast,), {}) > search.speed_max_score_v1((safe,), {})
    assert search.speed_max_score_v1((safe,), {}) > search.speed_max_score_v1((eliminated,), {})


def test_lap_time_score_v10_balances_first_and_best_and_allows_damage() -> None:
    search = load_tool("search")
    evaluator = load_tool("evaluator")
    clean_slow = _improved_trial(
        evaluator,
        raw_distance_m=700.0,
        lap_count=3,
        first_lap_time_seconds=512 / 60,
        best_lap_time_seconds=456 / 60,
    )
    damaged_balanced = _improved_trial(
        evaluator,
        raw_distance_m=710.0,
        damage=0.90,
        wall_contact_seconds=5.0,
        lap_count=3,
        first_lap_time_seconds=500 / 60,
        best_lap_time_seconds=440 / 60,
    )
    best_only_trade = _improved_trial(
        evaluator,
        raw_distance_m=710.0,
        lap_count=3,
        first_lap_time_seconds=530 / 60,
        best_lap_time_seconds=430 / 60,
    )
    eliminated = _improved_trial(
        evaluator,
        raw_distance_m=720.0,
        damage=1.0,
        survived=False,
        lap_count=3,
        first_lap_time_seconds=490 / 60,
        best_lap_time_seconds=420 / 60,
    )

    assert search.lap_time_score_v10((damaged_balanced,), {}) > search.lap_time_score_v10((clean_slow,), {})
    assert search.lap_time_score_v10((damaged_balanced,), {}) > search.lap_time_score_v10((best_only_trade,), {})
    assert search.lap_time_score_v10((clean_slow,), {}) > search.lap_time_score_v10((eliminated,), {})


def test_faster_line_preset_is_isolated_and_uses_planned_bounds() -> None:
    search = load_tool("search")

    existing_names = search.faster_parameter_space().names
    base, space = search.preset_configuration("faster-line")
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


def test_seed_checkpoint_applies_fixed_context_and_searched_vector(tmp_path: Path) -> None:
    search = load_tool("search")
    checkpoint = tmp_path / "parent.json"
    checkpoint.write_text(
        json.dumps(
            {
                "best_parameter_vector": {
                    "startup_speed_cap_mps": 20.25,
                    "startup_speed_cap_seconds": 2.90,
                },
                "checkpoint_context": {
                    "straight_target_speed_mps": 25.075,
                    "front_brake_start_m": 11.919,
                    "line_target_slew_per_tick": 0.02765,
                },
            }
        ),
        encoding="utf-8",
    )

    base, space = search.preset_configuration("faster-line-v12", seed_checkpoint=checkpoint)
    initial = dict(zip(space.names, space.initial_mean, strict=True))

    assert base.straight_target_speed_mps == 25.075
    assert base.front_brake_start_m == 11.919
    assert base.line_target_slew_per_tick == 0.02765
    assert initial["startup_speed_cap_mps"] == 20.25
    assert initial["startup_speed_cap_seconds"] == 2.90


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


def test_faster_line_v4_raises_the_v3_ceilings_and_adds_target_release() -> None:
    search = load_tool("search")

    base, space = search.preset_configuration("faster-line-v4")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    initial = {spec.name: spec.initial for spec in space.specs}

    assert len(space.specs) == 17
    assert base.pose_invariant_racing_line
    assert base.pose_invariant_speed_curvature
    # Generation 0 must reproduce v3: the release rate starts at the outward slew.
    assert initial["line_target_release_per_tick"] == initial["line_target_slew_per_tick"]
    assert bounds["line_target_release_per_tick"][0] < 0.005

    # Every ceiling v3 pinned against must have moved up, and every floor down.
    assert bounds["throttle_gain"][1] > 2.00
    assert bounds["curvature_lateral_ratio"][1] > 0.30
    assert bounds["maximum_racing_line_offset_ratio"][1] > 0.90
    assert bounds["racing_line_offset_ratio"][1] > 0.90
    assert bounds["racing_line_entry_offset_ratio"][1] > 0.90
    assert bounds["heading_steer_gain"][0] < 0.40

    # Half-track 3.3 m, hull half-width 0.63 m, barrier inner face 4.7 m, so the
    # body edge reaches the barrier near 4.07 m of offset. Keep real margin.
    body_edge_m = bounds["maximum_racing_line_offset_ratio"][1] * 3.3 + 0.63
    assert body_edge_m < 4.07 - 0.25, body_edge_m
    for ratio in ("racing_line_offset_ratio", "racing_line_entry_offset_ratio", "racing_line_exit_offset_ratio"):
        assert bounds[ratio][1] >= bounds["maximum_racing_line_offset_ratio"][1]


def test_faster_line_v5_restores_wall_margin_and_holds_the_line_clamp() -> None:
    search = load_tool("search")

    _base, space = search.preset_configuration("faster-line-v5")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}

    assert len(space.specs) == 17
    # A seed-110 sweep showed a high retraction threshold oscillates the target
    # rather than protecting it: at 2.0 the trial collapses to 234 m with 728
    # AVOID ticks. The search must stay free to pick a low value.
    assert bounds["line_clearance_m"][0] == 0.0

    # The line clamp deliberately keeps v4's ceiling. v4 reached 3.77 m of
    # offset, putting the body edge at 4.40 m against a barrier at 4.70 m, so
    # widening further buys contact rather than lap time.
    assert bounds["maximum_racing_line_offset_ratio"][1] == 0.95
    for ratio in ("racing_line_offset_ratio", "racing_line_entry_offset_ratio", "racing_line_exit_offset_ratio"):
        assert bounds[ratio][1] == 0.95

    # The genes v4 pinned that are not wall-safety limits do get more room.
    assert bounds["curvature_lateral_ratio"][1] > 0.60
    assert bounds["heading_steer_gain"][0] < 0.15
    assert bounds["throttle_gain"][1] > 4.00


def test_faster_line_v6_adds_structural_variation_without_widening_the_line_clamp() -> None:
    search = load_tool("search")
    bake = load_tool("bake")

    base, space = search.preset_configuration("faster-line-v6")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    initial = {spec.name: spec.initial for spec in space.specs}

    assert len(space.specs) == 20
    assert base.pose_invariant_racing_line
    assert base.pose_invariant_speed_curvature
    assert base.maximum_racing_line_offset_ratio == 0.95
    assert "maximum_racing_line_offset_ratio" not in space.names

    # V6 changes the available behaviours instead of only inflating mutation.
    for name in (
        "yaw_damping_gain",
        "steer_slew_per_tick",
        "curvature_heading_degrees",
        "yaw_speed_reduction",
    ):
        assert name in space.names
        spec = next(spec for spec in space.specs if spec.name == name)
        assert initial[name] == spec.clamp(getattr(base, name))

    # Reopen only the non-geometric bounds that the v5 winner pressed.
    assert bounds["curvature_lateral_ratio"][1] > 1.20
    assert bounds["heading_steer_gain"][0] < 0.05
    assert bounds["line_turn_sensitivity"][0] < 0.002
    assert bounds["line_target_release_per_tick"][1] > 0.25
    assert bounds["line_clearance_m"][0] == 0.0
    for ratio in ("racing_line_offset_ratio", "racing_line_entry_offset_ratio", "racing_line_exit_offset_ratio"):
        assert bounds[ratio][1] == base.maximum_racing_line_offset_ratio

    # Searched steering dynamics must not also be frozen into GA checkpoint
    # context; bake still carries the two fixed line-frame compensations.
    assert search._checkpoint_context("faster-line-v6", base) == {
        "preview_line_compensation": base.preview_line_compensation,
        "wall_balance_line_compensation": base.wall_balance_line_compensation,
    }
    assert search.parse_args(["faster-line-v6"]).preset == "faster-line-v6"
    assert bake.parse_args(["--preset", "faster-line-v6"]).preset == "faster-line-v6"


def test_faster_line_v7_targets_the_measured_wall_speed_bottleneck() -> None:
    search = load_tool("search")
    bake = load_tool("bake")

    base, space = search.preset_configuration("faster-line-v7")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}

    assert len(space.specs) == 16
    assert base.pose_invariant_racing_line
    assert base.pose_invariant_speed_curvature
    assert base.maximum_racing_line_offset_ratio == 0.95
    for name in (
        "side_slow_start_m",
        "side_speed_floor",
        "avoid_front_wall_m",
        "avoid_diagonal_wall_m",
        "avoid_side_wall_m",
        "avoid_speed_mps",
        "avoid_steer_gain",
    ):
        assert name in space.names
    for name in (
        "heading_steer_gain",
        "center_steer_gain",
        "yaw_damping_gain",
        "steer_slew_per_tick",
        "racing_line_offset_ratio",
        "line_target_slew_per_tick",
    ):
        assert name not in space.names

    # The clean diagnostic values are admitted, while the failed side-speed
    # floor of 0.8 is deliberately outside the box.
    assert bounds["avoid_front_wall_m"][0] <= 2.5
    assert bounds["avoid_diagonal_wall_m"][0] <= 1.2
    assert bounds["avoid_side_wall_m"][0] <= 0.7
    assert bounds["avoid_speed_mps"][1] >= 8.0
    assert bounds["side_speed_floor"][1] < 0.8
    assert not set(search._checkpoint_context("faster-line-v7", base)) & set(space.names)

    args = search.parse_args(["faster-line-v7", "--objective", "lap-time"])
    assert args.preset == "faster-line-v7"
    assert args.objective == "lap-time"
    assert bake.parse_args(["--preset", "faster-line-v7"]).preset == "faster-line-v7"


def test_faster_line_v8_moves_v7s_pressed_bounds_and_drops_zeroed_speed_genes() -> None:
    search = load_tool("search")
    bake = load_tool("bake")

    base, space = search.preset_configuration("faster-line-v8")
    v7_base, v7_space = search.preset_configuration("faster-line-v7")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    v7_bounds = {spec.name: (spec.minimum, spec.maximum) for spec in v7_space.specs}

    assert len(space.specs) == 14
    assert base.maximum_racing_line_offset_ratio == 0.95
    assert "steering_speed_reduction" not in space.names
    assert "yaw_speed_reduction" not in space.names
    assert bounds["throttle_gain"][1] > v7_bounds["throttle_gain"][1]
    assert bounds["curvature_lateral_ratio"][1] > v7_bounds["curvature_lateral_ratio"][1]
    assert bounds["avoid_diagonal_wall_m"][0] < v7_bounds["avoid_diagonal_wall_m"][0]
    assert bounds["avoid_side_wall_m"][0] < v7_bounds["avoid_side_wall_m"][0]
    assert search._checkpoint_context("faster-line-v8", base)["steering_speed_reduction"] == (
        v7_base.steering_speed_reduction
    )

    args = search.parse_args(["faster-line-v8", "--objective", "lap-time-v2"])
    assert args.preset == "faster-line-v8"
    assert args.objective == "lap-time-v2"
    assert bake.parse_args(["--preset", "faster-line-v8"]).preset == "faster-line-v8"


def test_faster_line_v9_restores_a_safe_side_threshold_and_consistency_objective() -> None:
    search = load_tool("search")
    bake = load_tool("bake")

    base, space = search.preset_configuration("faster-line-v9")
    v8_base, v8_space = search.preset_configuration("faster-line-v8")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    v8_bounds = {spec.name: (spec.minimum, spec.maximum) for spec in v8_space.specs}

    assert len(space.specs) == 14
    assert base.maximum_racing_line_offset_ratio == 0.95
    assert bounds["avoid_side_wall_m"][0] > v8_bounds["avoid_side_wall_m"][0]
    assert bounds["throttle_gain"][1] > v8_bounds["throttle_gain"][1]
    assert not set(search._checkpoint_context("faster-line-v9", base)) & set(space.names)
    assert search._checkpoint_context("faster-line-v9", base)["steering_speed_reduction"] == (
        v8_base.steering_speed_reduction
    )

    args = search.parse_args(["faster-line-v9", "--objective", "lap-time-v3"])
    assert args.preset == "faster-line-v9"
    assert args.objective == "lap-time-v3"
    assert bake.parse_args(["--preset", "faster-line-v9"]).preset == "faster-line-v9"


def test_faster_line_v10_searches_the_clean_first_corner_and_official_seeds() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v10")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    manifest = seeds.generate_seed_manifest()

    assert len(space.specs) == 16
    assert base.startup_speed_cap_mps == 19.0
    assert base.startup_speed_cap_seconds == 3.5
    assert bounds["startup_speed_cap_mps"] == (16.0, 24.0)
    assert bounds["startup_speed_cap_seconds"] == (2.5, 5.0)
    assert "steering_speed_reduction" in space.names
    assert "yaw_damping_gain" in space.names
    assert "line_target_slew_per_tick" not in space.names
    context = search._checkpoint_context("faster-line-v10", base)
    assert not set(context) & set(space.names)
    assert context["straight_target_speed_mps"] == base.straight_target_speed_mps
    assert context["front_brake_start_m"] == base.front_brake_start_m

    assert search._full_evaluation_seeds("faster-line-v10", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v10", manifest, 0)[-2:] == manifest.official
    assert search._full_evaluation_seeds("faster-line-v9", manifest) == manifest.training

    args = search.parse_args(["faster-line-v10", "--objective", "lap-time-v4"])
    assert args.preset == "faster-line-v10"
    assert args.objective == "lap-time-v4"
    assert bake.parse_args(["--preset", "faster-line-v10"]).preset == "faster-line-v10"


def test_faster_line_v11_keeps_v10s_box_but_ranks_total_race_time() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v11")
    v10_base, v10_space = search.preset_configuration("faster-line-v10")
    manifest = seeds.generate_seed_manifest()

    assert base == v10_base
    assert space == v10_space
    assert search._full_evaluation_seeds("faster-line-v11", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v11", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v11", "--objective", "lap-time-v5"])
    assert args.preset == "faster-line-v11"
    assert args.objective == "lap-time-v5"
    assert bake.parse_args(["--preset", "faster-line-v11"]).preset == "faster-line-v11"


def test_faster_line_v12_keeps_the_box_and_uses_tick_ranking() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v12")
    v11_base, v11_space = search.preset_configuration("faster-line-v11")
    manifest = seeds.generate_seed_manifest()

    assert base == v11_base
    assert space == v11_space
    assert search._full_evaluation_seeds("faster-line-v12", manifest) == manifest.training + manifest.official
    args = search.parse_args(["faster-line-v12", "--objective", "lap-time-v6"])
    assert args.preset == "faster-line-v12"
    assert args.objective == "lap-time-v6"
    assert bake.parse_args(["--preset", "faster-line-v12"]).preset == "faster-line-v12"


def test_faster_line_v13_keeps_tick_ranking_after_context_seed_fix() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v13")
    v12_base, v12_space = search.preset_configuration("faster-line-v12")
    manifest = seeds.generate_seed_manifest()

    assert base == v12_base
    assert space == v12_space
    assert search._full_evaluation_seeds("faster-line-v13", manifest) == manifest.training + manifest.official
    args = search.parse_args(["faster-line-v13", "--objective", "lap-time-v6"])
    assert args.preset == "faster-line-v13"
    assert args.objective == "lap-time-v6"
    assert bake.parse_args(["--preset", "faster-line-v13"]).preset == "faster-line-v13"


def test_faster_line_v14_targets_only_the_validated_sweeper_speed_bonus() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v14")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    manifest = seeds.generate_seed_manifest()

    assert space.names == (
        "sweeper_minimum_duration_s",
        "sweeper_speed_hold_seconds",
        "sweeper_target_speed_bonus_mps",
    )
    assert base.sweeper_minimum_duration_s == 1.7
    assert base.sweeper_speed_hold_seconds == 0.9
    assert base.sweeper_target_speed_bonus_mps == 1.5
    assert bounds["sweeper_minimum_duration_s"] == (1.25, 2.20)
    assert bounds["sweeper_speed_hold_seconds"] == (0.15, 1.20)
    assert bounds["sweeper_target_speed_bonus_mps"] == (0.10, 3.00)
    assert not set(search._checkpoint_context("faster-line-v14", base)) & set(space.names)
    assert search._full_evaluation_seeds("faster-line-v14", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v14", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v14", "--objective", "lap-time-v6"])
    assert args.preset == "faster-line-v14"
    assert args.objective == "lap-time-v6"
    assert bake.parse_args(["--preset", "faster-line-v14"]).preset == "faster-line-v14"


def test_faster_line_v15_targets_the_previewed_sweeper_entry() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v15")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    manifest = seeds.generate_seed_manifest()

    assert space.names == (
        "sweeper_preview_minimum_far_curvature",
        "sweeper_preview_maximum_far_curvature",
        "sweeper_preview_speed_hold_seconds",
        "sweeper_preview_target_speed_bonus_mps",
    )
    assert base.sweeper_preview_minimum_far_curvature == 0.10
    assert base.sweeper_preview_maximum_far_curvature == 0.14
    assert base.sweeper_preview_speed_hold_seconds == 2.30
    assert base.sweeper_preview_target_speed_bonus_mps == 0.0
    assert bounds["sweeper_preview_minimum_far_curvature"] == (0.07, 0.13)
    assert bounds["sweeper_preview_maximum_far_curvature"] == (0.11, 0.18)
    assert bounds["sweeper_preview_speed_hold_seconds"] == (0.50, 3.00)
    assert bounds["sweeper_preview_target_speed_bonus_mps"] == (0.0, 4.0)
    assert not set(search._checkpoint_context("faster-line-v15", base)) & set(space.names)
    assert search._full_evaluation_seeds("faster-line-v15", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v15", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v15", "--objective", "lap-time-v6"])
    assert args.preset == "faster-line-v15"
    assert args.objective == "lap-time-v6"
    assert bake.parse_args(["--preset", "faster-line-v15"]).preset == "faster-line-v15"


def test_faster_line_v16_reopens_the_launch_duration_floor() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v16")
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    manifest = seeds.generate_seed_manifest()

    assert space.names == ("startup_speed_cap_mps", "startup_speed_cap_seconds")
    assert bounds["startup_speed_cap_mps"] == (21.5, 22.9)
    assert bounds["startup_speed_cap_seconds"] == (1.85, 2.50)
    assert not set(search._checkpoint_context("faster-line-v16", base)) & set(space.names)
    assert search._full_evaluation_seeds("faster-line-v16", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v16", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v16", "--objective", "lap-time-v6"])
    assert args.preset == "faster-line-v16"
    assert args.objective == "lap-time-v6"
    assert bake.parse_args(["--preset", "faster-line-v16"]).preset == "faster-line-v16"


def test_faster_line_v17_jointly_tunes_launch_and_preview_under_equal_lap_weight() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v17")
    manifest = seeds.generate_seed_manifest()

    assert space.names == (
        "sweeper_preview_minimum_far_curvature",
        "sweeper_preview_maximum_far_curvature",
        "sweeper_preview_speed_hold_seconds",
        "sweeper_preview_target_speed_bonus_mps",
        "startup_speed_cap_mps",
        "startup_speed_cap_seconds",
    )
    assert not set(search._checkpoint_context("faster-line-v17", base)) & set(space.names)
    assert search._full_evaluation_seeds("faster-line-v17", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v17", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v17", "--objective", "lap-time-v7"])
    assert args.preset == "faster-line-v17"
    assert args.objective == "lap-time-v7"
    assert bake.parse_args(["--preset", "faster-line-v17"]).preset == "faster-line-v17"


def test_faster_line_v18_targets_only_the_corner_exit_bonus() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v18")
    manifest = seeds.generate_seed_manifest()

    assert space.names == ("corner_exit_target_speed_bonus_mps",)
    assert base.corner_exit_target_speed_bonus_mps == 0.0
    assert not set(search._checkpoint_context("faster-line-v18", base)) & set(space.names)
    assert search._full_evaluation_seeds("faster-line-v18", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v18", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v18", "--objective", "lap-time-v7"])
    assert args.preset == "faster-line-v18"
    assert args.objective == "lap-time-v7"
    assert bake.parse_args(["--preset", "faster-line-v18"]).preset == "faster-line-v18"


def test_faster_line_v19_reopens_the_two_pinned_bounds() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v19")
    manifest = seeds.generate_seed_manifest()

    assert space.names == ("corner_target_speed_mps", "front_stop_m")
    # V13's elites pinned both genes exactly on the bound v10 gave them; the new
    # box must actually clear those edges or the reopening is a no-op.
    corner, front = space.specs
    assert corner.minimum < 14.0
    assert front.maximum > 1.60
    assert not set(search._checkpoint_context("faster-line-v19", base)) & set(space.names)
    # The v18 lever stays fixed rather than silently reverting to its default.
    assert "corner_exit_target_speed_bonus_mps" in search._checkpoint_context("faster-line-v19", base)
    assert search._full_evaluation_seeds("faster-line-v19", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v19", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v19", "--objective", "lap-time-v7"])
    assert args.preset == "faster-line-v19"
    assert args.objective == "lap-time-v7"
    assert bake.parse_args(["--preset", "faster-line-v19"]).preset == "faster-line-v19"


def test_faster_line_v20_reopens_the_launch_box_on_both_sides() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v20")
    manifest = seeds.generate_seed_manifest()

    assert space.names == ("startup_speed_cap_mps", "startup_speed_cap_seconds")
    # V16's box was 21.5-22.9 m/s over 1.85-2.50 s; v20 must clear it on the two
    # sides its elites pushed against, or the reopening is a no-op.
    cap, hold = space.specs
    assert cap.maximum > 22.9
    assert hold.minimum < 1.85
    context = search._checkpoint_context("faster-line-v20", base)
    assert not set(context) & set(space.names)
    # V19's two winning genes stay fixed rather than reverting to the v13 box.
    assert "corner_target_speed_mps" in context
    assert "front_stop_m" in context
    assert search._full_evaluation_seeds("faster-line-v20", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v20", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v20", "--objective", "lap-time-v7"])
    assert args.preset == "faster-line-v20"
    assert args.objective == "lap-time-v7"
    assert bake.parse_args(["--preset", "faster-line-v20"]).preset == "faster-line-v20"


def test_faster_line_v21_retests_the_straight_speed_ceiling() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v21")
    manifest = seeds.generate_seed_manifest()

    assert space.names == ("straight_target_speed_mps", "front_brake_start_m")
    # D-040 rejected 26-27 m/s under the old corner approach; the retest is
    # meaningless unless the box actually reaches that range again.
    straight, brake = space.specs
    assert straight.maximum > 27.0
    # A higher straight target only survives if the ramp may start further out
    # than v4's 14.0 m ceiling.
    assert brake.maximum > 14.0
    context = search._checkpoint_context("faster-line-v21", base)
    assert not set(context) & set(space.names)
    # V19's corner approach and v16's launch both stay fixed, since the whole
    # hypothesis is that they are what makes a higher straight target survivable.
    for name in ("corner_target_speed_mps", "front_stop_m", "startup_speed_cap_mps"):
        assert name in context
    assert search._full_evaluation_seeds("faster-line-v21", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v21", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v21", "--objective", "lap-time-v7"])
    assert args.preset == "faster-line-v21"
    assert args.objective == "lap-time-v7"
    assert bake.parse_args(["--preset", "faster-line-v21"]).preset == "faster-line-v21"


def test_faster_line_v22_searches_the_speed_profile_under_best_lap_ranking() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v22")
    manifest = seeds.generate_seed_manifest()

    assert space.names == (
        "straight_target_speed_mps",
        "corner_target_speed_mps",
        "front_brake_start_m",
        "front_stop_m",
    )
    # The reopened bounds v19 and v21 established must survive into the joint
    # space, or v22 re-imposes the boxes those runs disproved.
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    assert bounds["corner_target_speed_mps"][0] < 14.0
    assert bounds["front_stop_m"][1] > 1.60
    assert bounds["front_brake_start_m"][1] > 14.0
    assert bounds["straight_target_speed_mps"][1] > 27.0
    context = search._checkpoint_context("faster-line-v22", base)
    assert not set(context) & set(space.names)
    # The launch stays fixed: v20 proved it is already at its floor.
    assert "startup_speed_cap_mps" in context
    assert search._full_evaluation_seeds("faster-line-v22", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v22", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v22", "--objective", "lap-time-v8"])
    assert args.preset == "faster-line-v22"
    assert args.objective == "lap-time-v8"
    assert bake.parse_args(["--preset", "faster-line-v22"]).preset == "faster-line-v22"


def test_faster_line_v23_reopens_the_pinned_line_timing_bounds() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v23")
    manifest = seeds.generate_seed_manifest()

    assert space.names == ("line_turn_sensitivity", "line_target_release_per_tick")
    # Both genes finished exactly on a bound of the box that last searched them:
    # sensitivity on its 0.002 floor and release on its 0.25 ceiling.
    sensitivity, release = space.specs
    assert sensitivity.minimum < 0.002
    assert release.maximum > 0.25
    context = search._checkpoint_context("faster-line-v23", base)
    assert not set(context) & set(space.names)
    # The v19 corner approach stays fixed; v23 changes line timing, not pace.
    for name in ("corner_target_speed_mps", "front_stop_m", "straight_target_speed_mps"):
        assert name in context
    assert search._full_evaluation_seeds("faster-line-v23", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v23", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v23", "--objective", "lap-time-v8"])
    assert args.preset == "faster-line-v23"
    assert args.objective == "lap-time-v8"
    assert bake.parse_args(["--preset", "faster-line-v23"]).preset == "faster-line-v23"


def test_faster_line_v24_searches_opening_drift_and_real_speed_caps() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v24")
    manifest = seeds.generate_seed_manifest()
    assert space.names == (
        "straight_target_speed_mps",
        "startup_speed_cap_mps",
        "startup_drift_brake",
        "startup_drift_trigger_front_m",
        "startup_drift_minimum_steer",
        "startup_drift_pulse_seconds",
        "startup_drift_steer_gain",
        "startup_drift_straighten_seconds",
    )
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    assert bounds["straight_target_speed_mps"][1] > 30.0
    assert bounds["startup_speed_cap_mps"][1] > 24.5
    assert bounds["startup_drift_brake"][0] == 0.0
    context = search._checkpoint_context("faster-line-v24", base)
    assert not set(context) & set(space.names)
    assert "line_turn_sensitivity" in context
    assert "startup_drift_window_seconds" in context
    assert search._full_evaluation_seeds("faster-line-v24", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v24", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v24", "--objective", "lap-time-v8"])
    assert args.preset == "faster-line-v24"
    assert args.objective == "lap-time-v8"
    assert bake.parse_args(["--preset", "faster-line-v24"]).preset == "faster-line-v24"


def test_faster_line_v25_searches_local_corridor_speed_under_relaxed_damage() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v25")
    manifest = seeds.generate_seed_manifest()
    assert space.names == (
        "long_straight_minimum_duration_s",
        "long_straight_maximum_local_curvature",
        "long_straight_speed_bonus_seconds",
        "long_straight_target_speed_bonus_mps",
        "startup_drift_brake",
        "startup_drift_trigger_front_m",
        "startup_drift_minimum_steer",
        "startup_drift_pulse_seconds",
        "startup_drift_steer_gain",
    )
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    assert bounds["long_straight_target_speed_bonus_mps"] == (0.0, 6.0)
    context = search._checkpoint_context("faster-line-v25", base)
    assert not set(context) & set(space.names)
    assert "straight_target_speed_mps" in context
    assert "startup_speed_cap_mps" in context
    assert search._full_evaluation_seeds("faster-line-v25", manifest) == manifest.training + manifest.official
    assert search._selection_evaluation_seeds("faster-line-v25", manifest, 0)[-2:] == manifest.official
    args = search.parse_args(["faster-line-v25", "--objective", "lap-time-v9"])
    assert args.preset == "faster-line-v25"
    assert args.objective == "lap-time-v9"
    assert bake.parse_args(["--preset", "faster-line-v25"]).preset == "faster-line-v25"


def test_faster_line_v26_searches_the_live_full_throttle_boundary() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v26")
    manifest = seeds.generate_seed_manifest()
    assert space.names == (
        "straight_target_speed_mps",
        "long_straight_minimum_duration_s",
        "long_straight_maximum_local_curvature",
        "long_straight_speed_bonus_seconds",
        "long_straight_target_speed_bonus_mps",
        "front_brake_start_m",
        "front_stop_m",
        "steering_speed_reduction",
    )
    bounds = {spec.name: (spec.minimum, spec.maximum) for spec in space.specs}
    assert bounds["straight_target_speed_mps"][1] == 45.0
    assert bounds["long_straight_speed_bonus_seconds"][1] >= 3.0
    assert bounds["long_straight_target_speed_bonus_mps"][1] >= 20.0
    context = search._checkpoint_context("faster-line-v26", base)
    assert not set(context) & set(space.names)
    assert "startup_drift_brake" in context
    assert search._full_evaluation_seeds("faster-line-v26", manifest) == search.GRADESCOPE_SPEED_SEEDS
    assert search._selection_evaluation_seeds("faster-line-v26", manifest, 0) == search.GRADESCOPE_SPEED_SEEDS
    args = search.parse_args(["faster-line-v26", "--objective", "speed-max-v1"])
    assert args.preset == "faster-line-v26"
    assert args.objective == "speed-max-v1"
    assert bake.parse_args(["--preset", "faster-line-v26"]).preset == "faster-line-v26"


def test_faster_line_v27_searches_ballistic_corridor_and_independent_drift() -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")

    base, space = search.preset_configuration("faster-line-v27")
    manifest = seeds.generate_seed_manifest()
    assert space.names == (
        "long_straight_minimum_duration_s",
        "long_straight_maximum_local_curvature",
        "long_straight_speed_bonus_seconds",
        "long_straight_target_speed_bonus_mps",
        "long_straight_drift_brake",
        "long_straight_drift_minimum_steer",
        "long_straight_drift_pulse_seconds",
        "long_straight_drift_override_after_seconds",
    )
    assert base.long_straight_target_speed_bonus_mps == 8.0
    assert base.long_straight_drift_brake == 0.90
    assert base.long_straight_drift_override_after_seconds == pytest.approx(10.0 / 3.0)
    assert base.startup_speed_cap_mps == RACE_FASTER_PARAMETERS.startup_speed_cap_mps
    assert base.sweeper_target_speed_bonus_mps == RACE_FASTER_PARAMETERS.sweeper_target_speed_bonus_mps
    assert base.sweeper_preview_target_speed_bonus_mps == RACE_FASTER_PARAMETERS.sweeper_preview_target_speed_bonus_mps
    context = search._checkpoint_context("faster-line-v27", base)
    assert not set(context) & set(space.names)
    assert context["startup_drift_brake"] == RACE_FASTER_PARAMETERS.startup_drift_brake
    expected_full = tuple(dict.fromkeys((*manifest.training, *search.GRADESCOPE_SPEED_SEEDS)))
    expected_selection = tuple(
        dict.fromkeys((*search.rotating_training_seeds(manifest.training, 0), *search.GRADESCOPE_SPEED_SEEDS))
    )
    assert search._full_evaluation_seeds("faster-line-v27", manifest) == expected_full
    assert search._selection_evaluation_seeds("faster-line-v27", manifest, 0) == expected_selection
    args = search.parse_args(["faster-line-v27", "--objective", "lap-time-v10"])
    assert args.preset == "faster-line-v27"
    assert args.objective == "lap-time-v10"
    assert bake.parse_args(["--preset", "faster-line-v27"]).preset == "faster-line-v27"


def test_faster_line_v28_reopens_only_v27s_pinned_activation_delay(tmp_path: Path) -> None:
    search = load_tool("search")
    bake = load_tool("bake")
    seeds = load_tool("seeds")
    checkpoint = tmp_path / "v27.json"
    checkpoint.write_text(
        json.dumps(
            {
                "best_parameter_vector": {
                    "long_straight_minimum_duration_s": 0.2551447212626662,
                    "long_straight_maximum_local_curvature": 0.005558362853538455,
                    "long_straight_speed_bonus_seconds": 0.4704340603764563,
                    "long_straight_target_speed_bonus_mps": 7.816043512736939,
                    "long_straight_drift_brake": 0.9837796506105618,
                    "long_straight_drift_minimum_steer": 0.06908986688717965,
                    "long_straight_drift_pulse_seconds": 0.1507770210285232,
                    "long_straight_drift_override_after_seconds": 3.55,
                }
            }
        ),
        encoding="utf-8",
    )

    base, space = search.preset_configuration("faster-line-v28", seed_checkpoint=checkpoint)
    manifest = seeds.generate_seed_manifest()
    assert space.names == ("long_straight_drift_override_after_seconds",)
    assert space.initial_mean == (3.55,)
    assert space.specs[0].maximum > 3.55
    context = search._checkpoint_context("faster-line-v28", base)
    assert not set(context) & set(space.names)
    assert context["long_straight_target_speed_bonus_mps"] == pytest.approx(7.816043512736939)
    assert context["long_straight_drift_brake"] == pytest.approx(0.9837796506105618)
    expected_full = tuple(dict.fromkeys((*manifest.training, *search.GRADESCOPE_SPEED_SEEDS)))
    assert search._full_evaluation_seeds("faster-line-v28", manifest) == expected_full
    args = search.parse_args(["faster-line-v28", "--objective", "lap-time-v10"])
    assert args.preset == "faster-line-v28"
    assert bake.parse_args(["--preset", "faster-line-v28"]).preset == "faster-line-v28"
