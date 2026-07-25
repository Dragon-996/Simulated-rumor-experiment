"""Run the corrected V2 ABM baselines on the four fixed networks."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping

from abm_baseline import load_network
from abm_baseline_v2 import RBEABMSimulationV2, V2SimulationParameters


ROUND_METRICS = (
    "active_spreaders_ratio",
    "silent_believers_ratio",
    "active_skeptics_ratio",
    "passive_skeptics_ratio",
    "believers_ratio",
    "skeptics_ratio",
    "penetration_rate",
    "system_entropy",
    "bipolarization_index",
    "focal_post_ratio",
    "silent_ratio",
    "ever_believed_ratio",
)

RUN_METRICS = (
    "peak_penetration_rate",
    "final_penetration_rate",
    "time_to_peak",
    "penetration_auc",
    "peak_system_entropy",
    "final_system_entropy",
    "final_ever_believed_ratio",
)

SEED_DIAGNOSTICS = (
    "seed_degree_sum",
    "unique_nonseed_neighbors",
    "initial_neighbor_coverage_ratio",
)

T_CRITICAL_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if degrees_of_freedom <= 30:
        return T_CRITICAL_975[degrees_of_freedom]
    if degrees_of_freedom <= 40:
        return 2.021
    if degrees_of_freedom <= 60:
        return 2.000
    if degrees_of_freedom <= 120:
        return 1.980
    return 1.960


def calculate_statistics(values: Iterable[float]) -> Mapping[str, float | int]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("Cannot calculate statistics for an empty sequence")
    mean = fmean(data)
    if len(data) == 1:
        sample_sd = 0.0
        ci_error = 0.0
    else:
        sample_sd = stdev(data)
        ci_error = t_critical_975(len(data) - 1) * sample_sd / len(data) ** 0.5
    return {
        "n": len(data),
        "mean": mean,
        "std": sample_sd,
        "ci_lower": mean - ci_error,
        "ci_upper": mean + ci_error,
        "ci_error": ci_error,
    }


def summarize_condition(run_results: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not run_results:
        raise ValueError("A condition must contain at least one run")
    trajectories = [
        [result["initial_summary"]]
        + [round_result["summary"] for round_result in result["rounds"]]
        for result in run_results
    ]
    round_statistics: dict[str, Any] = {}
    for position, point in enumerate(trajectories[0]):
        round_statistics[f"Round_{point['round']}"] = {
            metric: calculate_statistics(
                trajectory[position][metric] for trajectory in trajectories
            )
            for metric in ROUND_METRICS
        }
    return {
        "network_file": run_results[0]["network_file"],
        "network_metrics": run_results[0]["network_metrics"],
        "model_family": run_results[0]["model_family"],
        "model_variant": run_results[0]["model_variant"],
        "seed_strategy": run_results[0]["seed_strategy"],
        "seed_node_sets": [result["seed_nodes"] for result in run_results],
        "seed_diagnostic_statistics": {
            metric: calculate_statistics(
                result["seed_diagnostics"][metric] for result in run_results
            )
            for metric in SEED_DIAGNOSTICS
        },
        "heterogeneity_strength": run_results[0]["heterogeneity_strength"],
        "population_trait_means": run_results[0]["population_trait_means"],
        "parameters": run_results[0]["parameters"],
        "total_runs": len(run_results),
        "run_seeds": [result["run_seed"] for result in run_results],
        "round_statistics": round_statistics,
        "run_metric_statistics": {
            metric: calculate_statistics(
                result["run_metrics"][metric] for result in run_results
            )
            for metric in RUN_METRICS
        },
        "raw_run_metrics": [result["run_metrics"] for result in run_results],
    }


def write_json(file_path: Path, data: Mapping[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def paired_seed_sets(
    agent_ids: tuple[str, ...], seed_count: int, num_runs: int, seed_base: int
) -> list[tuple[str, ...]]:
    """Create one identical node-ID seed set per replicate for all networks."""
    return [
        tuple(random.Random(seed_base + index).sample(agent_ids, seed_count))
        for index in range(num_runs)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected V2 ABM baselines.")
    parser.add_argument("--config", default="abm_parameters_v2.json")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--save-agent-traces", action="store_true")
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if args.runs is not None:
        config["num_runs"] = args.runs
    if args.rounds is not None:
        config["rounds"] = args.rounds
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.save_agent_traces:
        config["save_agent_traces"] = True

    num_runs = int(config["num_runs"])
    if num_runs <= 0:
        raise ValueError("num_runs must be positive")
    model_variants = list(config["model_variants"])
    if set(model_variants) != {"ABM-0", "ABM-H"}:
        raise ValueError("model_variants must contain ABM-0 and ABM-H")
    seed_strategies = list(config["seed_strategies"])
    if not seed_strategies or not set(seed_strategies) <= {
        "structural", "paired_random"
    }:
        raise ValueError("seed_strategies may contain structural and paired_random")

    parameters = V2SimulationParameters.from_mapping(config)
    config_dir = config_path.parent
    output_root = Path(config["output_dir"])
    if not output_root.is_absolute():
        output_root = config_dir / output_root
    experiment_id = datetime.now().strftime("experiment_%Y%m%d-%H%M%S")
    experiment_dir = output_root / experiment_id
    raw_dir = experiment_dir / "raw_runs"
    experiment_dir.mkdir(parents=True, exist_ok=False)

    networks: dict[str, Any] = {}
    reference_signature: str | None = None
    reference_ids: tuple[str, ...] | None = None
    for name, relative_path in config["network_files"].items():
        network_path = Path(relative_path)
        if not network_path.is_absolute():
            network_path = config_dir / network_path
        network = load_network(network_path)
        if reference_signature is None:
            reference_signature = network.attribute_signature
            reference_ids = tuple(network.agent_ids)
        elif network.attribute_signature != reference_signature:
            raise ValueError("Agent attributes differ across network files")
        if set(network.agent_ids) != set(reference_ids or ()):
            raise ValueError("Agent IDs differ across network files")
        networks[name] = network

    assert reference_ids is not None
    seed_count = max(1, int(len(reference_ids) * parameters.initial_spreader_ratio))
    paired_sets = paired_seed_sets(
        reference_ids, seed_count, num_runs, int(config["paired_seed_base"])
    )
    write_json(experiment_dir / "config_snapshot.json", config)

    condition_summaries: list[Mapping[str, Any]] = []
    for strategy in seed_strategies:
        print(f"Seed strategy: {strategy}")
        for network_name, network in networks.items():
            isolates = [
                agent_id for agent_id, profile in network.profiles.items()
                if not profile.neighbors
            ]
            print(
                f"  [{network_name}] nodes={len(network.profiles)}, "
                f"edges={network.metrics.get('edges')}, isolates={isolates or 'none'}"
            )
            for model_variant in model_variants:
                runs: list[Mapping[str, Any]] = []
                for run_index in range(num_runs):
                    run_seed = int(config["run_seed_base"]) + run_index
                    explicit_seeds = (
                        paired_sets[run_index] if strategy == "paired_random" else None
                    )
                    simulation = RBEABMSimulationV2(
                        network=network,
                        model_variant=model_variant,
                        parameters=parameters,
                        run_seed=run_seed,
                        seed_strategy=strategy,
                        seed_nodes=explicit_seeds,
                        save_agent_traces=bool(config.get("save_agent_traces", False)),
                    )
                    result = dict(simulation.run())
                    result["network_name"] = network_name
                    runs.append(result)
                    write_json(
                        raw_dir / strategy / network_name / model_variant
                        / f"run_seed_{run_seed}.json",
                        result,
                    )
                condition = dict(summarize_condition(runs))
                condition["network_name"] = network_name
                condition_summaries.append(condition)
                peak = condition["run_metric_statistics"]["peak_penetration_rate"]
                print(
                    f"    {model_variant}: peak={peak['mean']:.3f} "
                    f"(95% CI {peak['ci_lower']:.3f}, {peak['ci_upper']:.3f})"
                )

    summary = {
        "schema_version": "2.0",
        "experiment_id": experiment_id,
        "created_at": datetime.now().isoformat(),
        "model_family": config["model_family"],
        "experiment_scope": (
            "Corrected rumor-propagation baselines on fixed agents and fixed "
            "pairwise networks, with structural and paired-random seeding."
        ),
        "confidence_interval": (
            "Two-sided 95% Student t interval over Monte Carlo run-level values."
        ),
        "parameter_status": config.get("parameter_status", "unspecified"),
        "config": config,
        "conditions": condition_summaries,
    }
    summary_path = experiment_dir / "summaries" / "abm_v2_monte_carlo_summary.json"
    latest_path = output_root / "abm_v2_latest_summary.json"
    write_json(summary_path, summary)
    write_json(latest_path, summary)
    print(f"Summary written to: {summary_path}")
    print(f"Latest-summary copy: {latest_path}")
    return summary_path


if __name__ == "__main__":
    main()
