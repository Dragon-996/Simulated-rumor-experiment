"""Plot V2 penetration/entropy trajectories and peak comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rbe-abm-v2-mpl-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {"ABM-0": "#145DA0", "ABM-H": "#D1495B"}
LINESTYLES = {"ABM-0": "-", "ABM-H": "--"}
MARKERS = {"ABM-0": "o", "ABM-H": "s"}
NETWORK_ORDER = ("BA", "WS", "ER", "Community")
MODEL_ORDER = ("ABM-0", "ABM-H")
STRATEGY_ORDER = ("structural", "paired_random")
STRATEGY_LABELS = {
    "structural": "Structural seeds",
    "paired_random": "Paired random seeds",
}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot corrected V2 ABM results.")
    parser.add_argument("--summary", default="abm_results_v2/abm_v2_latest_summary.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--strategy", default="structural", choices=STRATEGY_ORDER)
    return parser.parse_args()


def ordered(present: set[str], preferred: tuple[str, ...]) -> list[str]:
    return [x for x in preferred if x in present] + sorted(present - set(preferred))


def load_lookup(path: Path) -> tuple[Mapping[str, Any], dict[tuple[str, str, str], Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for condition in summary.get("conditions", []):
        key = (
            condition["seed_strategy"],
            condition["network_name"],
            condition["model_variant"],
        )
        if key in lookup:
            raise ValueError(f"Duplicate condition: {key}")
        lookup[key] = condition
    if not lookup:
        raise ValueError("Summary contains no conditions")
    return summary, lookup


def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    saved = []
    for suffix in ("svg", "pdf", "tiff", "png"):
        path = base.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix in {"tiff", "png"}:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def round_items(condition: Mapping[str, Any]) -> list[tuple[int, Mapping[str, Any]]]:
    return sorted(
        ((int(key.split("_")[-1]), value) for key, value in condition["round_statistics"].items()),
        key=lambda item: item[0],
    )


def write_source_data(
    lookup: Mapping[tuple[str, str, str], Mapping[str, Any]], output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "abm_v2_trajectory_source_data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "seed_strategy", "network", "model", "round", "metric", "n",
            "mean", "std", "ci_lower", "ci_upper",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (strategy, network, model), condition in sorted(lookup.items()):
            for round_number, metrics in round_items(condition):
                for metric in ("penetration_rate", "system_entropy"):
                    stats = metrics[metric]
                    writer.writerow({
                        "seed_strategy": strategy, "network": network,
                        "model": model, "round": round_number, "metric": metric,
                        **{key: stats[key] for key in ("n", "mean", "std", "ci_lower", "ci_upper")},
                    })
    with (output / "abm_v2_peak_source_data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "seed_strategy", "network", "model", "metric", "n", "mean",
            "std", "ci_lower", "ci_upper",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (strategy, network, model), condition in sorted(lookup.items()):
            for metric in ("peak_penetration_rate", "peak_system_entropy"):
                stats = condition["run_metric_statistics"][metric]
                writer.writerow({
                    "seed_strategy": strategy, "network": network, "model": model,
                    "metric": metric,
                    **{key: stats[key] for key in ("n", "mean", "std", "ci_lower", "ci_upper")},
                })


def plot_trajectories(
    lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
    networks: list[str], models: list[str], strategy: str, output: Path,
) -> list[Path]:
    fig, axes = plt.subplots(2, len(networks), figsize=(11.2, 5.2), sharex=True)
    for column, network in enumerate(networks):
        for row, metric in enumerate(("penetration_rate", "system_entropy")):
            ax = axes[row, column]
            for model in models:
                items = round_items(lookup[(strategy, network, model)])
                x = [number for number, _ in items]
                stats = [values[metric] for _, values in items]
                y = [value["mean"] for value in stats]
                lower = [max(0.0, value["ci_lower"]) for value in stats]
                upper_limit = 1.0 if metric == "penetration_rate" else 2.0
                upper = [min(upper_limit, value["ci_upper"]) for value in stats]
                ax.plot(
                    x, y, color=PALETTE[model], linestyle=LINESTYLES[model],
                    marker=MARKERS[model], markevery=3, markersize=3,
                    linewidth=1.5, label=f"{model} (n={stats[0]['n']})",
                )
                ax.fill_between(x, lower, upper, color=PALETTE[model], alpha=0.13, linewidth=0)
            ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
            ax.set_ylim(0, 1 if metric == "penetration_rate" else 2)
            if row == 0:
                ax.set_title(network)
            if column == 0:
                ax.set_ylabel("Rumor penetration" if row == 0 else "State entropy (bits)")
            if row == 1:
                ax.set_xlabel("Simulation round")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(STRATEGY_LABELS[strategy], y=1.05, fontsize=9)
    return save_figure(fig, output / f"abm_v2_penetration_entropy_{strategy}")


def plot_peaks(
    lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
    networks: list[str], models: list[str], strategies: list[str], output: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, len(strategies), figsize=(7.4, 3.2), sharey=True, squeeze=False)
    for column, strategy in enumerate(strategies):
        ax = axes[0, column]
        x = list(range(len(networks)))
        for model_index, model in enumerate(models):
            offset = -0.09 if model_index == 0 else 0.09
            stats = [lookup[(strategy, network, model)]["run_metric_statistics"]["peak_penetration_rate"] for network in networks]
            means = [item["mean"] for item in stats]
            errors = [
                [mean - item["ci_lower"] for mean, item in zip(means, stats)],
                [item["ci_upper"] - mean for mean, item in zip(means, stats)],
            ]
            ax.errorbar(
                [value + offset for value in x], means, yerr=errors,
                fmt=MARKERS[model], color=PALETTE[model], capsize=3,
                markersize=5, linewidth=0, elinewidth=1.1, label=model,
            )
        ax.set_title(STRATEGY_LABELS[strategy])
        ax.set_xticks(x, networks)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
        if column == 0:
            ax.set_ylabel("Peak rumor penetration")
        ax.legend(loc="best")
    return save_figure(fig, output / "abm_v2_peak_penetration_by_seed_strategy")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    _, lookup = load_lookup(summary_path)
    networks = ordered({key[1] for key in lookup}, NETWORK_ORDER)
    models = ordered({key[2] for key in lookup}, MODEL_ORDER)
    strategies = ordered({key[0] for key in lookup}, STRATEGY_ORDER)
    required = [(s, n, m) for s in strategies for n in networks for m in models]
    missing = [key for key in required if key not in lookup]
    if missing:
        raise ValueError(f"Missing conditions: {missing}")
    output = Path(args.output_dir).resolve() if args.output_dir else summary_path.parent / "figures"
    apply_style()
    write_source_data(lookup, output)
    files = plot_trajectories(lookup, networks, models, args.strategy, output)
    files += plot_peaks(lookup, networks, models, strategies, output)
    print(f"Source data and figures written to: {output}")
    for path in files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
