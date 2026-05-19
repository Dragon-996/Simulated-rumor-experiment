import numpy as np
import scipy.stats as st
import json
import os
from datetime import datetime

# Import the core class from your main program
# Note: Make sure your main_polar.py filename has not been changed
from main_polar import SocialSimulation


def calculate_statistics(data_list):
    """
    Calculate mean, standard deviation, and 95% confidence interval
    """
    n = len(data_list)
    mean = np.mean(data_list)
    std = np.std(data_list, ddof=1)  # Sample standard deviation

    # Calculate the 95% confidence interval margin of error
    if std == 0:
        ci_error = 0.0
    else:
        # Use t-distribution to calculate confidence interval for small samples (N<30)
        ci_error = st.t.ppf(0.975, n - 1) * (std / np.sqrt(n))

    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - ci_error,
        "ci_upper": mean + ci_error,
        "ci_error": ci_error  # Often used as error bar length when plotting
    }


def run_experiments():
    # ==========================================
    # 1. Basic Experiment Configuration
    # ==========================================
    config = {
        "agent_file": "net_ba_100.json",
        "model_provider": "qwen",
        "api_key": "sk-ea17a3710a9f4ebaa5bfb4d1ac4e87fd",
        "model_name": "qwen-turbo",
        "temperature": 0.7,
        "rounds": 15,
        "seed_strategy": "structural",
        "initial_spreader_ratio": 0.10, # 10% of people act as influencers to set the pace (5% each for support and oppose)
        "initial_support": "[Opinion Discussion] Strongly support AI replacing tedious work, humans should be completely liberated to create greater value!",
        "initial_oppose": "[Opinion Discussion] Resolutely oppose the abuse of AI! Large-scale replacement of jobs by AI will lead to severe unemployment and social crisis, must be boycotted!",
    }

    num_runs = 10  # Number of independent Monte Carlo repeat experiments
    rounds = config["rounds"]

    # ==========================================
    # 2. Prepare Data Containers
    # Structure: { round_1: { metric_1: [], metric_2: [] }, round_2: {...} }
    # ==========================================
    target_metrics = [
        "support_count",
        "oppose_count",
        "neutral_count",
        "support_ratio",  # <-- New
        "oppose_ratio",  # <-- New
        "neutral_ratio",  # <-- New
        "system_entropy",
        "bipolarization_index",
        "penetration_rate"
    ]

    raw_data = {r: {m: [] for m in target_metrics} for r in range(1, rounds + 1)}

    # ==========================================
    # 3. Start Monte Carlo Loop
    # ==========================================
    print(f"========== Starting Monte Carlo Experiments (Total runs: {num_runs}) ==========")
    start_time = datetime.now()

    for run_idx in range(num_runs):
        print(f"\n>>> Executing independent simulation {run_idx + 1}/{num_runs}")

        # [Core]: Dynamically allocate global random seed to ensure different topology and sampling each time, but overall reproducibility
        config["seed"] = 1000 + run_idx

        # Instantiate and run
        sim = SocialSimulation(config)
        results = sim.run_simulation(rounds=rounds)

        # Extract data for each round and put it into the large container
        for round_result in results["rounds"]:
            r_num = round_result["round"]
            summary = round_result["summary"]
            for m in target_metrics:
                raw_data[r_num][m].append(summary[m])

    # ==========================================
    # 4. Data Summary and Statistical Calculation
    # ==========================================
    print("\n========== Experiments finished, calculating statistical metrics ==========")
    final_statistics = {}

    for r in range(1, rounds + 1):
        final_statistics[f"Round_{r}"] = {}
        print(f"\n[Round {r} Statistical Results]")
        for m in target_metrics:
            stats = calculate_statistics(raw_data[r][m])
            final_statistics[f"Round_{r}"][m] = stats

            # Console print for review
            print(f"  - {m}: Mean {stats['mean']:.3f} ± {stats['ci_error']:.3f} (95% CI)")

    # ==========================================
    # 5. Save the final statistical results as JSON for plotting
    # ==========================================
    timestamp = start_time.strftime("%Y%m%d-%H%M%S")
    # Tag the generated data files differently based on whether debunking is enabled
    if config.get("enable_debunking"):
        output_filename = f"monte_carlo_DEBUNKING_stats_{timestamp}.json"
    else:
        output_filename = f"monte_carlo_BASELINE_stats_{timestamp}.json"

    output_data = {
        "experiment_config": config,
        "total_runs": num_runs,
        "statistics": final_statistics,
        "raw_data_for_plot": raw_data  # Keep raw data for future scatter plots or box plots
    }

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Monte Carlo statistics complete! Results saved to: {output_filename}")


if __name__ == "__main__":
    run_experiments()