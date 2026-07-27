# Simulated Rumor Experiment

An LLM-driven multi-agent simulation framework for studying rumor propagation, debunking interventions, opinion dynamics, and polarization on complex social networks.

## LLM-Driven Agent-Based Social Simulation

This repository provides a lightweight implementation of an LLM-driven agent-based simulation framework. It supports experiments on:

* rumor diffusion;
* official debunking interventions;
* opinion activation and polarization;
* network-topology effects;
* population-size robustness;
* initial seed-selection sensitivity;
* prompt-formulation sensitivity;
* Monte Carlo repeated simulations.

The framework is intended for controlled computational experiments. It should not be interpreted as a calibrated predictor of real-world human behavior or public-opinion distributions.

The complete workflow contains five main steps:

1. Generate heterogeneous agents.
2. Build a social network.
3. Select or modify the prompt formulation.
4. Run a single LLM-driven simulation.
5. Run repeated Monte Carlo experiments.

---

## File Overview

| File                   | Description                                                                                                                                                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `agent_generator_x.py` | Generates agents with demographic attributes, Big Five personality traits, and interests.                                                                                                                                                     |
| `agent_network.py`     | Builds BA, ER, WS, and attribute-based community networks. For compatibility with the existing code, the attribute-based community option is still identified by `"hyper"`.                                                                   |
| `prompt_universal.py`  | Constructs prompts using agent attributes, recent memory, visible neighboring posts, task instructions, and structured-output constraints. It also contains the P0, P1, and P2 prompt formulations used in the prompt-sensitivity experiment. |
| `LLM_call.py`          | Provides a unified interface for calling different LLM APIs.                                                                                                                                                                                  |
| `main_run.py`          | Runs one complete simulation, including rumor initialization and optional debunking interventions.                                                                                                                                            |
| `run_monte_carlo.py`   | Runs repeated simulations and calculates mean values, standard deviations, and 95% confidence intervals.                                                                                                                                      |

---

## Requirements

Python 3.9 or later is recommended.

Install the required packages:

```bash
pip install networkx numpy scipy requests openai zhipuai
```

Depending on the selected model provider, additional SDK packages may be required.

---

## API Key Setup

Before running the simulation, configure the required API key.

For example, in Linux or macOS:

```bash
export QWEN_API_KEY="your_api_key_here"
```

In Windows PowerShell:

```powershell
$env:QWEN_API_KEY="your_api_key_here"
```

Environment variables are recommended instead of writing real API keys directly into the source code.

Example:

```python
import os

config = {
    "api_key": os.getenv("QWEN_API_KEY")
}
```

The environment-variable name should be changed when another provider is used.

---

## Step 1: Generate Agents

Run:

```bash
python agent_generator_x.py --seed 46 --out .
```

This generates agent files such as:

```text
agents_N50_seed46.json
agents_N100_seed46.json
agents_N500_seed46.json
```

The default population sizes are defined in `agent_generator_x.py`:

```python
network_scales = [50, 100, 500]
```

Modify this list to generate other population sizes.

The random seed should be recorded to make agent attributes reproducible across experiments.

---

## Step 2: Generate a Network

Open `agent_network.py` and modify the execution settings:

```python
INPUT_FILE = "agents_N100_seed46.json"
OUTPUT_FILE = "net_ba_100_seed46.json"

NETWORK_TYPE = "ba"   # Options: "ba", "er", "ws", "hyper"
AVG_DEGREE = 4.0
SEED = 46
```

Then run:

```bash
python agent_network.py
```

Available network options are:

| Option    | Network Type                                                                      |
| --------- | --------------------------------------------------------------------------------- |
| `"ba"`    | Barabási–Albert scale-free network                                                |
| `"er"`    | Erdős–Rényi random network                                                        |
| `"ws"`    | Watts–Strogatz small-world network                                                |
| `"hyper"` | Attribute-based community network retained under the legacy option name `"hyper"` |

### Important Note About the `"hyper"` Option

Although the option is named `"hyper"` in the existing source code, it does not represent a mathematical hypergraph in the current version of the repository.

Instead, this option constructs a community-oriented network based on similarities or relationships among agent attributes. The option name is retained only to preserve compatibility with the existing scripts and configuration files.

Therefore, results generated with:

```python
NETWORK_TYPE = "hyper"
```

should be described as results from an **attribute-based community network**, not as results from a hypergraph.

The generated network file, such as:

```text
net_ba_100_seed46.json
```

is subsequently used as the input for the simulation.

---

## Step 3: Select and Modify the Prompt

The prompt templates are defined in:

```text
prompt_universal.py
```

A complete prompt may include:

* the agent's demographic profile;
* Big Five personality descriptions;
* recent self-generated posts;
* visible posts from neighboring agents;
* the current simulation topic;
* the behavioral decision task;
* structured-output instructions;
* JSON-format constraints.

No separate system prompt is used. The role instructions, agent information, social context, task description, and output constraints are all included in the user prompt.

---

## Prompt-Sensitivity Experiment

The repository contains three prompt formulations used in the prompt-sensitivity experiment:

| Prompt | Description                                                                                                                                                               |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `P0`   | Integrated baseline formulation. Belief evaluation, sharing decisions, and content generation are presented in one integrated task.                                       |
| `P1`   | Credibility-first formulation. The agent is first instructed to evaluate the credibility of the information before providing the structured belief and sharing decisions. |
| `P2`   | Decomposed belief–sharing formulation. Belief formation and public sharing are presented as two more explicitly separated decisions.                                      |

The three formulations are intended to test whether the simulation trajectory depends on how the same decision task is linguistically and structurally formulated.

Before running a prompt-sensitivity experiment, manually select the corresponding P0, P1, or P2 prompt block in `prompt_universal.py`.

Only the prompt formulation should be changed. The following settings should remain identical across P0, P1, and P2:

* agent profiles;
* network structure;
* network-generation seed;
* initial rumor;
* seed-node strategy;
* initial spreader ratio;
* number of rounds;
* number of visible neighboring posts;
* memory length;
* LLM provider and model;
* temperature and other API parameters;
* synchronous update mechanism;
* Monte Carlo repetition count.

The current rumor-related structured output is:

```json
{
  "thought_process": "...",
  "is_believed": true,
  "will_spread": true,
  "new_post": "..."
}
```

To change the agent behavior instructions, edit the relevant prompt formulation in `prompt_universal.py`.

To change the maximum number of visible neighboring posts, edit:

```python
neighbor_posts_list = neighbor_posts_list[:8]
```

To change the retained self-history length, edit:

```python
own_history_posts[-3:]
```

These values should be kept unchanged when the purpose is to compare P0, P1, and P2 fairly.

---

## Manually Changing the Opinion-Polarization Topic

For opinion-dynamics or polarization experiments, the topic-specific wording must be modified manually in `prompt_universal.py`.

For example, the current prompt may contain the structured field:

```json
{
  "stance": "support / oppose / neutral"
}
```

It may also contain a topic-specific instruction such as:

```text
What is your core stance?

Extremely important: fill in "support" if you support the topic that AI will
completely replace human jobs in the future, fill in "oppose" if you oppose
it. If you do not care about the topic, or if no one in your social circle
has mentioned it, you MUST fill in "neutral".
```

Before changing the experimental topic, replace all topic-specific wording in this instruction.

For example, when changing from AI job replacement to another topic, revise both:

1. the topic statement shown to the agent; and
2. the semantic definitions of `"support"` and `"oppose"`.

The topic description and the stance-classification rule must refer to exactly the same proposition. Otherwise, the returned stance field may become ambiguous or internally inconsistent.

A generic version is:

```text
What is your core stance?

Fill in "support" if you support the focal proposition, fill in "oppose" if
you oppose the focal proposition, and fill in "neutral" if you currently have
no clear position or have not encountered relevant discussion.
```

The generic placeholder `"focal proposition"` should be replaced by the complete experimental topic before running the program.

When comparing multiple topics, keep the remaining settings fixed wherever possible, including:

* network;
* agent profiles;
* population size;
* LLM model;
* simulation rounds;
* prompt structure;
* output schema;
* number of visible posts;
* memory length;
* update mechanism.

Only the focal topic and its corresponding support–oppose interpretation should be changed.

---

## Step 4: Run One Simulation

Open `main_run.py` and modify the configuration:

```python
config = {
    "agent_file": "net_ba_100_seed46.json",
    "model_provider": "qwen",
    "api_key": os.getenv("QWEN_API_KEY"),
    "model_name": "qwen-turbo",
    "temperature": 0.7,
    "rounds": 15,
    "seed_strategy": "structural",
    "initial_spreader_ratio": 0.05,
    "initial_rumor": (
        "Urgent Notice: The east city water source is polluted, "
        "do not drink tap water!"
    ),
    "enable_debunking": False,
}
```

Then run:

```bash
python main_run.py
```

Key parameters are:

| Parameter                | Meaning                                                                 |
| ------------------------ | ----------------------------------------------------------------------- |
| `agent_file`             | Networked agent file generated by `agent_network.py`.                   |
| `model_provider`         | LLM provider, such as `"qwen"`, `"deepseek"`, `"zhipu"`, or `"openai"`. |
| `model_name`             | API model name used in the simulation.                                  |
| `temperature`            | Sampling temperature.                                                   |
| `rounds`                 | Number of simulation rounds.                                            |
| `seed_strategy`          | Initial rumor seed-selection strategy.                                  |
| `initial_spreader_ratio` | Proportion of agents initialized as rumor spreaders.                    |
| `initial_rumor`          | Rumor message injected into the selected seed nodes.                    |
| `enable_debunking`       | Whether official correction information is introduced.                  |

Available seed strategies include:

| Option            | Description                                                                             |
| ----------------- | --------------------------------------------------------------------------------------- |
| `"structural"`    | Prioritizes structurally central or high-degree nodes.                                  |
| `"psychological"` | Selects nodes according to the attribute-based scoring rule implemented in the program. |
| `"random"`        | Randomly selects initial spreaders.                                                     |

The `"psychological"` option is an operational seed-selection rule based on agent attributes. It should not be interpreted as a validated psychological measurement or a claim that the agents reproduce human psychological processes.

---

## Synchronous Round Updating

The simulation uses round-based synchronous updating.

At the beginning of each round:

1. the currently available posts are copied into a round-level snapshot;
2. agents are processed in a randomized order;
3. every agent reads information from the same beginning-of-round snapshot;
4. newly generated posts are temporarily stored;
5. the public post collection is updated only after all agents have completed the round.

Therefore, a post generated by an earlier processed agent cannot be observed by another agent during the same round. It becomes visible in the following round.

---

## Debunking Settings

Debunking is enabled by setting:

```python
"enable_debunking": True
```

For a baseline rumor-diffusion experiment, use:

```python
"enable_debunking": False
```

The intervention timing and official correction message are controlled manually in `main_run.py`, inside the block beginning with:

```python
if self.config.get("enable_debunking", False):
```

An example official correction message is:

```python
official_message = (
    "Water Quality Monitoring Center: The east city water source is NOT "
    "polluted, all water quality tests are normal!\n"
    "[Government News] 72-hour continuous monitoring has been conducted "
    "at 15 sampling points across the city, and the data show that the "
    "water quality is safe.\n"
    "[Water Quality Monitoring Center] All 106 water quality indicators "
    "passed (Test No: 2026WS001). Do not believe rumors.\n"
)
```

### Manual Intervention-Schedule Selection

The intervention schedule is not selected automatically. Readers must manually modify the round conditions in `main_run.py` according to the experiment being reproduced.

### Finite Intervention Window

For an intervention active only from rounds 7 to 10:

```python
if 7 <= self.current_round <= 10:
    official_message = (
        "Water Quality Monitoring Center: The east city water source is NOT "
        "polluted, all water quality tests are normal!\n"
        "[Government News] 72-hour continuous monitoring has been conducted "
        "at 15 sampling points across the city, and the data show that the "
        "water quality is safe.\n"
        "[Water Quality Monitoring Center] All 106 water quality indicators "
        "passed (Test No: 2026WS001). Do not believe rumors.\n"
    )
```

Outside rounds 7–10, no new official correction should be injected.

### Continuous Intervention

For a continuous intervention from rounds 7 to 15:

```python
if 7 <= self.current_round <= 15:
    official_message = (
        "Water Quality Monitoring Center: The east city water source is NOT "
        "polluted, all water quality tests are normal!\n"
        "[Government News] 72-hour continuous monitoring has been conducted "
        "at 15 sampling points across the city, and the data show that the "
        "water quality is safe.\n"
        "[Water Quality Monitoring Center] All 106 water quality indicators "
        "passed (Test No: 2026WS001). Do not believe rumors.\n"
    )
```

The following two-branch form is equivalent when the same message is used in both branches:

```python
if 7 <= self.current_round <= 10:
    official_message = (...)
elif 11 <= self.current_round <= 15:
    official_message = (...)
```

Because the two intervals are consecutive, this produces a continuous intervention from rounds 7 through 15.

### Non-Consecutive Intervention Rounds

For an intermittent schedule active in rounds 6–7, 9–10, and 12–13:

```python
if self.current_round in {6, 7, 9, 10, 12, 13}:
    official_message = (
        "Water Quality Monitoring Center: The east city water source is NOT "
        "polluted, all water quality tests are normal!\n"
        "[Government News] 72-hour continuous monitoring has been conducted "
        "at 15 sampling points across the city, and the data show that the "
        "water quality is safe.\n"
        "[Water Quality Monitoring Center] All 106 water quality indicators "
        "passed (Test No: 2026WS001). Do not believe rumors.\n"
    )
```

The code comment describing the intervention plan must be consistent with the actual conditional statement. For example, the comment:

```python
# According to your plan, deploy in rounds 6-7, 9-10, 12-13
```

should only be used when the condition actually selects rounds 6–7, 9–10, and 12–13.

Using this comment above a condition such as:

```python
if 7 <= self.current_round <= 10:
```

would be misleading because the comment and the implemented schedule describe different interventions.

When comparing intervention strategies, keep the official correction content identical and change only the intervention window.

---

## Step 5: Run Monte Carlo Experiments

Open `run_monte_carlo.py` and modify the experimental configuration.

Set the number of repeated runs:

```python
num_runs = 10
```

Then run:

```bash
python run_monte_carlo.py
```

The script calculates:

* mean values;
* standard deviations;
* 95% confidence intervals;
* raw run-level data for subsequent plotting and analysis.

Example output filenames include:

```text
monte_carlo_BASELINE_stats_YYYYMMDD-HHMMSS.json
```

and:

```text
monte_carlo_DEBUNKING_stats_YYYYMMDD-HHMMSS.json
```

When comparing experimental conditions, use the same number of repeated runs and preserve all control settings that are not part of the intended experimental manipulation.

---

## Output Files

A single simulation creates a folder such as:

```text
simulation_results_YYYYMMDD-HHMMSS/
```

The folder may contain:

| File                        | Description                                  |
| --------------------------- | -------------------------------------------- |
| `simulation.log`            | Complete execution log.                      |
| `round_*.json`              | Round-level simulation states and decisions. |
| `simulation_summary_*.json` | Full simulation summary.                     |
| `post_history_*.json`       | Complete history of generated posts.         |

Monte Carlo experiments additionally produce aggregated statistical files containing repeated-run results.

---

## Reproducing Different Experiments

The repository does not automatically select every experiment reported in the associated study. Some conditions must be configured manually.

| Experiment                          | Required Manual Change                                                        |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| Network-topology comparison         | Change `NETWORK_TYPE` and regenerate the network.                             |
| Population-size robustness          | Generate agents and networks for the required population size.                |
| Seed-strategy sensitivity           | Change `seed_strategy` in `main_run.py`.                                      |
| Prompt-sensitivity analysis         | Select P0, P1, or P2 in `prompt_universal.py`.                                |
| Topic-polarization comparison       | Replace all topic-specific wording and stance definitions in the prompt.      |
| Baseline rumor diffusion            | Set `enable_debunking` to `False`.                                            |
| Debunking experiment                | Set `enable_debunking` to `True` and manually select the intervention rounds. |
| Finite versus continuous correction | Keep the correction text fixed and modify only the intervention window.       |
| Cross-model comparison              | Change `model_provider`, `model_name`, and the corresponding API key.         |

For every experiment, record the prompt version, model name, network file, random seeds, intervention schedule, and simulation parameters.

---

## Notes

* API model names may need to be adjusted according to the models currently provided by each API service.
* The `"hyper"` network option is a legacy code identifier for the attribute-based community structure and should not be reported as a mathematical hypergraph.
* P0, P1, and P2 must be selected manually in `prompt_universal.py`.
* Polarization topics and their support–oppose definitions must be edited manually and kept semantically consistent.
* Debunking schedules must be selected manually in `main_run.py`.
* Comments describing intervention rounds should always match the actual conditional statements.
* Real API keys should not be committed to a public repository.
* Raw outputs should be inspected for parsing failures or invalid structured responses before calculating summary statistics.
* Results should be interpreted as comparisons within the specified model–prompt–network system rather than as calibrated estimates of human response probabilities.
