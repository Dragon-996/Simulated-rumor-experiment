"""Corrected reversible belief-expression ABM baselines (V2).

V2 keeps the original pre-generated agents and adjacency lists but corrects the
decision logic used in V1:

* silent agents produce no focal-event post;
* rumor and skeptical contacts have separate, opposite effects on belief;
* contact counts drive belief adoption without a second exposure multiplier;
* belief retention and public participation are distinct mechanisms;
* the former hypergraph projection is treated as a community network label by
  the experiment configuration, with no claim of higher-order interaction.

The V1 files are not modified and remain independently runnable.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from abm_baseline import AgentProfile, NetworkData, clip_probability


MODEL_HETEROGENEITY = {"ABM-0": 0.0, "ABM-H": 1.0}

POST_RUMOR = "rumor"
POST_SKEPTICAL = "skeptical"

STATE_ACTIVE_SPREADER = "active_spreader"
STATE_SILENT_BELIEVER = "silent_believer"
STATE_ACTIVE_SKEPTIC = "active_skeptic"
STATE_PASSIVE_SKEPTIC = "passive_skeptic"


@dataclass(frozen=True)
class V2SimulationParameters:
    """Shared V2 parameters; all values are probabilities in [0, 1]."""

    rounds: int
    initial_spreader_ratio: float
    max_neighbor_posts: int
    rumor_transmission_probability: float
    belief_retention_probability: float
    skeptical_correction_probability: float
    believer_sharing_probability: float
    participation_persistence_probability: float
    skeptic_expression_probability: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "V2SimulationParameters":
        parameters = cls(
            rounds=int(values["rounds"]),
            initial_spreader_ratio=float(values["initial_spreader_ratio"]),
            max_neighbor_posts=int(values["max_neighbor_posts"]),
            rumor_transmission_probability=float(
                values["rumor_transmission_probability"]
            ),
            belief_retention_probability=float(
                values["belief_retention_probability"]
            ),
            skeptical_correction_probability=float(
                values["skeptical_correction_probability"]
            ),
            believer_sharing_probability=float(
                values["believer_sharing_probability"]
            ),
            participation_persistence_probability=float(
                values["participation_persistence_probability"]
            ),
            skeptic_expression_probability=float(
                values["skeptic_expression_probability"]
            ),
        )
        parameters.validate()
        return parameters

    def validate(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")
        if not 0.0 < self.initial_spreader_ratio <= 1.0:
            raise ValueError("initial_spreader_ratio must be in (0, 1]")
        if self.max_neighbor_posts <= 0:
            raise ValueError("max_neighbor_posts must be positive")
        for name in (
            "rumor_transmission_probability",
            "belief_retention_probability",
            "skeptical_correction_probability",
            "believer_sharing_probability",
            "participation_persistence_probability",
            "skeptic_expression_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class V2NodeState:
    """Dynamic node state. None means no focal-event post was published."""

    believed: bool
    participated: bool
    post_type: str | None

    def __post_init__(self) -> None:
        if not self.participated and self.post_type is not None:
            raise ValueError("A non-participant cannot publish a focal-event post")
        if self.participated and self.post_type not in {POST_RUMOR, POST_SKEPTICAL}:
            raise ValueError("A participant must publish rumor or skeptical content")
        if self.believed and self.participated and self.post_type != POST_RUMOR:
            raise ValueError("An active believer must publish rumor content")
        if not self.believed and self.participated and self.post_type != POST_SKEPTICAL:
            raise ValueError("An active non-believer must publish skeptical content")

    @property
    def state_label(self) -> str:
        if self.believed and self.participated:
            return STATE_ACTIVE_SPREADER
        if self.believed:
            return STATE_SILENT_BELIEVER
        if self.participated:
            return STATE_ACTIVE_SKEPTIC
        return STATE_PASSIVE_SKEPTIC


@dataclass(frozen=True)
class V2Exposure:
    """Observed focal-event posts; silent neighbors do not enter this feed."""

    available_focal_neighbor_ids: tuple[str, ...]
    observed_neighbor_ids: tuple[str, ...]
    rumor_posts: int
    skeptical_posts: int

    @property
    def focal_posts(self) -> int:
        return self.rumor_posts + self.skeptical_posts

    @property
    def rumor_share(self) -> float:
        if self.focal_posts == 0:
            return 0.0
        return self.rumor_posts / self.focal_posts


class RBEABMSimulationV2:
    """One corrected V2 Monte Carlo simulation run."""

    def __init__(
        self,
        network: NetworkData,
        model_variant: str,
        parameters: V2SimulationParameters,
        run_seed: int,
        seed_strategy: str = "structural",
        seed_nodes: Sequence[str] | None = None,
        save_agent_traces: bool = False,
    ) -> None:
        if model_variant not in MODEL_HETEROGENEITY:
            raise ValueError(f"Unknown model variant: {model_variant}")
        if seed_strategy not in {"structural", "paired_random"}:
            raise ValueError(f"Unknown seed strategy: {seed_strategy}")

        self.network = network
        self.model_variant = model_variant
        self.parameters = parameters
        self.run_seed = int(run_seed)
        self.seed_strategy = seed_strategy
        self.save_agent_traces = bool(save_agent_traces)
        self.rng = random.Random(self.run_seed)
        self.heterogeneity_strength = MODEL_HETEROGENEITY[model_variant]
        self.population_trait_means = network.population_trait_means

        expected_seed_count = max(
            1, int(len(network.profiles) * parameters.initial_spreader_ratio)
        )
        if seed_nodes is None:
            if seed_strategy != "structural":
                raise ValueError("paired_random requires explicit seed_nodes")
            selected_seed_nodes = self._select_structural_seeds(expected_seed_count)
        else:
            selected_seed_nodes = tuple(str(agent_id) for agent_id in seed_nodes)
            if len(selected_seed_nodes) != expected_seed_count:
                raise ValueError(
                    f"Expected {expected_seed_count} seeds, got {len(selected_seed_nodes)}"
                )
            if len(set(selected_seed_nodes)) != len(selected_seed_nodes):
                raise ValueError("seed_nodes contains duplicates")
            missing = set(selected_seed_nodes) - set(network.profiles)
            if missing:
                raise ValueError(f"Unknown seed nodes: {sorted(missing, key=int)}")

        self.seed_nodes = tuple(selected_seed_nodes)
        self.states = self._initialize_states()
        self.ever_believed = set(self.seed_nodes)

    def _select_structural_seeds(self, count: int) -> tuple[str, ...]:
        ranked = sorted(
            self.network.profiles,
            key=lambda agent_id: (
                -len(self.network.profiles[agent_id].neighbors),
                int(agent_id),
            ),
        )
        return tuple(ranked[:count])

    def _initialize_states(self) -> dict[str, V2NodeState]:
        seed_set = set(self.seed_nodes)
        return {
            agent_id: V2NodeState(
                believed=agent_id in seed_set,
                participated=agent_id in seed_set,
                post_type=POST_RUMOR if agent_id in seed_set else None,
            )
            for agent_id in self.network.agent_ids
        }

    def _effective_traits(self, profile: AgentProfile) -> Mapping[str, float]:
        strength = self.heterogeneity_strength
        raw_values = {
            "belief_susceptibility": profile.belief_susceptibility,
            "conformity": profile.conformity,
            "expression": profile.expression,
        }
        return {
            name: clip_probability(
                self.population_trait_means[name]
                + strength * (raw_value - self.population_trait_means[name])
            )
            for name, raw_value in raw_values.items()
        }

    def _observe_neighbors(
        self,
        agent_id: str,
        post_snapshot: Mapping[str, str | None],
    ) -> V2Exposure:
        focal_neighbors = [
            neighbor_id
            for neighbor_id in self.network.profiles[agent_id].neighbors
            if post_snapshot[neighbor_id] is not None
        ]
        self.rng.shuffle(focal_neighbors)
        observed = tuple(focal_neighbors[: self.parameters.max_neighbor_posts])
        rumor_posts = sum(
            post_snapshot[neighbor_id] == POST_RUMOR for neighbor_id in observed
        )
        skeptical_posts = sum(
            post_snapshot[neighbor_id] == POST_SKEPTICAL for neighbor_id in observed
        )
        return V2Exposure(
            available_focal_neighbor_ids=tuple(focal_neighbors),
            observed_neighbor_ids=observed,
            rumor_posts=rumor_posts,
            skeptical_posts=skeptical_posts,
        )

    def _individual_contact_probabilities(
        self,
        profile: AgentProfile,
        exposure: V2Exposure,
    ) -> tuple[float, float, Mapping[str, float]]:
        traits = self._effective_traits(profile)
        social_direction = 2.0 * exposure.rumor_share - 1.0
        belief_modifier = 1.0 + self.heterogeneity_strength * (
            traits["belief_susceptibility"]
            - self.population_trait_means["belief_susceptibility"]
            + (
                traits["conformity"] - self.population_trait_means["conformity"]
            )
            * social_direction
        )
        rumor_contact_probability = clip_probability(
            self.parameters.rumor_transmission_probability * belief_modifier
        )

        expression_modifier = 1.0 + self.heterogeneity_strength * (
            traits["expression"] - self.population_trait_means["expression"]
        )
        believer_sharing_probability = clip_probability(
            self.parameters.believer_sharing_probability * expression_modifier
        )
        return rumor_contact_probability, believer_sharing_probability, traits

    def _transition(
        self,
        previous_state: V2NodeState,
        exposure: V2Exposure,
        profile: AgentProfile,
    ) -> tuple[V2NodeState, Mapping[str, float], Mapping[str, float], Mapping[str, float]]:
        rumor_contact_probability, sharing_probability, traits = (
            self._individual_contact_probabilities(profile, exposure)
        )
        rumor_pressure = 1.0 - (
            1.0 - rumor_contact_probability
        ) ** exposure.rumor_posts
        skeptical_pressure = 1.0 - (
            1.0 - self.parameters.skeptical_correction_probability
        ) ** exposure.skeptical_posts

        retained_belief = (
            self.parameters.belief_retention_probability
            if previous_state.believed
            else 0.0
        )
        belief_before_skepticism = 1.0 - (1.0 - retained_belief) * (
            1.0 - rumor_pressure
        )
        belief_probability = clip_probability(
            belief_before_skepticism * (1.0 - skeptical_pressure)
        )
        belief_draw = self.rng.random()
        believed = belief_draw < belief_probability

        if believed:
            fresh_participation_probability = sharing_probability
        else:
            fresh_participation_probability = 1.0 - (
                1.0 - self.parameters.skeptic_expression_probability
            ) ** exposure.rumor_posts

        retained_participation = (
            self.parameters.participation_persistence_probability
            if previous_state.participated
            else 0.0
        )
        participation_probability = clip_probability(
            1.0
            - (1.0 - retained_participation)
            * (1.0 - fresh_participation_probability)
        )
        participation_draw = self.rng.random()
        participated = participation_draw < participation_probability

        if not participated:
            post_type = None
        else:
            post_type = POST_RUMOR if believed else POST_SKEPTICAL
        new_state = V2NodeState(
            believed=believed,
            participated=participated,
            post_type=post_type,
        )
        probabilities = {
            "rumor_contact_probability": rumor_contact_probability,
            "rumor_pressure": rumor_pressure,
            "skeptical_pressure": skeptical_pressure,
            "belief_before_skepticism": belief_before_skepticism,
            "belief_probability": belief_probability,
            "fresh_participation_probability": fresh_participation_probability,
            "participation_probability": participation_probability,
        }
        draws = {
            "belief_draw": belief_draw,
            "participation_draw": participation_draw,
        }
        return new_state, probabilities, draws, traits

    def _summarize_states(self, round_number: int) -> Mapping[str, Any]:
        labels = [state.state_label for state in self.states.values()]
        counts = {
            STATE_ACTIVE_SPREADER: labels.count(STATE_ACTIVE_SPREADER),
            STATE_SILENT_BELIEVER: labels.count(STATE_SILENT_BELIEVER),
            STATE_ACTIVE_SKEPTIC: labels.count(STATE_ACTIVE_SKEPTIC),
            STATE_PASSIVE_SKEPTIC: labels.count(STATE_PASSIVE_SKEPTIC),
        }
        total = len(self.states)
        proportions = {name: value / total for name, value in counts.items()}
        entropy = -sum(
            proportion * math.log2(proportion)
            for proportion in proportions.values()
            if proportion > 0.0
        )
        believers = (
            counts[STATE_ACTIVE_SPREADER] + counts[STATE_SILENT_BELIEVER]
        ) / total
        skeptics = 1.0 - believers
        focal_post_ratio = (
            counts[STATE_ACTIVE_SPREADER] + counts[STATE_ACTIVE_SKEPTIC]
        ) / total
        return {
            "round": round_number,
            "active_spreaders": counts[STATE_ACTIVE_SPREADER],
            "silent_believers": counts[STATE_SILENT_BELIEVER],
            "active_skeptics": counts[STATE_ACTIVE_SKEPTIC],
            "passive_skeptics": counts[STATE_PASSIVE_SKEPTIC],
            "active_spreaders_ratio": proportions[STATE_ACTIVE_SPREADER],
            "silent_believers_ratio": proportions[STATE_SILENT_BELIEVER],
            "active_skeptics_ratio": proportions[STATE_ACTIVE_SKEPTIC],
            "passive_skeptics_ratio": proportions[STATE_PASSIVE_SKEPTIC],
            "believers_ratio": believers,
            "skeptics_ratio": skeptics,
            "penetration_rate": believers,
            "system_entropy": entropy,
            "bipolarization_index": 1.0 - abs(believers - skeptics),
            "focal_post_ratio": focal_post_ratio,
            "silent_ratio": 1.0 - focal_post_ratio,
            "ever_believed_ratio": len(self.ever_believed) / total,
        }

    def _run_round(self, round_number: int) -> Mapping[str, Any]:
        post_snapshot = {
            agent_id: state.post_type for agent_id, state in self.states.items()
        }
        agent_order = list(self.network.agent_ids)
        self.rng.shuffle(agent_order)
        next_states: dict[str, V2NodeState] = {}
        traces: dict[str, Any] = {}

        for agent_id in agent_order:
            profile = self.network.profiles[agent_id]
            previous_state = self.states[agent_id]
            exposure = self._observe_neighbors(agent_id, post_snapshot)
            new_state, probabilities, draws, traits = self._transition(
                previous_state, exposure, profile
            )
            next_states[agent_id] = new_state
            if new_state.believed:
                self.ever_believed.add(agent_id)
            if self.save_agent_traces:
                traces[agent_id] = {
                    "previous_state": asdict(previous_state),
                    "exposure": asdict(exposure),
                    "effective_traits": dict(traits),
                    "probabilities": dict(probabilities),
                    "random_draws": dict(draws),
                    "new_state": asdict(new_state),
                    "state_label": new_state.state_label,
                }

        self.states = next_states
        result: dict[str, Any] = {
            "round": round_number,
            "summary": self._summarize_states(round_number),
        }
        if self.save_agent_traces:
            result["agent_decisions"] = traces
        return result

    def _seed_diagnostics(self) -> Mapping[str, Any]:
        seed_set = set(self.seed_nodes)
        unique_neighbors = {
            neighbor_id
            for seed_id in self.seed_nodes
            for neighbor_id in self.network.profiles[seed_id].neighbors
            if neighbor_id not in seed_set
        }
        degrees = [
            len(self.network.profiles[seed_id].neighbors) for seed_id in self.seed_nodes
        ]
        return {
            "seed_count": len(self.seed_nodes),
            "seed_degrees": degrees,
            "seed_degree_sum": sum(degrees),
            "unique_nonseed_neighbors": len(unique_neighbors),
            "initial_neighbor_coverage_ratio": len(unique_neighbors)
            / len(self.network.profiles),
        }

    def run(self) -> Mapping[str, Any]:
        initial_summary = self._summarize_states(0)
        round_results = [
            self._run_round(round_number)
            for round_number in range(1, self.parameters.rounds + 1)
        ]
        trajectory = [initial_summary] + [
            result["summary"] for result in round_results
        ]
        penetration = [point["penetration_rate"] for point in trajectory]
        entropy = [point["system_entropy"] for point in trajectory]
        peak_penetration = max(penetration)
        return {
            "schema_version": "2.0",
            "model_family": "Rumor Belief-Expression ABM (RBE-ABM)",
            "model_variant": self.model_variant,
            "network_file": self.network.source_file,
            "network_metrics": dict(self.network.metrics),
            "run_seed": self.run_seed,
            "seed_strategy": self.seed_strategy,
            "seed_nodes": list(self.seed_nodes),
            "seed_diagnostics": dict(self._seed_diagnostics()),
            "heterogeneity_strength": self.heterogeneity_strength,
            "population_trait_means": dict(self.population_trait_means),
            "parameters": asdict(self.parameters),
            "initial_summary": initial_summary,
            "rounds": round_results,
            "run_metrics": {
                "peak_penetration_rate": peak_penetration,
                "final_penetration_rate": penetration[-1],
                "time_to_peak": penetration.index(peak_penetration),
                "penetration_auc": sum(
                    (penetration[index] + penetration[index + 1]) / 2.0
                    for index in range(len(penetration) - 1)
                ),
                "peak_system_entropy": max(entropy),
                "final_system_entropy": entropy[-1],
                "final_ever_believed_ratio": trajectory[-1][
                    "ever_believed_ratio"
                ],
            },
        }
