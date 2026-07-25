"""Network loading utilities shared by the RBE-ABM V2 baseline.

The original V2 scripts imported these definitions from ``abm_baseline.py``,
but that source file was not present in the public repository. This module
reconstructs the deterministic data-loading layer required by V2. It does not
add a behavioral mechanism.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping


def clip_probability(value: float) -> float:
    """Clip a numeric value to the closed probability interval [0, 1]."""

    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class AgentProfile:
    """Static agent information used by the heterogeneous ABM variant."""

    agent_id: str
    neighbors: tuple[str, ...]
    belief_susceptibility: float
    conformity: float
    expression: float
    big_five: Mapping[str, float]


@dataclass(frozen=True)
class NetworkData:
    """One fixed network plus the unchanged agent attributes attached to it."""

    source_file: str
    profiles: Mapping[str, AgentProfile]
    metrics: Mapping[str, Any]
    agent_ids: tuple[str, ...]
    population_trait_means: Mapping[str, float]
    attribute_signature: str


def _sort_key(agent_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(agent_id))
    except ValueError:
        return (1, agent_id)


def _neighbor_ids(raw_neighbors: Any) -> tuple[str, ...]:
    result: list[str] = []
    for item in raw_neighbors or []:
        if isinstance(item, Mapping):
            if "id" not in item:
                raise ValueError(f"Neighbor object has no 'id': {item!r}")
            result.append(str(item["id"]))
        else:
            result.append(str(item))
    return tuple(sorted(dict.fromkeys(result), key=_sort_key))


def _trait_values(
    agent: Mapping[str, Any],
) -> tuple[dict[str, float], float, float, float]:
    raw_big_five = agent["psychology"]["big_five"]
    big_five = {
        name: float(raw_big_five[name])
        for name in (
            "Openness",
            "Conscientiousness",
            "Extraversion",
            "Agreeableness",
            "Neuroticism",
        )
    }
    belief_susceptibility = (
        big_five["Openness"]
        + (1.0 - big_five["Conscientiousness"])
        + big_five["Neuroticism"]
    ) / 3.0
    conformity = big_five["Agreeableness"]
    expression = (
        big_five["Extraversion"] + big_five["Neuroticism"]
    ) / 2.0
    return (
        big_five,
        clip_probability(belief_susceptibility),
        clip_probability(conformity),
        clip_probability(expression),
    )


def _attribute_signature(agents: Mapping[str, Mapping[str, Any]]) -> str:
    """Hash agent attributes while deliberately excluding network neighbors."""

    payload: dict[str, Any] = {}
    for agent_id in sorted((str(value) for value in agents), key=_sort_key):
        agent = agents[agent_id]
        payload[agent_id] = {
            key: value for key, value in agent.items() if key != "neighbors"
        }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_network(file_path: str | Path) -> NetworkData:
    """Load and validate a fixed network JSON file."""

    path = Path(file_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw_agents = data.get("agents")
    if not isinstance(raw_agents, Mapping) or not raw_agents:
        raise ValueError(f"{path.name} has no non-empty 'agents' mapping")

    normalized_agents = {str(key): value for key, value in raw_agents.items()}
    agent_ids = tuple(sorted(normalized_agents, key=_sort_key))
    known_ids = set(agent_ids)
    profiles: dict[str, AgentProfile] = {}

    for agent_id in agent_ids:
        agent = normalized_agents[agent_id]
        neighbors = _neighbor_ids(agent.get("neighbors", []))
        unknown = set(neighbors) - known_ids
        if unknown:
            raise ValueError(
                f"{path.name}: agent {agent_id} has unknown neighbors "
                f"{sorted(unknown, key=_sort_key)}"
            )
        big_five, susceptibility, conformity, expression = _trait_values(agent)
        profiles[agent_id] = AgentProfile(
            agent_id=agent_id,
            neighbors=neighbors,
            belief_susceptibility=susceptibility,
            conformity=conformity,
            expression=expression,
            big_five=big_five,
        )

    means = {
        "belief_susceptibility": fmean(
            profile.belief_susceptibility for profile in profiles.values()
        ),
        "conformity": fmean(
            profile.conformity for profile in profiles.values()
        ),
        "expression": fmean(
            profile.expression for profile in profiles.values()
        ),
    }
    return NetworkData(
        source_file=path.name,
        profiles=profiles,
        metrics=dict(data.get("metrics", {})),
        agent_ids=agent_ids,
        population_trait_means=means,
        attribute_signature=_attribute_signature(normalized_agents),
    )
