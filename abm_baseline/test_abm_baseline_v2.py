"""Regression tests for the corrected, separately saved V2 baseline."""

from __future__ import annotations

import unittest
from pathlib import Path

from abm_baseline import load_network
from abm_baseline_v2 import (
    POST_RUMOR,
    POST_SKEPTICAL,
    RBEABMSimulationV2,
    V2Exposure,
    V2NodeState,
    V2SimulationParameters,
)
from run_abm_experiments_v2 import paired_seed_sets


ROOT = Path(__file__).resolve().parent


def parameters(**overrides: float | int) -> V2SimulationParameters:
    values: dict[str, float | int] = {
        "rounds": 3,
        "initial_spreader_ratio": 0.05,
        "max_neighbor_posts": 8,
        "rumor_transmission_probability": 0.4,
        "belief_retention_probability": 1.0,
        "skeptical_correction_probability": 0.1,
        "believer_sharing_probability": 0.45,
        "participation_persistence_probability": 0.35,
        "skeptic_expression_probability": 0.2,
    }
    values.update(overrides)
    return V2SimulationParameters.from_mapping(values)


class V2MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = load_network(ROOT / "net_ba_100_seed46.json")

    def simulation(self, variant: str = "ABM-0", **overrides: float | int) -> RBEABMSimulationV2:
        return RBEABMSimulationV2(
            network=self.network,
            model_variant=variant,
            parameters=parameters(**overrides),
            run_seed=123,
        )

    def test_silent_agent_has_no_post(self) -> None:
        state = V2NodeState(believed=False, participated=False, post_type=None)
        self.assertIsNone(state.post_type)
        with self.assertRaises(ValueError):
            V2NodeState(believed=False, participated=False, post_type=POST_SKEPTICAL)

    def test_skeptical_only_exposure_cannot_create_belief(self) -> None:
        simulation = self.simulation()
        profile = self.network.profiles[self.network.agent_ids[0]]
        exposure = V2Exposure(("1",), ("1",), rumor_posts=0, skeptical_posts=1)
        _, probabilities, _, _ = simulation._transition(
            V2NodeState(False, False, None), exposure, profile
        )
        self.assertEqual(probabilities["belief_probability"], 0.0)

    def test_belief_does_not_decay_without_counter_information(self) -> None:
        simulation = self.simulation(belief_retention_probability=1.0)
        profile = self.network.profiles[self.network.agent_ids[0]]
        exposure = V2Exposure((), (), rumor_posts=0, skeptical_posts=0)
        state, probabilities, _, _ = simulation._transition(
            V2NodeState(True, False, None), exposure, profile
        )
        self.assertEqual(probabilities["belief_probability"], 1.0)
        self.assertTrue(state.believed)

    def test_nonbeliever_without_rumor_exposure_stays_silent(self) -> None:
        simulation = self.simulation()
        profile = self.network.profiles[self.network.agent_ids[0]]
        exposure = V2Exposure((), (), rumor_posts=0, skeptical_posts=0)
        state, probabilities, _, _ = simulation._transition(
            V2NodeState(False, False, None), exposure, profile
        )
        self.assertEqual(probabilities["participation_probability"], 0.0)
        self.assertFalse(state.participated)
        self.assertIsNone(state.post_type)

    def test_single_rumor_contact_can_create_active_spreader(self) -> None:
        simulation = self.simulation(
            rumor_transmission_probability=1.0,
            believer_sharing_probability=1.0,
        )
        profile = self.network.profiles[self.network.agent_ids[0]]
        exposure = V2Exposure(("1",), ("1",), rumor_posts=1, skeptical_posts=0)
        state, probabilities, _, _ = simulation._transition(
            V2NodeState(False, False, None), exposure, profile
        )
        self.assertEqual(probabilities["belief_probability"], 1.0)
        self.assertEqual(probabilities["participation_probability"], 1.0)
        self.assertTrue(state.believed)
        self.assertEqual(state.post_type, POST_RUMOR)

    def test_abm0_is_homogeneous_and_abmh_varies(self) -> None:
        homogeneous = self.simulation("ABM-0")
        heterogeneous = self.simulation("ABM-H")
        profiles = list(self.network.profiles.values())
        homogeneous_traits = [homogeneous._effective_traits(p) for p in profiles]
        heterogeneous_traits = [heterogeneous._effective_traits(p) for p in profiles]
        self.assertEqual(len({tuple(v.values()) for v in homogeneous_traits}), 1)
        self.assertGreater(len({tuple(v.values()) for v in heterogeneous_traits}), 1)

    def test_run_is_deterministic_given_seed(self) -> None:
        first = self.simulation("ABM-H").run()
        second = self.simulation("ABM-H").run()
        self.assertEqual(first, second)

    def test_paired_random_seed_sets_are_reproducible(self) -> None:
        ids = tuple(str(index) for index in range(100))
        first = paired_seed_sets(ids, 5, 10, 2000)
        second = paired_seed_sets(ids, 5, 10, 2000)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertTrue(all(len(set(seed_set)) == 5 for seed_set in first))


if __name__ == "__main__":
    unittest.main()
