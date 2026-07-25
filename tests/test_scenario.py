"""Scenario and world validation tests."""

import numpy as np
import pytest

from dl_robotics import GridScenario, GridWorldBatch, InteractionRule


class _PermissiveRule(InteractionRule):
    def resolve(
        self,
        scenario: GridScenario,
        positions: np.ndarray,
        desired_positions: np.ndarray,
        blocked: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del scenario, positions, blocked
        return (
            desired_positions.copy(),
            np.zeros(desired_positions.shape[0], dtype=np.int32),
        )


def test_scenario_validates_geometry_and_has_stable_fingerprint() -> None:
    scenario = GridScenario.from_config(
        {
            "width": 4,
            "height": 3,
            "starts": [[0, 0], [2, 3]],
            "goals": [[2, 3], [0, 0]],
            "walls": [[1, 1]],
        }
    )
    restored = GridScenario.from_config(
        {
            "width": 4,
            "height": 3,
            "starts": [[0, 0], [2, 3]],
            "goals": [[2, 3], [0, 0]],
            "walls": [[1, 1]],
        }
    )

    assert scenario.num_agents == 2
    assert scenario.fingerprint == restored.fingerprint

    with pytest.raises(ValueError, match="unique"):
        GridScenario(
            width=2,
            height=2,
            starts=((0, 0), (0, 0)),
            goals=((1, 0), (1, 1)),
        )

    with pytest.raises(ValueError, match="pairs"):
        GridScenario.from_config(
            {
                "width": 2,
                "height": 2,
                "starts": [[0, 0, 1]],
                "goals": [[1, 1]],
            }
        )

    with pytest.raises(TypeError, match="contain integers"):
        GridScenario.from_config(
            {
                "width": 2,
                "height": 2,
                "starts": [[0.5, 0]],
                "goals": [[1, 1]],
            }
        )

    with pytest.raises(TypeError, match="boolean"):
        GridScenario.from_config(
            {
                "width": 2,
                "height": 2,
                "starts": [[0, 0]],
                "goals": [[1, 1]],
                "lock_agents_at_goal": "false",
            }
        )

    with pytest.raises(TypeError, match="stored in a tuple"):
        GridScenario(
            width=2,
            height=2,
            starts=[(0, 0)],  # type: ignore[arg-type]
            goals=((1, 1),),
        )


def test_world_rejects_vertex_conflicts_and_edge_swaps() -> None:
    vertex_world = GridWorldBatch(
        GridScenario(
            width=3,
            height=1,
            starts=((0, 0), (0, 2)),
            goals=((0, 2), (0, 0)),
        )
    )
    vertex_events = vertex_world.step(np.asarray([[2, 4]], dtype=np.int32))

    assert vertex_world.positions.tolist() == [[[0, 0], [0, 2]]]
    assert vertex_events.actor_collisions.tolist() == [2]

    swap_world = GridWorldBatch(
        GridScenario(
            width=2,
            height=1,
            starts=((0, 0), (0, 1)),
            goals=((0, 1), (0, 0)),
        )
    )
    swap_events = swap_world.step(np.asarray([[2, 4]], dtype=np.int32))

    assert swap_world.positions.tolist() == [[[0, 0], [0, 1]]]
    assert swap_events.actor_collisions.tolist() == [2]


def test_world_cascades_rejections_from_stationary_actors() -> None:
    world = GridWorldBatch(
        GridScenario(
            width=3,
            height=1,
            starts=((0, 0), (0, 1), (0, 2)),
            goals=((0, 2), (0, 0), (0, 1)),
        )
    )

    events = world.step(np.asarray([[2, 2, 0]], dtype=np.int32))

    assert world.positions.tolist() == [[[0, 0], [0, 1], [0, 2]]]
    assert events.actor_collisions.tolist() == [2]


def test_world_allows_a_four_actor_rotation_without_edge_swaps() -> None:
    world = GridWorldBatch(
        GridScenario(
            width=2,
            height=2,
            starts=((0, 0), (0, 1), (1, 1), (1, 0)),
            goals=((0, 1), (1, 1), (1, 0), (0, 0)),
        )
    )

    events = world.step(np.asarray([[2, 3, 4, 1]], dtype=np.int32))

    assert world.positions.tolist() == [
        [[0, 1], [1, 1], [1, 0], [0, 0]]
    ]
    assert events.actor_collisions.tolist() == [0]


def test_world_tracks_velocity_acceleration_and_wall_collisions() -> None:
    world = GridWorldBatch(
        GridScenario(
            width=3,
            height=2,
            starts=((0, 0),),
            goals=((0, 2),),
            walls=((1, 1),),
        )
    )

    world.step(np.asarray([[2]], dtype=np.int32))
    events = world.step(np.asarray([[3]], dtype=np.int32))

    assert world.positions.tolist() == [[[0, 1]]]
    assert world.velocities.tolist() == [[[0, 0]]]
    assert world.accelerations.tolist() == [[[0, -1]]]
    assert events.wall_collisions.tolist() == [1]
    assert world.path_lengths.tolist() == [[1]]


def test_world_allows_an_explicit_interaction_rule_override() -> None:
    world = GridWorldBatch(
        GridScenario(
            width=2,
            height=1,
            starts=((0, 0), (0, 1)),
            goals=((0, 1), (0, 0)),
        ),
        interaction_rule=_PermissiveRule(),
    )

    events = world.step(np.asarray([[2, 4]], dtype=np.int32))

    assert world.positions.tolist() == [[[0, 1], [0, 0]]]
    assert events.actor_collisions.tolist() == [0]
