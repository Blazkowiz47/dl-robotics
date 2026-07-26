"""Interaction-rule registry and configuration tests."""

from __future__ import annotations

import numpy as np
import pytest
from dl_core.environments import make_environment

from dl_robotics import (
    ExclusiveCellRule,
    GridScenario,
    InteractionRule,
    make_interaction_rule,
    register_interaction_rule,
)


@register_interaction_rule("test_permissive")
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


def test_make_interaction_rule_accepts_supported_configurations() -> None:
    """Names, mappings, and existing objects should share one public factory."""
    existing_rule = _PermissiveRule()

    assert isinstance(make_interaction_rule(None), ExclusiveCellRule)
    assert isinstance(
        make_interaction_rule("exclusive_cell"),
        ExclusiveCellRule,
    )
    assert isinstance(
        make_interaction_rule({"name": "exclusive"}),
        ExclusiveCellRule,
    )
    assert make_interaction_rule(existing_rule) is existing_rule


def test_make_interaction_rule_validates_configuration() -> None:
    """Malformed or unsupported rule configuration should fail clearly."""
    with pytest.raises(ValueError, match="non-empty string"):
        make_interaction_rule({})
    with pytest.raises(ValueError, match="does not accept"):
        make_interaction_rule({"name": "exclusive_cell", "priority": 2})
    with pytest.raises(TypeError, match="registered name"):
        make_interaction_rule(3)
    with pytest.raises(NotImplementedError, match="exclusive_cell_typo"):
        make_interaction_rule("exclusive_cell_typo")


def test_register_interaction_rule_rejects_unrelated_classes() -> None:
    """Invalid extensions should fail while their module is imported."""
    with pytest.raises(TypeError, match="must inherit InteractionRule"):

        @register_interaction_rule("invalid_test_rule")
        class InvalidRule:
            pass


def test_environment_constructs_registered_rule_from_yaml_shape() -> None:
    """Environment mappings should resolve locally registered rules."""
    environment = make_environment(
        {
            "name": "robotics_mapf",
            "interaction_rule": {"name": "test_permissive"},
            "scenario": {
                "width": 2,
                "height": 1,
                "starts": [[0, 0], [0, 1]],
                "goals": [[0, 1], [0, 0]],
            },
        }
    )
    environment.reset()

    _, _, terminated, _, info = environment.step(2 + (4 * 5))

    assert terminated
    assert info["actor_collisions"] == 0
    assert environment.world.positions.tolist() == [[[0, 1], [0, 0]]]
    environment.close()
