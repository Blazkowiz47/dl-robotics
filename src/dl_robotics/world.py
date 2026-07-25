"""Batched discrete-time physics for 2D grid actors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .scenario import GridScenario

IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True)
class StepEvents:
    """Per-world events emitted by one simultaneous physics step."""

    boundary_collisions: IntArray
    wall_collisions: IntArray
    actor_collisions: IntArray
    newly_reached: IntArray

    @property
    def collisions(self) -> IntArray:
        """Return total rejected moves per world."""
        return (
            self.boundary_collisions
            + self.wall_collisions
            + self.actor_collisions
        )


class InteractionRule(ABC):
    """Extensible resolver for simultaneous actor/object interactions."""

    @abstractmethod
    def resolve(
        self,
        scenario: GridScenario,
        positions: IntArray,
        desired_positions: IntArray,
        blocked: BoolArray,
    ) -> tuple[IntArray, IntArray]:
        """Resolve actor conflicts and return positions and collision counts."""


class ExclusiveCellRule(InteractionRule):
    """Reject vertex conflicts, edge swaps, and moves into stationary actors."""

    def resolve(
        self,
        scenario: GridScenario,
        positions: IntArray,
        desired_positions: IntArray,
        blocked: BoolArray,
    ) -> tuple[IntArray, IntArray]:
        """Apply exclusive-cell collision rules."""
        return self._resolve(
            scenario,
            positions,
            desired_positions,
            blocked,
        )

    def _resolve(
        self,
        scenario: GridScenario,
        positions: IntArray,
        desired_positions: IntArray,
        blocked: BoolArray,
    ) -> tuple[IntArray, IntArray]:
        del scenario
        resolved = desired_positions.copy()
        actor_collisions = np.zeros(positions.shape[0], dtype=np.int32)
        for world_index in range(positions.shape[0]):
            current = positions[world_index]
            desired = resolved[world_index]
            rejected = blocked[world_index].copy()
            changed = True
            while changed:
                changed = False
                candidate = desired.copy()
                candidate[rejected] = current[rejected]
                for actor_index in range(current.shape[0]):
                    for other_index in range(actor_index + 1, current.shape[0]):
                        same_target = np.array_equal(
                            candidate[actor_index],
                            candidate[other_index],
                        )
                        edge_swap = np.array_equal(
                            candidate[actor_index],
                            current[other_index],
                        ) and np.array_equal(
                            candidate[other_index],
                            current[actor_index],
                        )
                        if same_target or edge_swap:
                            for index in (actor_index, other_index):
                                if not rejected[index] and not np.array_equal(
                                    desired[index],
                                    current[index],
                                ):
                                    rejected[index] = True
                                    changed = True
            resolved[world_index, rejected] = current[rejected]
            actor_collisions[world_index] = int(
                np.logical_and(rejected, ~blocked[world_index]).sum()
            )
        return resolved, actor_collisions


class GridWorldBatch:
    """Preallocated state for one or more identical MAPF scenarios."""

    ACTION_DELTAS = np.asarray(
        [
            [0, 0],
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
        ],
        dtype=np.int32,
    )

    def __init__(
        self,
        scenario: GridScenario,
        *,
        num_worlds: int = 1,
        interaction_rule: InteractionRule | None = None,
    ):
        if isinstance(num_worlds, bool) or not isinstance(num_worlds, int):
            raise TypeError("num_worlds must be an integer")
        if num_worlds <= 0:
            raise ValueError("num_worlds must be positive")
        if interaction_rule is not None and not isinstance(
            interaction_rule,
            InteractionRule,
        ):
            raise TypeError("interaction_rule must be an InteractionRule")
        self.scenario = scenario
        self.num_worlds = num_worlds
        self.interaction_rule = (
            ExclusiveCellRule()
            if interaction_rule is None
            else interaction_rule
        )
        self._start_positions = np.asarray(scenario.starts, dtype=np.int32)
        self._goal_positions = np.asarray(scenario.goals, dtype=np.int32)
        self.wall_mask = np.zeros(
            (scenario.height, scenario.width),
            dtype=np.bool_,
        )
        if scenario.walls:
            wall_coordinates = np.asarray(scenario.walls, dtype=np.int32)
            self.wall_mask[
                wall_coordinates[:, 0],
                wall_coordinates[:, 1],
            ] = True
        self.positions = np.empty(
            (self.num_worlds, scenario.num_agents, 2),
            dtype=np.int32,
        )
        self.velocities = np.zeros_like(self.positions)
        self.accelerations = np.zeros_like(self.positions)
        self.reached = np.zeros(
            (self.num_worlds, scenario.num_agents),
            dtype=np.bool_,
        )
        self.steps = np.zeros(self.num_worlds, dtype=np.int32)
        self.path_lengths = np.zeros(
            (self.num_worlds, scenario.num_agents),
            dtype=np.int32,
        )
        self.sum_of_costs = np.zeros(self.num_worlds, dtype=np.int32)
        self.boundary_collision_counts = np.zeros(
            self.num_worlds,
            dtype=np.int32,
        )
        self.wall_collision_counts = np.zeros(
            self.num_worlds,
            dtype=np.int32,
        )
        self.actor_collision_counts = np.zeros(
            self.num_worlds,
            dtype=np.int32,
        )
        self.reset()

    def reset(self, mask: BoolArray | None = None) -> None:
        """Reset selected worlds to their scenario starts."""
        self._reset(mask)

    def _reset(self, mask: BoolArray | None = None) -> None:
        if mask is None:
            reset_mask = np.ones(self.num_worlds, dtype=np.bool_)
        else:
            mask_values = np.asarray(mask)
            if not np.issubdtype(mask_values.dtype, np.bool_):
                raise TypeError("World reset mask must use a boolean dtype")
            reset_mask = mask_values
            if reset_mask.shape != (self.num_worlds,):
                raise ValueError("World reset mask has an invalid shape")
        self.positions[reset_mask] = self._start_positions
        self.velocities[reset_mask] = 0
        self.accelerations[reset_mask] = 0
        self.steps[reset_mask] = 0
        self.path_lengths[reset_mask] = 0
        self.sum_of_costs[reset_mask] = 0
        self.boundary_collision_counts[reset_mask] = 0
        self.wall_collision_counts[reset_mask] = 0
        self.actor_collision_counts[reset_mask] = 0
        self.reached[reset_mask] = np.all(
            self.positions[reset_mask] == self._goal_positions,
            axis=2,
        )

    def step(self, actions: IntArray) -> StepEvents:
        """Advance every world using simultaneous per-actor actions."""
        return self._step(actions)

    def _step(self, actions: IntArray) -> StepEvents:
        action_array = np.asarray(actions)
        expected_shape = (self.num_worlds, self.scenario.num_agents)
        if action_array.shape != expected_shape:
            raise ValueError(f"Actor actions must have shape {expected_shape}")
        if not np.issubdtype(action_array.dtype, np.integer):
            raise TypeError("Actor actions must use an integer dtype")
        if np.any(action_array < 0) or np.any(action_array >= len(self.ACTION_DELTAS)):
            raise ValueError("Actor actions must be in [0, 4]")

        desired_velocities = self.ACTION_DELTAS[action_array]
        if self.scenario.lock_agents_at_goal:
            desired_velocities = desired_velocities.copy()
            desired_velocities[self.reached] = 0
        desired_positions = self.positions + desired_velocities
        outside = np.logical_or(
            np.logical_or(
                desired_positions[..., 0] < 0,
                desired_positions[..., 0] >= self.scenario.height,
            ),
            np.logical_or(
                desired_positions[..., 1] < 0,
                desired_positions[..., 1] >= self.scenario.width,
            ),
        )
        safe_positions = desired_positions.copy()
        safe_positions[..., 0] = np.clip(
            safe_positions[..., 0],
            0,
            self.scenario.height - 1,
        )
        safe_positions[..., 1] = np.clip(
            safe_positions[..., 1],
            0,
            self.scenario.width - 1,
        )
        hits_wall = self.wall_mask[
            safe_positions[..., 0],
            safe_positions[..., 1],
        ]
        blocked = np.logical_or(outside, hits_wall)
        safe_positions[blocked] = self.positions[blocked]
        resolved_positions, actor_collisions = self.interaction_rule.resolve(
            self.scenario,
            self.positions,
            safe_positions,
            blocked,
        )
        resolved_positions = np.asarray(resolved_positions)
        actor_collisions = np.asarray(actor_collisions)
        if resolved_positions.shape != self.positions.shape:
            raise ValueError("Interaction rule returned an invalid position shape")
        if not np.issubdtype(resolved_positions.dtype, np.integer):
            raise TypeError("Interaction rule positions must use an integer dtype")
        if (
            np.any(resolved_positions[..., 0] < 0)
            or np.any(resolved_positions[..., 0] >= self.scenario.height)
            or np.any(resolved_positions[..., 1] < 0)
            or np.any(resolved_positions[..., 1] >= self.scenario.width)
        ):
            raise ValueError("Interaction rule positions must remain in bounds")
        if actor_collisions.shape != (self.num_worlds,):
            raise ValueError("Interaction rule returned an invalid collision shape")
        if not np.issubdtype(actor_collisions.dtype, np.integer):
            raise TypeError("Interaction rule collisions must use an integer dtype")
        if np.any(actor_collisions < 0):
            raise ValueError("Interaction rule collisions cannot be negative")
        resolved_positions = resolved_positions.astype(np.int32, copy=False)
        actor_collisions = actor_collisions.astype(np.int32, copy=False)
        old_velocities = self.velocities.copy()
        self.velocities = resolved_positions - self.positions
        self.accelerations = self.velocities - old_velocities
        self.positions = resolved_positions
        moved = np.any(self.velocities != 0, axis=2)
        self.path_lengths += moved.astype(np.int32)
        self.sum_of_costs += (~self.reached).sum(axis=1).astype(np.int32)
        previously_reached = self.reached.copy()
        self.reached = np.all(
            self.positions == self._goal_positions,
            axis=2,
        )
        newly_reached = np.logical_and(
            self.reached,
            ~previously_reached,
        ).sum(axis=1).astype(np.int32)
        self.steps += 1
        events = StepEvents(
            boundary_collisions=outside.sum(axis=1).astype(np.int32),
            wall_collisions=np.logical_and(hits_wall, ~outside)
            .sum(axis=1)
            .astype(np.int32),
            actor_collisions=actor_collisions,
            newly_reached=newly_reached,
        )
        self.boundary_collision_counts += events.boundary_collisions
        self.wall_collision_counts += events.wall_collisions
        self.actor_collision_counts += events.actor_collisions
        return events
