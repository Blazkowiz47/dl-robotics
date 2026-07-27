"""Configurable model-observation construction for grid worlds."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from dl_core.core import ComponentRegistry

from .rendering import GridRenderer
from .scenario import GridScenario
from .world import GridWorldBatch


class GridObservationBuilder:
    """Build model observations from batched grid-world state."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GridObservationBuilder:
        """Create a builder from its YAML-compatible configuration."""
        return cls(**dict(config))

    def observation_space(self, scenario: GridScenario) -> gym.Space[Any]:
        """Return the Gymnasium space produced for one world."""
        return self._observation_space(scenario)

    def _observation_space(self, scenario: GridScenario) -> gym.Space[Any]:
        raise NotImplementedError

    def build(self, world: GridWorldBatch) -> Any:
        """Build one model observation per world."""
        return self._build(world)

    def _build(self, world: GridWorldBatch) -> Any:
        raise NotImplementedError


class SemanticGridObservationBuilder(GridObservationBuilder):
    """Build the default seven-channel MAPF state tensor."""

    def _observation_space(self, scenario: GridScenario) -> gym.Space[np.ndarray]:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7, scenario.height, scenario.width),
            dtype=np.float32,
        )

    def _build(self, world: GridWorldBatch) -> np.ndarray:
        observations = np.zeros(
            (
                world.num_worlds,
                7,
                world.scenario.height,
                world.scenario.width,
            ),
            dtype=np.float32,
        )
        observations[:, 0] = world.wall_mask
        actor_scale = float(world.scenario.num_agents)
        world_indices = np.arange(world.num_worlds)
        for actor_index in range(world.scenario.num_agents):
            rows = world.positions[:, actor_index, 0]
            columns = world.positions[:, actor_index, 1]
            identity = (actor_index + 1) / actor_scale
            observations[world_indices, 1, rows, columns] = identity
            goal_row, goal_column = world.goal_positions[actor_index]
            observations[:, 2, goal_row, goal_column] = identity
            observations[
                world_indices, 3, rows, columns
            ] = world.velocities[:, actor_index, 0]
            observations[
                world_indices, 4, rows, columns
            ] = world.velocities[:, actor_index, 1]
            observations[
                world_indices, 5, rows, columns
            ] = world.accelerations[:, actor_index, 0] / 2.0
            observations[
                world_indices, 6, rows, columns
            ] = world.accelerations[:, actor_index, 1] / 2.0
        return observations


class RenderedGridObservationBuilder(GridObservationBuilder):
    """Build HWC uint8 model observations with configurable actor/goal shapes."""

    def __init__(
        self,
        *,
        output_size: int | None = None,
        cell_size: int = 48,
        show_grid: bool = True,
        actor_shape: str = "circle",
        goal_shape: str = "square",
        show_actor_ids: bool = True,
    ) -> None:
        if output_size is not None and (
            isinstance(output_size, bool)
            or not isinstance(output_size, int)
        ):
            raise TypeError("output_size must be an integer or None")
        if output_size is not None and output_size <= 0:
            raise ValueError("output_size must be positive")
        self.output_size = output_size
        self.renderer = GridRenderer(
            cell_size=cell_size,
            show_grid=show_grid,
            actor_shape=actor_shape,
            goal_shape=goal_shape,
            show_actor_ids=show_actor_ids,
        )

    def _observation_space(self, scenario: GridScenario) -> gym.Space[np.ndarray]:
        height = (
            self.output_size
            if self.output_size is not None
            else scenario.height * self.renderer.cell_size
        )
        width = (
            self.output_size
            if self.output_size is not None
            else scenario.width * self.renderer.cell_size
        )
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )

    def _build(self, world: GridWorldBatch) -> np.ndarray:
        frames = []
        for world_index in range(world.num_worlds):
            if self.output_size is None:
                frames.append(
                    self.renderer.render_world(world, world_index)
                )
            else:
                frames.append(
                    self.renderer.render_world_at_size(
                        world,
                        self.output_size,
                        world_index,
                    )
                )
        return np.stack(frames)


OBSERVATION_BUILDER_REGISTRY = ComponentRegistry("Observation builder")
OBSERVATION_BUILDER_REGISTRY.register_class(
    "semantic_grid",
    SemanticGridObservationBuilder,
)
OBSERVATION_BUILDER_REGISTRY.register_class(
    "semantic",
    SemanticGridObservationBuilder,
)
OBSERVATION_BUILDER_REGISTRY.register_class(
    "rendered_grid",
    RenderedGridObservationBuilder,
)
OBSERVATION_BUILDER_REGISTRY.register_class(
    "rgb_grid",
    RenderedGridObservationBuilder,
)


def register_observation_builder(names: str | list[str]):
    """Register an observation-builder class under one or more names."""

    def decorator(
        builder_class: type[GridObservationBuilder],
    ) -> type[GridObservationBuilder]:
        if not isinstance(builder_class, type) or not issubclass(
            builder_class,
            GridObservationBuilder,
        ):
            raise TypeError(
                "Registered observation builders must inherit "
                "GridObservationBuilder"
            )
        OBSERVATION_BUILDER_REGISTRY.register(names)(builder_class)
        return builder_class

    return decorator


def make_observation_builder(
    config: str | Mapping[str, Any] | GridObservationBuilder | None,
) -> GridObservationBuilder:
    """Create an observation builder from config or an existing instance."""
    if config is None:
        return SemanticGridObservationBuilder()
    if isinstance(config, GridObservationBuilder):
        return config
    if isinstance(config, str):
        builder_name = config
        builder_config: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        builder_name = config.get("name")
        if not isinstance(builder_name, str) or not builder_name:
            raise ValueError(
                "observation_builder.name must be a non-empty string"
            )
        builder_config = {
            key: value for key, value in config.items() if key != "name"
        }
    else:
        raise TypeError(
            "observation_builder must be a registered name, mapping, or "
            "GridObservationBuilder instance"
        )

    registered_builders = OBSERVATION_BUILDER_REGISTRY.registered_items()
    if builder_name not in registered_builders:
        raise NotImplementedError(
            f"Observation builder '{builder_name}' not found. Available "
            f"observation builders: {list(registered_builders)}"
        )
    builder_class = registered_builders[builder_name]
    if not issubclass(builder_class, GridObservationBuilder):
        raise TypeError(
            f"Registered observation builder '{builder_name}' must inherit "
            "GridObservationBuilder"
        )
    return builder_class.from_config(builder_config)


__all__ = [
    "OBSERVATION_BUILDER_REGISTRY",
    "GridObservationBuilder",
    "RenderedGridObservationBuilder",
    "SemanticGridObservationBuilder",
    "make_observation_builder",
    "register_observation_builder",
]
