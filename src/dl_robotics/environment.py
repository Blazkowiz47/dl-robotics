"""Gymnasium-compatible centralized MAPF environments."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from dl_core.core.registry import register_environment
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space

from .observations import make_observation_builder
from .rendering import make_grid_renderer
from .rules import make_interaction_rule
from .scenario import GridScenario
from .world import GridWorldBatch, StepEvents


class _GridMAPFMixin:
    """Shared public observation and episode-information implementation."""

    def build_observations(self) -> Any:
        """Build the model and replay observation for every world."""
        return self._build_observations()

    def _build_observations(self) -> Any:
        return self.observation_builder.build(self.world)

    def build_observation(self, world_index: int = 0) -> Any:
        """Build the model and replay observation for one world."""
        return self._build_observation(world_index)

    def _build_observation(self, world_index: int = 0) -> Any:
        if isinstance(world_index, bool) or not isinstance(world_index, int):
            raise TypeError("world_index must be an integer")
        if not 0 <= world_index < self.world.num_worlds:
            raise IndexError("world_index is out of range")
        return self.build_observations()[world_index]

    def _info(
        self,
        world_index: int,
        events: StepEvents | None,
        success: bool,
    ) -> dict[str, Any]:
        return {
            "scenario": self.scenario.name,
            "scenario_fingerprint": self.scenario_fingerprint,
            "is_success": success,
            "collisions": (
                0
                if events is None
                else int(
                    events.boundary_collisions[world_index]
                    + events.wall_collisions[world_index]
                    + events.actor_collisions[world_index]
                )
            ),
            "boundary_collisions": (
                0
                if events is None
                else int(events.boundary_collisions[world_index])
            ),
            "wall_collisions": (
                0 if events is None else int(events.wall_collisions[world_index])
            ),
            "actor_collisions": (
                0
                if events is None
                else int(events.actor_collisions[world_index])
            ),
            "episode_collisions": int(
                self.world.boundary_collision_counts[world_index]
                + self.world.wall_collision_counts[world_index]
                + self.world.actor_collision_counts[world_index]
            ),
            "episode_boundary_collisions": int(
                self.world.boundary_collision_counts[world_index]
            ),
            "episode_wall_collisions": int(
                self.world.wall_collision_counts[world_index]
            ),
            "episode_actor_collisions": int(
                self.world.actor_collision_counts[world_index]
            ),
            "reached_agents": int(self.world.reached[world_index].sum()),
            "total_agents": self.scenario.num_agents,
            "makespan": int(self.world.steps[world_index]),
            "sum_of_costs": int(self.world.sum_of_costs[world_index]),
            "path_length": int(self.world.path_lengths[world_index].sum()),
        }


@register_environment("robotics_mapf")
class GridMAPFEnvironment(_GridMAPFMixin, gym.Env[np.ndarray, int]):
    """Scalar centralized MAPF environment."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"]}

    def __init__(self, config: dict[str, Any]):
        scenario_config = config.get("scenario")
        if not isinstance(scenario_config, dict):
            raise TypeError("environment.scenario must be a mapping")
        self.scenario = GridScenario.from_config(scenario_config)
        self.scenario_fingerprint = self.scenario.fingerprint

        reward_config = config.get("rewards", {})
        if not isinstance(reward_config, dict):
            raise TypeError("environment.rewards must be a mapping")
        self.step_reward = float(reward_config.get("step", -0.01))
        self.progress_reward = float(reward_config.get("progress", 0.1))
        self.collision_reward = float(reward_config.get("collision", -0.25))
        self.goal_reward = float(reward_config.get("goal", 1.0))
        self.success_reward = float(reward_config.get("success", 5.0))
        reward_values = np.asarray(
            [
                self.step_reward,
                self.progress_reward,
                self.collision_reward,
                self.goal_reward,
                self.success_reward,
            ]
        )
        if not np.isfinite(reward_values).all():
            raise ValueError("Environment rewards must be finite")

        self.world = GridWorldBatch(
            self.scenario,
            num_worlds=1,
            interaction_rule=make_interaction_rule(
                config.get("interaction_rule")
            ),
        )
        self.observation_builder = make_observation_builder(
            config.get("observation_builder")
        )

        render_config = config.get("render", {})
        if not isinstance(render_config, dict):
            raise TypeError("environment.render must be a mapping")
        self.renderer = make_grid_renderer(render_config)
        self.single_action_space = gym.spaces.Discrete(
            5**self.scenario.num_agents
        )
        self.single_observation_space = self.observation_builder.observation_space(
            self.scenario
        )
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the MAPF world."""
        return self._reset(seed=seed, options=options)

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        del options
        self.world.reset()
        success = bool(self.world.reached[0].all())
        return self.build_observation(), self._info(0, None, success)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance all actors simultaneously."""
        return self._step(action)

    def _step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        joint_actions = np.asarray([action])
        if joint_actions.shape != (1,):
            raise ValueError("Joint actions must have shape (1,)")
        if not np.issubdtype(joint_actions.dtype, np.integer):
            raise TypeError("Joint actions must use an integer dtype")
        if np.any(joint_actions < 0) or np.any(
            joint_actions >= self.single_action_space.n
        ):
            raise ValueError("Joint action is outside the configured space")
        actor_actions = np.empty(
            (1, self.scenario.num_agents),
            dtype=np.int32,
        )
        remaining = joint_actions.astype(np.int64, copy=True)
        for actor_index in range(self.scenario.num_agents):
            actor_actions[:, actor_index] = remaining % 5
            remaining //= 5

        before_distance = np.abs(
            self.world.positions - self.world.goal_positions
        ).sum(axis=(1, 2))
        events = self.world.step(actor_actions)
        after_distance = np.abs(
            self.world.positions - self.world.goal_positions
        ).sum(axis=(1, 2))
        terminated = self.world.reached.all(axis=1)
        rewards = (
            self.step_reward
            + self.progress_reward * (before_distance - after_distance)
            + self.collision_reward * events.collisions
            + self.goal_reward * events.newly_reached
            + self.success_reward * terminated
        ).astype(np.float32)
        truncated = np.logical_and(
            self.world.steps >= self.scenario.max_steps,
            ~terminated,
        )
        observations = self.build_observations()
        info = self._info(0, events, bool(terminated[0]))
        return (
            observations[0],
            float(rewards[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            info,
        )

    def render(self) -> np.ndarray:
        """Return the current world as an RGB frame."""
        return self._render()

    def _render(self) -> np.ndarray:
        return self.renderer.render_world(self.world)

    def close(self) -> None:
        """Release environment resources."""
        self._close()

    def _close(self) -> None:
        return None


@register_environment("robotics_mapf_vector")
class GridMAPFVectorEnvironment(_GridMAPFMixin, gym.vector.VectorEnv):
    """Native same-step vector MAPF environment."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array"],
        "autoreset_mode": AutoresetMode.SAME_STEP,
    }

    def __init__(self, config: dict[str, Any]):
        num_envs = config.get("num_envs", 1)
        if isinstance(num_envs, bool) or not isinstance(num_envs, int):
            raise TypeError("environment.num_envs must be an integer")
        if num_envs <= 0:
            raise ValueError("environment.num_envs must be positive")
        self.num_envs = num_envs

        scenario_config = config.get("scenario")
        if not isinstance(scenario_config, dict):
            raise TypeError("environment.scenario must be a mapping")
        self.scenario = GridScenario.from_config(scenario_config)
        self.scenario_fingerprint = self.scenario.fingerprint

        reward_config = config.get("rewards", {})
        if not isinstance(reward_config, dict):
            raise TypeError("environment.rewards must be a mapping")
        self.step_reward = float(reward_config.get("step", -0.01))
        self.progress_reward = float(reward_config.get("progress", 0.1))
        self.collision_reward = float(reward_config.get("collision", -0.25))
        self.goal_reward = float(reward_config.get("goal", 1.0))
        self.success_reward = float(reward_config.get("success", 5.0))
        reward_values = np.asarray(
            [
                self.step_reward,
                self.progress_reward,
                self.collision_reward,
                self.goal_reward,
                self.success_reward,
            ]
        )
        if not np.isfinite(reward_values).all():
            raise ValueError("Environment rewards must be finite")

        self.world = GridWorldBatch(
            self.scenario,
            num_worlds=num_envs,
            interaction_rule=make_interaction_rule(
                config.get("interaction_rule")
            ),
        )
        self.observation_builder = make_observation_builder(
            config.get("observation_builder")
        )

        render_config = config.get("render", {})
        if not isinstance(render_config, dict):
            raise TypeError("environment.render must be a mapping")
        self.renderer = make_grid_renderer(render_config)
        self.single_action_space = gym.spaces.Discrete(
            5**self.scenario.num_agents
        )
        self.single_observation_space = self.observation_builder.observation_space(
            self.scenario
        )
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(
            self.single_observation_space,
            num_envs,
        )

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset all or selected vector lanes."""
        return self._reset(seed=seed, options=options)

    def _reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if isinstance(seed, list):
            if len(seed) != self.num_envs:
                raise ValueError("seed list must have one value per environment")
            if any(
                value is not None
                and (isinstance(value, bool) or not isinstance(value, int))
                for value in seed
            ):
                raise TypeError("seed values must be integers or None")
        elif seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise TypeError("seed must be an integer, a list, or None")
        reset_mask = np.ones(self.num_envs, dtype=np.bool_)
        if options is not None and "reset_mask" in options:
            reset_values = np.asarray(options["reset_mask"])
            if not np.issubdtype(reset_values.dtype, np.bool_):
                raise TypeError("reset_mask must use a boolean dtype")
            reset_mask = reset_values
            if reset_mask.shape != (self.num_envs,):
                raise ValueError("reset_mask must have shape [num_envs]")
        self.world.reset(reset_mask)
        infos = [
            self._info(
                index,
                None,
                bool(self.world.reached[index].all()),
            )
            for index in range(self.num_envs)
        ]
        return self.build_observations(), self._batch_infos(infos)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        """Advance every MAPF environment and same-step autoreset completed lanes."""
        return self._step(actions)

    def _step(
        self,
        actions: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        joint_actions = np.asarray(actions)
        if joint_actions.shape != (self.num_envs,):
            raise ValueError(
                f"Joint actions must have shape ({self.num_envs},)"
            )
        if not np.issubdtype(joint_actions.dtype, np.integer):
            raise TypeError("Joint actions must use an integer dtype")
        if np.any(joint_actions < 0) or np.any(
            joint_actions >= self.single_action_space.n
        ):
            raise ValueError("Joint action is outside the configured space")
        actor_actions = np.empty(
            (self.num_envs, self.scenario.num_agents),
            dtype=np.int32,
        )
        remaining = joint_actions.astype(np.int64, copy=True)
        for actor_index in range(self.scenario.num_agents):
            actor_actions[:, actor_index] = remaining % 5
            remaining //= 5

        before_distance = np.abs(
            self.world.positions - self.world.goal_positions
        ).sum(axis=(1, 2))
        events = self.world.step(actor_actions)
        after_distance = np.abs(
            self.world.positions - self.world.goal_positions
        ).sum(axis=(1, 2))
        terminated = self.world.reached.all(axis=1)
        rewards = (
            self.step_reward
            + self.progress_reward * (before_distance - after_distance)
            + self.collision_reward * events.collisions
            + self.goal_reward * events.newly_reached
            + self.success_reward * terminated
        ).astype(np.float32)
        truncated = np.logical_and(
            self.world.steps >= self.scenario.max_steps,
            ~terminated,
        )
        observations = self.build_observations()
        infos = [
            self._info(index, events, bool(terminated[index]))
            for index in range(self.num_envs)
        ]
        done = np.logical_or(terminated, truncated)
        if done.any():
            final_observations = observations.copy()
            final_infos = [dict(info) for info in infos]
            self.world.reset(done)
            observations = self.build_observations()
            for index in np.flatnonzero(done):
                infos[int(index)] = self._info(
                    int(index),
                    None,
                    bool(self.world.reached[int(index)].all()),
                )
            batched_infos = self._batch_infos(infos)
            batched_infos["final_obs"] = final_observations
            batched_infos["_final_obs"] = done
            batched_infos["final_info"] = np.asarray(
                final_infos,
                dtype=object,
            )
            batched_infos["_final_info"] = done
        else:
            batched_infos = self._batch_infos(infos)
        return observations, rewards, terminated, truncated, batched_infos

    def _batch_infos(self, infos: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            key: np.asarray([info[key] for info in infos])
            for key in infos[0]
        }

    def render(self) -> np.ndarray:
        """Return every vector lane as an RGB frame batch."""
        return self._render()

    def _render(self) -> np.ndarray:
        return np.stack(
            [
                self.renderer.render_world(self.world, world_index)
                for world_index in range(self.num_envs)
            ]
        )

    def close(self) -> None:
        """Release environment resources."""
        self._close()

    def _close(self) -> None:
        return None
