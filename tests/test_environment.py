"""Gymnasium and dl-core integration tests."""

import numpy as np
import pytest
from dl_core.environments import make_environment
from gymnasium.vector import AutoresetMode

import dl_robotics


def _config() -> dict:
    return {
        "scenario": {
            "name": "swap",
            "width": 3,
            "height": 2,
            "max_steps": 2,
            "starts": [[0, 0], [1, 2]],
            "goals": [[0, 2], [1, 0]],
            "walls": [],
        },
        "rewards": {
            "step": -0.01,
            "progress": 0.1,
            "collision": -0.25,
            "goal": 1.0,
            "success": 5.0,
        },
    }


def test_scalar_environment_exposes_semantic_observations_and_metrics() -> None:
    assert dl_robotics.__version__ == "0.0.3"
    environment = make_environment({"name": "robotics_mapf", **_config()})

    observation, info = environment.reset(seed=3)
    next_observation, reward, terminated, truncated, next_info = environment.step(
        2 + (4 * 5)
    )

    assert environment.observation_space.contains(observation)
    assert environment.observation_space.contains(next_observation)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["scenario"] == "swap"
    assert next_info["path_length"] == 2
    environment.close()


def test_scalar_environment_retains_episode_collision_totals() -> None:
    environment = make_environment({"name": "robotics_mapf", **_config()})
    environment.reset()

    _, _, _, _, first_info = environment.step(1 + (3 * 5))
    _, _, _, _, second_info = environment.step(1 + (3 * 5))

    assert first_info["collisions"] == 2
    assert first_info["episode_collisions"] == 2
    assert second_info["collisions"] == 2
    assert second_info["episode_collisions"] == 4
    environment.close()


def test_vector_environment_same_step_autoresets_and_preserves_final_state() -> None:
    environment = make_environment(
        {
            "name": "robotics_mapf_vector",
            "num_envs": 2,
            **_config(),
        }
    )
    observations, _ = environment.reset(seed=[1, 2])

    first_actions = np.asarray([2 + (4 * 5), 2 + (4 * 5)])
    observations, _, _, _, _ = environment.step(first_actions)
    reset_observations, _, terminated, truncated, infos = environment.step(
        first_actions
    )

    assert environment.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
    assert observations.shape == (2, 7, 2, 3)
    assert terminated.tolist() == [True, True]
    assert truncated.tolist() == [False, False]
    assert infos["_final_obs"].tolist() == [True, True]
    assert infos["final_info"][0]["is_success"] is True
    assert not np.array_equal(infos["final_obs"][0], reset_observations[0])
    environment.close()


def test_vector_environment_masks_only_the_completed_lane() -> None:
    config = _config()
    config["scenario"]["max_steps"] = 3
    environment = make_environment(
        {
            "name": "robotics_mapf_vector",
            "num_envs": 2,
            **config,
        }
    )
    environment.reset(seed=[1, 2])

    observations, _, terminated, truncated, infos = environment.step(
        np.asarray([2 + (4 * 5), 0])
    )
    observations, _, terminated, truncated, infos = environment.step(
        np.asarray([2 + (4 * 5), 0])
    )

    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, False]
    assert infos["_final_obs"].tolist() == [True, False]
    assert infos["final_info"][0]["is_success"] is True
    assert np.count_nonzero(observations[:, 1]) == 4
    environment.close()


def test_vector_environment_validates_seed_count() -> None:
    environment = make_environment(
        {
            "name": "robotics_mapf_vector",
            "num_envs": 2,
            **_config(),
        }
    )

    with pytest.raises(ValueError, match="one value per environment"):
        environment.reset(seed=[1])
    with pytest.raises(TypeError, match="boolean dtype"):
        environment.reset(options={"reset_mask": [1, 0]})

    environment.close()

    with pytest.raises(TypeError, match="must be an integer"):
        make_environment(
            {
                "name": "robotics_mapf_vector",
                "num_envs": 1.5,
                **_config(),
            }
        )


def test_vector_autoreset_reports_success_when_reset_state_is_already_goal() -> None:
    environment = make_environment(
        {
            "name": "robotics_mapf_vector",
            "num_envs": 1,
            "scenario": {
                "width": 1,
                "height": 1,
                "starts": [[0, 0]],
                "goals": [[0, 0]],
            },
        }
    )
    environment.reset()

    _, _, terminated, truncated, infos = environment.step(np.asarray([0]))

    assert terminated.tolist() == [True]
    assert truncated.tolist() == [False]
    assert infos["is_success"].tolist() == [True]
    assert infos["final_info"][0]["is_success"] is True
    environment.close()
