"""Rendering, media, and robotics episode-manager tests."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from dl_core.core import EpisodeContext, EpisodeResult, Transition
from dl_core.utils import ArtifactManager

from dl_robotics import (
    GridMAPFEnvironment,
    GridRenderer,
    RoboticsEpisodeManager,
    write_animation,
)


def _environment() -> GridMAPFEnvironment:
    return GridMAPFEnvironment(
        {
            "scenario": {
                "name": "render",
                "width": 3,
                "height": 2,
                "max_steps": 2,
                "starts": [[0, 0], [1, 2]],
                "goals": [[0, 2], [1, 0]],
                "walls": [[0, 1]],
            },
            "render": {"cell_size": 16},
        }
    )


def test_renderer_and_environment_return_rgb_uint8_frames() -> None:
    environment = _environment()
    observation, _ = environment.reset()
    frame = environment.render()
    rendered_observation = GridRenderer(
        cell_size=16
    ).render_observation(observation)

    assert frame.shape == (32, 48, 3)
    assert frame.dtype == np.uint8
    assert np.array_equal(frame, rendered_observation)
    environment.close()


def test_renderer_keeps_actor_and_goal_identity_colors_aligned() -> None:
    observation = np.zeros((3, 1, 3), dtype=np.float32)
    observation[2, 0, 0] = 0.5
    observation[2, 0, 2] = 1.0
    observation[1, 0, 1] = 1.0

    frame = GridRenderer(cell_size=16, show_grid=False).render_observation(
        observation
    )

    assert frame[8, 20].tolist() == [220, 38, 38]


def test_renderer_supports_solid_triangles_and_hollow_circles() -> None:
    observation = np.zeros((3, 1, 2), dtype=np.float32)
    observation[1, 0, 0] = 1.0
    observation[2, 0, 1] = 1.0
    renderer = GridRenderer(
        cell_size=24,
        show_grid=False,
        actor_shape="triangle",
        goal_shape="circle",
        show_actor_ids=False,
    )

    frame = renderer.render_observation(observation)

    assert frame[12, 12].tolist() == [37, 99, 235]
    assert frame[12, 36].tolist() == [248, 248, 248]
    assert frame[12, 43].tolist() == [37, 99, 235]


def test_renderer_passes_rgb_model_observations_through() -> None:
    observation = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    frame = GridRenderer().render_observation(observation)

    assert np.array_equal(frame, observation)
    assert frame is not observation


def test_write_animation_creates_gif_and_mp4(tmp_path: Path) -> None:
    frames = np.stack(
        [
            np.full((16, 16, 3), value, dtype=np.uint8)
            for value in (0, 127, 255)
        ]
    )

    gif_path = write_animation(tmp_path / "episode.gif", frames, fps=4)
    mp4_path = write_animation(tmp_path / "episode.mp4", frames, fps=4)
    gif_reader = imageio.get_reader(gif_path)
    frame_metadata = gif_reader.get_meta_data(index=1)
    gif_reader.close()

    assert gif_path.stat().st_size > 0
    assert mp4_path.stat().st_size > 0
    assert frame_metadata["duration"] == 250


def test_write_animation_pads_odd_mp4_dimensions(tmp_path: Path) -> None:
    frames = np.zeros((2, 9, 11, 3), dtype=np.uint8)

    output_path = write_animation(tmp_path / "odd.mp4", frames)
    reader = imageio.get_reader(output_path)
    encoded_frame = reader.get_data(0)
    reader.close()

    assert encoded_frame.shape == (10, 12, 3)


def test_rendering_configuration_is_validated(tmp_path: Path) -> None:
    renderer = GridRenderer(cell_size=8)

    with pytest.raises(ValueError, match="height and width"):
        renderer.render_observation(np.zeros((3, 0, 2)))
    with pytest.raises(TypeError, match="fps must be an integer"):
        write_animation(
            tmp_path / "episode.gif",
            np.zeros((1, 8, 8, 3), dtype=np.uint8),
            fps=True,
        )
    with pytest.raises(TypeError, match="media_format"):
        RoboticsEpisodeManager({"media_format": 1})
    with pytest.raises(ValueError, match="fps must be positive"):
        RoboticsEpisodeManager({"fps": 0})
    with pytest.raises(ValueError, match="cannot exceed 100"):
        RoboticsEpisodeManager({"media_format": "gif", "fps": 101})


def test_robotics_episode_manager_emits_metrics_and_media(tmp_path: Path) -> None:
    environment = _environment()
    initial_observation, reset_info = environment.reset()
    next_observation, reward, _, _, step_info = environment.step(0)
    artifact_manager = ArtifactManager(
        run_name="robotics-manager",
        output_dir=str(tmp_path),
    )
    manager = RoboticsEpisodeManager(
        {
            "capture_phases": ["evaluation"],
            "info_keys": ["episode_collisions"],
            "media_format": "gif",
            "fps": 4,
            "cell_size": 16,
        },
        artifact_manager=artifact_manager,
    )
    manager.set_name("robotics")
    manager.begin_episode(
        EpisodeContext(
            episode_id="evaluation-00000000",
            episode=0,
            environment_index=0,
            phase="evaluation",
            seed=3,
            initial_observation=initial_observation,
            reset_info=reset_info,
        )
    )
    manager.record_transition(
        0,
        Transition(
            observation=initial_observation,
            action=0,
            reward=reward,
            next_observation=next_observation,
            terminated=False,
            truncated=True,
            info=step_info,
        ),
        phase="evaluation",
    )
    result = EpisodeResult(
        episode=0,
        episode_return=reward,
        length=1,
        terminated=False,
        truncated=True,
        final_info=step_info,
    )

    record = manager.end_episode(0, result, phase="evaluation")

    assert record.metrics["robotics/reached_fraction"] == 0.0
    assert record.metrics["robotics/makespan"] == 1.0
    assert Path(record.artifact_paths["gif"]).stat().st_size > 0
    assert Path(record.artifact_paths["trajectory"]).exists()
    environment.close()
