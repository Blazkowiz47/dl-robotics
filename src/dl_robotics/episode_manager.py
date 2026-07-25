"""MAPF episode metrics and visual artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
from dl_core.core import (
    EpisodeRecord,
    EpisodeResult,
    config_field,
    register_episode_manager,
)
from dl_core.episode_managers import StandardEpisodeManager
from dl_core.utils import ArtifactManager

from .rendering import GridRenderer, write_animation


@register_episode_manager("robotics")
class RoboticsEpisodeManager(StandardEpisodeManager):
    """Track MAPF metrics and optionally render captured trajectories."""

    CONFIG_FIELDS = StandardEpisodeManager.CONFIG_FIELDS + [
        config_field(
            "media_format",
            "str",
            "Captured trajectory media: none, gif, mp4, or both.",
            default="gif",
        ),
        config_field(
            "fps",
            "int",
            "Playback frames per second for GIF and MP4 artifacts.",
            default=8,
        ),
        config_field(
            "cell_size",
            "int",
            "Rendered pixels per grid cell.",
            default=48,
        ),
    ]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        artifact_manager: ArtifactManager | None = None,
        trainer: Any = None,
    ):
        super().__init__(config, artifact_manager, trainer)
        self.media_format = self.config.get("media_format", "gif")
        if not isinstance(self.media_format, str):
            raise TypeError("media_format must be a string")
        if self.media_format not in {"none", "gif", "mp4", "both"}:
            raise ValueError("media_format must be none, gif, mp4, or both")
        self.fps = self.config.get("fps", 8)
        if isinstance(self.fps, bool) or not isinstance(self.fps, int):
            raise TypeError("fps must be an integer")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.media_format in {"gif", "both"} and self.fps > 100:
            raise ValueError("GIF fps cannot exceed 100")
        self.renderer = GridRenderer(
            cell_size=self.config.get("cell_size", 48)
        )

    def summarize_episode(
        self,
        record: EpisodeRecord,
        result: EpisodeResult,
        **statistics: float,
    ) -> dict[str, float]:
        """Return generic and MAPF-specific episode metrics."""
        return self._summarize_episode(record, result, **statistics)

    def _summarize_episode(
        self,
        record: EpisodeRecord,
        result: EpisodeResult,
        **statistics: float,
    ) -> dict[str, float]:
        metrics = super().summarize_episode(record, result, **statistics)
        final_info = result.final_info
        for info_key, metric_key in (
            ("episode_collisions", "robotics/collisions"),
            ("episode_boundary_collisions", "robotics/boundary_collisions"),
            ("episode_wall_collisions", "robotics/wall_collisions"),
            ("episode_actor_collisions", "robotics/actor_collisions"),
            ("makespan", "robotics/makespan"),
            ("sum_of_costs", "robotics/sum_of_costs"),
            ("path_length", "robotics/path_length"),
        ):
            value = final_info.get(info_key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[metric_key] = float(value)
        reached_agents = final_info.get("reached_agents")
        total_agents = final_info.get("total_agents")
        if (
            isinstance(reached_agents, (int, np.integer))
            and isinstance(total_agents, (int, np.integer))
            and total_agents > 0
        ):
            metrics["robotics/reached_fraction"] = (
                float(reached_agents) / float(total_agents)
            )
        return metrics

    def end_episode(
        self,
        environment_index: int,
        result: EpisodeResult,
        *,
        phase: str | None = None,
    ) -> EpisodeRecord:
        """Finalize metrics, trajectories, and configured visual artifacts."""
        return self._end_episode(environment_index, result, phase=phase)

    def _end_episode(
        self,
        environment_index: int,
        result: EpisodeResult,
        *,
        phase: str | None = None,
    ) -> EpisodeRecord:
        record = super()._end_episode(
            environment_index,
            result,
            phase=phase,
        )
        if (
            self.media_format == "none"
            or not record.observations
            or self.artifact_manager is None
        ):
            return record
        frames = [
            self.renderer.render_observation(observation)
            for observation in record.observations
        ]
        episode_dir = (
            self.artifact_manager.get_final_dir()
            / "episodes"
            / record.context.phase
        )
        formats = (
            ("gif", "mp4")
            if self.media_format == "both"
            else (self.media_format,)
        )
        for media_format in formats:
            media_path = write_animation(
                episode_dir
                / f"{record.context.episode_id}.{media_format}",
                frames,
                fps=self.fps,
            )
            record.artifact_paths[media_format] = str(media_path)
            result.artifact_paths[media_format] = str(media_path)
        return record
