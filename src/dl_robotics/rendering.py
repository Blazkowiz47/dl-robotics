"""Fast RGB rendering and portable episode media."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from dl_core.core import ComponentRegistry
from numpy.typing import NDArray

from .world import GridWorldBatch

UInt8Image = NDArray[np.uint8]


class GridRenderer:
    """Render semantic observations or live worlds without a display server."""

    _PALETTE = np.asarray(
        [
            [37, 99, 235],
            [220, 38, 38],
            [22, 163, 74],
            [147, 51, 234],
            [234, 88, 12],
            [8, 145, 178],
            [190, 24, 93],
            [101, 163, 13],
        ],
        dtype=np.uint8,
    )

    def __init__(
        self,
        *,
        cell_size: int = 48,
        show_grid: bool = True,
        actor_shape: str = "circle",
        goal_shape: str = "square",
        show_actor_ids: bool = True,
    ):
        if isinstance(cell_size, bool) or not isinstance(cell_size, int):
            raise TypeError("cell_size must be an integer")
        if cell_size < 8:
            raise ValueError("cell_size must be at least 8")
        if not isinstance(show_grid, bool):
            raise TypeError("show_grid must be a boolean")
        supported_shapes = {"circle", "square", "triangle"}
        if actor_shape not in supported_shapes:
            raise ValueError(
                "actor_shape must be circle, square, or triangle"
            )
        if goal_shape not in supported_shapes:
            raise ValueError(
                "goal_shape must be circle, square, or triangle"
            )
        if not isinstance(show_actor_ids, bool):
            raise TypeError("show_actor_ids must be a boolean")
        self.cell_size = cell_size
        self.show_grid = show_grid
        self.actor_shape = actor_shape
        self.goal_shape = goal_shape
        self.show_actor_ids = show_actor_ids
        self._resized_backgrounds: dict[tuple[str, int], UInt8Image] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GridRenderer:
        """Create a renderer from its YAML-compatible configuration."""
        return cls(**dict(config))

    def draw_goal(
        self,
        frame: UInt8Image,
        *,
        row: int,
        column: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw one hollow goal marker."""
        self._draw_goal(
            frame,
            row=row,
            column=column,
            color=color,
        )

    def _draw_goal(
        self,
        frame: UInt8Image,
        *,
        row: int,
        column: int,
        color: tuple[int, int, int],
    ) -> None:
        inset = max(3, self.cell_size // 7)
        x0 = column * self.cell_size + inset
        y0 = row * self.cell_size + inset
        x1 = (column + 1) * self.cell_size - inset - 1
        y1 = (row + 1) * self.cell_size - inset - 1
        thickness = max(2, self.cell_size // 24)
        if self.goal_shape == "circle":
            cv2.circle(
                frame,
                ((x0 + x1) // 2, (y0 + y1) // 2),
                max(3, (x1 - x0) // 2),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        elif self.goal_shape == "triangle":
            points = np.asarray(
                [
                    [(x0 + x1) // 2, y0],
                    [x1, y1],
                    [x0, y1],
                ],
                dtype=np.int32,
            )
            cv2.polylines(
                frame,
                [points],
                True,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)

    def draw_actor(
        self,
        frame: UInt8Image,
        *,
        row: int,
        column: int,
        color: tuple[int, int, int],
        identity: int,
    ) -> None:
        """Draw one solid actor marker."""
        self._draw_actor(
            frame,
            row=row,
            column=column,
            color=color,
            identity=identity,
        )

    def _draw_actor(
        self,
        frame: UInt8Image,
        *,
        row: int,
        column: int,
        color: tuple[int, int, int],
        identity: int,
    ) -> None:
        center = (
            column * self.cell_size + self.cell_size // 2,
            row * self.cell_size + self.cell_size // 2,
        )
        radius = max(3, self.cell_size // 3)
        if self.actor_shape == "square":
            cv2.rectangle(
                frame,
                (center[0] - radius, center[1] - radius),
                (center[0] + radius, center[1] + radius),
                color,
                -1,
            )
        elif self.actor_shape == "triangle":
            points = np.asarray(
                [
                    [center[0], center[1] - radius],
                    [center[0] + radius, center[1] + radius],
                    [center[0] - radius, center[1] + radius],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(
                frame,
                [points],
                color,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.circle(
                frame,
                center,
                radius,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
        if self.show_actor_ids:
            cv2.putText(
                frame,
                str(identity),
                (
                    center[0] - self.cell_size // 9,
                    center[1] + self.cell_size // 9,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.3, self.cell_size / 90.0),
                (255, 255, 255),
                max(1, self.cell_size // 24),
                cv2.LINE_AA,
            )

    def render_observation(self, observation: Any) -> UInt8Image:
        """Render one `[channels, height, width]` semantic observation."""
        return self._render_observation(observation)

    def _render_observation(self, observation: Any) -> UInt8Image:
        values = np.asarray(observation)
        if (
            values.ndim == 3
            and values.shape[-1] == 3
            and values.dtype == np.uint8
        ):
            return values.copy()
        if values.ndim != 3 or values.shape[0] < 3:
            raise ValueError(
                "Semantic observation must have shape [channels, height, width]"
            )
        if values.shape[1] == 0 or values.shape[2] == 0:
            raise ValueError("Semantic observation height and width must be positive")
        if not np.isfinite(values).all():
            raise ValueError("Semantic observation must be finite")
        height, width = values.shape[1:]
        frame = np.full(
            (height * self.cell_size, width * self.cell_size, 3),
            248,
            dtype=np.uint8,
        )
        for row, column in np.argwhere(values[0] > 0.5):
            y0 = int(row) * self.cell_size
            x0 = int(column) * self.cell_size
            frame[
                y0 : y0 + self.cell_size,
                x0 : x0 + self.cell_size,
            ] = (45, 55, 72)

        identity_values = sorted(
            {
                float(value)
                for value in np.concatenate(
                    (np.ravel(values[1]), np.ravel(values[2]))
                )
                if value > 0.0
            }
        )
        identity_lookup = {
            value: index for index, value in enumerate(identity_values)
        }
        for row, column in np.argwhere(values[2] > 0.0):
            identity = identity_lookup[float(values[2, row, column])]
            color = tuple(int(value) for value in self._PALETTE[identity % 8])
            self.draw_goal(
                frame,
                row=int(row),
                column=int(column),
                color=color,
            )

        for row, column in np.argwhere(values[1] > 0.0):
            identity = identity_lookup[float(values[1, row, column])]
            color = tuple(int(value) for value in self._PALETTE[identity % 8])
            self.draw_actor(
                frame,
                row=int(row),
                column=int(column),
                color=color,
                identity=identity,
            )

        if self.show_grid:
            for row in range(height + 1):
                y = min(row * self.cell_size, frame.shape[0] - 1)
                cv2.line(frame, (0, y), (frame.shape[1] - 1, y), (205, 211, 220), 1)
            for column in range(width + 1):
                x = min(column * self.cell_size, frame.shape[1] - 1)
                cv2.line(frame, (x, 0), (x, frame.shape[0] - 1), (205, 211, 220), 1)
        return frame

    def render_world(self, world: GridWorldBatch, world_index: int = 0) -> UInt8Image:
        """Render one live world using its actor and goal state."""
        return self._render_world(world, world_index)

    def _render_world(
        self,
        world: GridWorldBatch,
        world_index: int = 0,
    ) -> UInt8Image:
        if isinstance(world_index, bool) or not isinstance(world_index, int):
            raise TypeError("world_index must be an integer")
        if not 0 <= world_index < world.num_worlds:
            raise IndexError("world_index is out of range")
        observation = np.zeros(
            (3, world.scenario.height, world.scenario.width),
            dtype=np.float32,
        )
        observation[0] = world.wall_mask
        scale = float(world.scenario.num_agents)
        for actor_index in range(world.scenario.num_agents):
            row, column = world.positions[world_index, actor_index]
            observation[1, row, column] = (actor_index + 1) / scale
            goal_row, goal_column = world.goal_positions[actor_index]
            observation[2, goal_row, goal_column] = (actor_index + 1) / scale
        return self._render_observation(observation)

    def render_world_at_size(
        self,
        world: GridWorldBatch,
        output_size: int,
        world_index: int = 0,
    ) -> UInt8Image:
        """Render directly at a square output size without a large intermediate."""
        return self._render_world_at_size(world, output_size, world_index)

    def _render_world_at_size(
        self,
        world: GridWorldBatch,
        output_size: int,
        world_index: int = 0,
    ) -> UInt8Image:
        if isinstance(output_size, bool) or not isinstance(output_size, int):
            raise TypeError("output_size must be an integer")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        if isinstance(world_index, bool) or not isinstance(world_index, int):
            raise TypeError("world_index must be an integer")
        if not 0 <= world_index < world.num_worlds:
            raise IndexError("world_index is out of range")

        cache_key = (world.scenario.fingerprint, output_size)
        background = self._resized_backgrounds.get(cache_key)
        if background is None:
            wall_density = cv2.resize(
                world.wall_mask.astype(np.uint8) * 255,
                (output_size, output_size),
                interpolation=cv2.INTER_AREA,
            )
            background = np.full(
                (output_size, output_size, 3),
                248,
                dtype=np.uint8,
            )
            background[wall_density >= 32] = (45, 55, 72)
            if (
                self.show_grid
                and output_size / world.scenario.height >= 4
                and output_size / world.scenario.width >= 4
            ):
                for row in range(world.scenario.height + 1):
                    y = min(
                        round(row * output_size / world.scenario.height),
                        output_size - 1,
                    )
                    cv2.line(
                        background,
                        (0, y),
                        (output_size - 1, y),
                        (205, 211, 220),
                        1,
                    )
                for column in range(world.scenario.width + 1):
                    x = min(
                        round(column * output_size / world.scenario.width),
                        output_size - 1,
                    )
                    cv2.line(
                        background,
                        (x, 0),
                        (x, output_size - 1),
                        (205, 211, 220),
                        1,
                    )
            row_scale = (output_size - 1) / max(
                world.scenario.height - 1,
                1,
            )
            column_scale = (output_size - 1) / max(
                world.scenario.width - 1,
                1,
            )
            radius = max(2, output_size // 128)
            thickness = max(1, radius // 2)
            for actor_index, (goal_row, goal_column) in enumerate(
                world.goal_positions
            ):
                center = (
                    round(int(goal_column) * column_scale),
                    round(int(goal_row) * row_scale),
                )
                color = tuple(
                    int(value) for value in self._PALETTE[actor_index % 8]
                )
                if self.goal_shape == "square":
                    cv2.rectangle(
                        background,
                        (center[0] - radius, center[1] - radius),
                        (center[0] + radius, center[1] + radius),
                        color,
                        thickness,
                    )
                elif self.goal_shape == "triangle":
                    points = np.asarray(
                        [
                            [center[0], center[1] - radius],
                            [center[0] + radius, center[1] + radius],
                            [center[0] - radius, center[1] + radius],
                        ],
                        dtype=np.int32,
                    )
                    cv2.polylines(
                        background,
                        [points],
                        True,
                        color,
                        thickness,
                        lineType=cv2.LINE_AA,
                    )
                else:
                    cv2.circle(
                        background,
                        center,
                        radius,
                        color,
                        thickness,
                        lineType=cv2.LINE_AA,
                    )
            self._resized_backgrounds[cache_key] = background

        frame = background.copy()
        row_scale = (output_size - 1) / max(world.scenario.height - 1, 1)
        column_scale = (output_size - 1) / max(world.scenario.width - 1, 1)
        radius = max(2, output_size // 128)
        for actor_index, (actor_row, actor_column) in enumerate(
            world.positions[world_index]
        ):
            center = (
                round(int(actor_column) * column_scale),
                round(int(actor_row) * row_scale),
            )
            color = tuple(
                int(value) for value in self._PALETTE[actor_index % 8]
            )
            if self.actor_shape == "square":
                cv2.rectangle(
                    frame,
                    (center[0] - radius, center[1] - radius),
                    (center[0] + radius, center[1] + radius),
                    color,
                    -1,
                )
            elif self.actor_shape == "triangle":
                points = np.asarray(
                    [
                        [center[0], center[1] - radius],
                        [center[0] + radius, center[1] + radius],
                        [center[0] - radius, center[1] + radius],
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(
                    frame,
                    [points],
                    color,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.circle(
                    frame,
                    center,
                    radius,
                    color,
                    -1,
                    lineType=cv2.LINE_AA,
                )
            if self.show_actor_ids and radius >= 6:
                cv2.putText(
                    frame,
                    str(actor_index),
                    (center[0] - radius // 2, center[1] + radius // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.3, radius / 20.0),
                    (255, 255, 255),
                    max(1, radius // 6),
                    cv2.LINE_AA,
                )
        return frame


GRID_RENDERER_REGISTRY = ComponentRegistry("Grid renderer")
GRID_RENDERER_REGISTRY.register_class("grid", GridRenderer)
GRID_RENDERER_REGISTRY.register_class("default", GridRenderer)


def register_grid_renderer(names: str | list[str]):
    """Register a grid-renderer class under one or more names."""

    def decorator(renderer_class: type[GridRenderer]) -> type[GridRenderer]:
        if not isinstance(renderer_class, type) or not issubclass(
            renderer_class,
            GridRenderer,
        ):
            raise TypeError(
                "Registered grid renderers must inherit GridRenderer"
            )
        GRID_RENDERER_REGISTRY.register(names)(renderer_class)
        return renderer_class

    return decorator


def make_grid_renderer(
    config: str | Mapping[str, Any] | GridRenderer | None,
) -> GridRenderer:
    """Create a renderer from config or an existing instance."""
    if config is None:
        return GridRenderer()
    if isinstance(config, GridRenderer):
        return config
    if isinstance(config, str):
        renderer_name = config
        renderer_config: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        renderer_name = config.get("name", "grid")
        if not isinstance(renderer_name, str) or not renderer_name:
            raise ValueError("render.name must be a non-empty string")
        renderer_config = {
            key: value for key, value in config.items() if key != "name"
        }
    else:
        raise TypeError(
            "render must be a registered name, mapping, or GridRenderer instance"
        )
    registered_renderers = GRID_RENDERER_REGISTRY.registered_items()
    if renderer_name not in registered_renderers:
        raise NotImplementedError(
            f"Grid renderer '{renderer_name}' not found. Available grid "
            f"renderers: {list(registered_renderers)}"
        )
    renderer_class = registered_renderers[renderer_name]
    if not issubclass(renderer_class, GridRenderer):
        raise TypeError(
            f"Registered grid renderer '{renderer_name}' must inherit GridRenderer"
        )
    return renderer_class.from_config(renderer_config)


def write_animation(
    path: str | Path,
    frames: list[Any] | NDArray[np.uint8],
    *,
    fps: int = 8,
) -> Path:
    """Write RGB frames to a GIF or MP4 selected by the path suffix."""
    if isinstance(fps, bool) or not isinstance(fps, int):
        raise TypeError("fps must be an integer")
    if fps <= 0:
        raise ValueError("fps must be positive")
    output_path = Path(path)
    try:
        frame_array = np.asarray(frames)
    except ValueError as error:
        raise ValueError(
            "frames must have consistent [height, width, 3] shapes"
        ) from error
    if (
        frame_array.ndim != 4
        or frame_array.shape[-1] != 3
        or frame_array.shape[0] == 0
    ):
        raise ValueError("frames must have shape [time, height, width, 3]")
    if frame_array.shape[1] == 0 or frame_array.shape[2] == 0:
        raise ValueError("frame height and width must be positive")
    if frame_array.dtype != np.uint8:
        raise TypeError("frames must use the uint8 dtype")
    suffix = output_path.suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        raise ValueError("Animation path must end in .gif or .mp4")
    if suffix == ".gif" and fps > 100:
        raise ValueError("GIF fps cannot exceed 100")
    if output_path.exists() and not output_path.is_file():
        raise ValueError("Animation path must not be an existing directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".gif":
        imageio.mimsave(
            output_path,
            frame_array,
            duration=1000.0 / fps,
            loop=0,
        )
    else:
        height_padding = frame_array.shape[1] % 2
        width_padding = frame_array.shape[2] % 2
        if height_padding or width_padding:
            frame_array = np.pad(
                frame_array,
                (
                    (0, 0),
                    (0, height_padding),
                    (0, width_padding),
                    (0, 0),
                ),
                mode="edge",
            )
        imageio.mimsave(
            output_path,
            frame_array,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
        )
    return output_path
