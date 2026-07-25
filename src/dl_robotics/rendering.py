"""Fast RGB rendering and portable episode media."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
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

    def __init__(self, *, cell_size: int = 48, show_grid: bool = True):
        if isinstance(cell_size, bool) or not isinstance(cell_size, int):
            raise TypeError("cell_size must be an integer")
        if cell_size < 8:
            raise ValueError("cell_size must be at least 8")
        if not isinstance(show_grid, bool):
            raise TypeError("show_grid must be a boolean")
        self.cell_size = cell_size
        self.show_grid = show_grid

    def render_observation(self, observation: Any) -> UInt8Image:
        """Render one `[channels, height, width]` semantic observation."""
        return self._render_observation(observation)

    def _render_observation(self, observation: Any) -> UInt8Image:
        values = np.asarray(observation)
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
            inset = max(3, self.cell_size // 7)
            x0 = int(column) * self.cell_size + inset
            y0 = int(row) * self.cell_size + inset
            x1 = (int(column) + 1) * self.cell_size - inset - 1
            y1 = (int(row) + 1) * self.cell_size - inset - 1
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

        for row, column in np.argwhere(values[1] > 0.0):
            identity = identity_lookup[float(values[1, row, column])]
            color = tuple(int(value) for value in self._PALETTE[identity % 8])
            center = (
                int(column) * self.cell_size + self.cell_size // 2,
                int(row) * self.cell_size + self.cell_size // 2,
            )
            cv2.circle(
                frame,
                center,
                max(3, self.cell_size // 3),
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
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
