"""Validated 2D grid scenarios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

Coordinate = tuple[int, int]


@dataclass(frozen=True, slots=True)
class GridScenario:
    """Static geometry and task definition for one MAPF world."""

    width: int
    height: int
    starts: tuple[Coordinate, ...]
    goals: tuple[Coordinate, ...]
    walls: tuple[Coordinate, ...] = ()
    max_steps: int = 100
    lock_agents_at_goal: bool = True
    name: str = "grid"

    def __post_init__(self) -> None:
        for label, value in (
            ("width", self.width),
            ("height", self.height),
            ("max_steps", self.max_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Scenario {label} must be an integer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Scenario width and height must be positive")
        if not self.starts:
            raise ValueError("A MAPF scenario requires at least one agent")
        if len(self.starts) != len(self.goals):
            raise ValueError("Scenario starts and goals must have equal lengths")
        if len(self.starts) > 8:
            raise ValueError(
                "The centralized joint-action environment supports at most 8 agents"
            )
        if self.max_steps <= 0:
            raise ValueError("Scenario max_steps must be positive")
        if not isinstance(self.lock_agents_at_goal, bool):
            raise TypeError("Scenario lock_agents_at_goal must be a boolean")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Scenario name must be a non-empty string")
        for label, coordinates in (
            ("start", self.starts),
            ("goal", self.goals),
            ("wall", self.walls),
        ):
            if not isinstance(coordinates, tuple):
                raise TypeError(
                    f"Scenario {label} coordinates must be stored in a tuple"
                )
            for coordinate in coordinates:
                if not isinstance(coordinate, tuple) or len(coordinate) != 2:
                    raise ValueError(
                        f"Scenario {label} coordinates must be (row, column) pairs"
                    )
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in coordinate
                ):
                    raise TypeError(
                        f"Scenario {label} coordinates must contain integers"
                    )
        if len(set(self.starts)) != len(self.starts):
            raise ValueError("Agent start cells must be unique")
        if len(set(self.goals)) != len(self.goals):
            raise ValueError("Agent goal cells must be unique")
        if len(set(self.walls)) != len(self.walls):
            raise ValueError("Wall cells must be unique")
        wall_cells = set(self.walls)
        for label, coordinates in (
            ("start", self.starts),
            ("goal", self.goals),
            ("wall", self.walls),
        ):
            for row, column in coordinates:
                if not 0 <= row < self.height or not 0 <= column < self.width:
                    raise ValueError(
                        f"Scenario {label} coordinate {(row, column)} is out of bounds"
                    )
        if wall_cells.intersection(self.starts):
            raise ValueError("Walls cannot overlap agent starts")
        if wall_cells.intersection(self.goals):
            raise ValueError("Walls cannot overlap agent goals")

    @property
    def num_agents(self) -> int:
        """Return the number of actors in the scenario."""
        return len(self.starts)

    @property
    def fingerprint(self) -> str:
        """Return a stable identity for experiment artifacts."""
        payload = {
            "width": self.width,
            "height": self.height,
            "starts": self.starts,
            "goals": self.goals,
            "walls": self.walls,
            "max_steps": self.max_steps,
            "lock_agents_at_goal": self.lock_agents_at_goal,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GridScenario:
        """Build a scenario from a serializable configuration mapping."""
        return cls._from_config(config)

    @classmethod
    def _from_config(cls, config: dict[str, Any]) -> GridScenario:
        if not isinstance(config, dict):
            raise TypeError("scenario must be a mapping")
        starts = config.get("starts")
        goals = config.get("goals")
        if not isinstance(starts, (list, tuple)):
            raise TypeError("scenario.starts must be a sequence")
        if not isinstance(goals, (list, tuple)):
            raise TypeError("scenario.goals must be a sequence")
        walls = config.get("walls", ())
        if not isinstance(walls, (list, tuple)):
            raise TypeError("scenario.walls must be a sequence")
        return cls(
            width=config.get("width", 8),
            height=config.get("height", 8),
            starts=tuple(tuple(coordinate) for coordinate in starts),
            goals=tuple(tuple(coordinate) for coordinate in goals),
            walls=tuple(tuple(coordinate) for coordinate in walls),
            max_steps=config.get("max_steps", 100),
            lock_agents_at_goal=config.get("lock_agents_at_goal", True),
            name=config.get("name", "grid"),
        )
