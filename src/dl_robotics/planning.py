"""Exact shortest-path utilities for static 2D grid geometry."""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from collections.abc import Iterator

from .scenario import Coordinate, GridScenario

GridPath = tuple[Coordinate, ...]
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


class PathNotFoundError(ValueError):
    """Raised when no traversable path connects two grid cells."""


def _validate_endpoints(
    scenario: GridScenario,
    start: Coordinate,
    goal: Coordinate,
) -> None:
    for label, coordinate in (("start", start), ("goal", goal)):
        if not isinstance(coordinate, tuple) or len(coordinate) != 2:
            raise TypeError(f"{label} must be a (row, column) tuple")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in coordinate
        ):
            raise TypeError(f"{label} coordinates must contain integers")
        row, column = coordinate
        if not 0 <= row < scenario.height or not 0 <= column < scenario.width:
            raise ValueError(f"{label} coordinate {coordinate} is out of bounds")
        if coordinate in scenario.walls:
            raise ValueError(f"{label} coordinate {coordinate} is a wall")


def _neighbors(
    scenario: GridScenario,
    coordinate: Coordinate,
    walls: set[Coordinate],
) -> Iterator[Coordinate]:
    for row_delta, column_delta in _DIRECTIONS:
        neighbor = (
            coordinate[0] + row_delta,
            coordinate[1] + column_delta,
        )
        if (
            0 <= neighbor[0] < scenario.height
            and 0 <= neighbor[1] < scenario.width
            and neighbor not in walls
        ):
            yield neighbor


def _reconstruct_path(
    parents: dict[Coordinate, Coordinate],
    start: Coordinate,
    goal: Coordinate,
) -> GridPath:
    path = [goal]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    return tuple(path)


def astar_path(
    scenario: GridScenario,
    start: Coordinate,
    goal: Coordinate,
) -> GridPath:
    """Return an exact shortest path using the Manhattan A* heuristic."""
    _validate_endpoints(scenario, start, goal)
    if start == goal:
        return (start,)

    sequence = itertools.count()
    frontier = [
        (
            abs(start[0] - goal[0]) + abs(start[1] - goal[1]),
            0,
            next(sequence),
            start,
        )
    ]
    best_cost = {start: 0}
    parents: dict[Coordinate, Coordinate] = {}
    walls = set(scenario.walls)

    while frontier:
        _, cost, _, current = heapq.heappop(frontier)
        if cost != best_cost[current]:
            continue
        if current == goal:
            return _reconstruct_path(parents, start, goal)
        for neighbor in _neighbors(scenario, current, walls):
            next_cost = cost + 1
            if next_cost >= best_cost.get(
                neighbor,
                scenario.width * scenario.height + 1,
            ):
                continue
            best_cost[neighbor] = next_cost
            parents[neighbor] = current
            priority = (
                next_cost
                + abs(neighbor[0] - goal[0])
                + abs(neighbor[1] - goal[1])
            )
            heapq.heappush(
                frontier,
                (priority, next_cost, next(sequence), neighbor),
            )

    raise PathNotFoundError(f"No path exists between {start} and {goal}")


def dijkstra_path(
    scenario: GridScenario,
    start: Coordinate,
    goal: Coordinate,
) -> GridPath:
    """Return an exact shortest path using Dijkstra's algorithm."""
    _validate_endpoints(scenario, start, goal)
    if start == goal:
        return (start,)

    sequence = itertools.count()
    frontier = [(0, next(sequence), start)]
    best_cost = {start: 0}
    parents: dict[Coordinate, Coordinate] = {}
    walls = set(scenario.walls)

    while frontier:
        cost, _, current = heapq.heappop(frontier)
        if cost != best_cost[current]:
            continue
        if current == goal:
            return _reconstruct_path(parents, start, goal)
        for neighbor in _neighbors(scenario, current, walls):
            next_cost = cost + 1
            if next_cost >= best_cost.get(
                neighbor,
                scenario.width * scenario.height + 1,
            ):
                continue
            best_cost[neighbor] = next_cost
            parents[neighbor] = current
            heapq.heappush(
                frontier,
                (next_cost, next(sequence), neighbor),
            )

    raise PathNotFoundError(f"No path exists between {start} and {goal}")


def bfs_path(
    scenario: GridScenario,
    start: Coordinate,
    goal: Coordinate,
) -> GridPath:
    """Return an exact shortest path using breadth-first search."""
    _validate_endpoints(scenario, start, goal)
    if start == goal:
        return (start,)

    frontier = deque([start])
    visited = {start}
    parents: dict[Coordinate, Coordinate] = {}
    walls = set(scenario.walls)

    while frontier:
        current = frontier.popleft()
        for neighbor in _neighbors(scenario, current, walls):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parents[neighbor] = current
            if neighbor == goal:
                return _reconstruct_path(parents, start, goal)
            frontier.append(neighbor)

    raise PathNotFoundError(f"No path exists between {start} and {goal}")


def dfs_path(
    scenario: GridScenario,
    start: Coordinate,
    goal: Coordinate,
) -> GridPath:
    """Return a deterministic depth-first path without optimality guarantees."""
    _validate_endpoints(scenario, start, goal)
    if start == goal:
        return (start,)

    frontier: list[tuple[Coordinate, Coordinate | None]] = [(start, None)]
    visited: set[Coordinate] = set()
    parents: dict[Coordinate, Coordinate] = {}
    walls = set(scenario.walls)

    while frontier:
        current, parent = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        if parent is not None:
            parents[current] = parent
        if current == goal:
            return _reconstruct_path(parents, start, goal)
        for neighbor in reversed(tuple(_neighbors(scenario, current, walls))):
            if neighbor not in visited:
                frontier.append((neighbor, current))

    raise PathNotFoundError(f"No path exists between {start} and {goal}")
