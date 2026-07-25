"""Static grid planning tests."""

from itertools import pairwise

import pytest

from dl_robotics import (
    GridScenario,
    PathNotFoundError,
    astar_path,
    bfs_path,
    dfs_path,
    dijkstra_path,
)


def test_astar_dijkstra_and_bfs_find_equal_shortest_paths() -> None:
    scenario = GridScenario(
        width=5,
        height=4,
        starts=((0, 0),),
        goals=((0, 4),),
        walls=((0, 2), (1, 2), (2, 2)),
    )

    astar = astar_path(scenario, scenario.starts[0], scenario.goals[0])
    dijkstra = dijkstra_path(
        scenario,
        scenario.starts[0],
        scenario.goals[0],
    )
    bfs = bfs_path(scenario, scenario.starts[0], scenario.goals[0])

    assert astar == dijkstra == bfs
    assert astar[0] == scenario.starts[0]
    assert astar[-1] == scenario.goals[0]
    assert len(astar) - 1 == 10
    assert not set(astar).intersection(scenario.walls)
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in pairwise(astar)
    )


@pytest.mark.parametrize(
    "planner",
    [astar_path, dijkstra_path, bfs_path, dfs_path],
)
def test_planners_handle_same_cell_and_unreachable_goals(planner) -> None:
    same_cell = GridScenario(
        width=1,
        height=1,
        starts=((0, 0),),
        goals=((0, 0),),
    )
    blocked = GridScenario(
        width=3,
        height=3,
        starts=((0, 0),),
        goals=((2, 2),),
        walls=((1, 0), (1, 1), (1, 2)),
    )

    assert planner(same_cell, (0, 0), (0, 0)) == ((0, 0),)
    with pytest.raises(PathNotFoundError, match="No path exists"):
        planner(blocked, blocked.starts[0], blocked.goals[0])


@pytest.mark.parametrize(
    "planner",
    [astar_path, dijkstra_path, bfs_path, dfs_path],
)
def test_planners_validate_arbitrary_endpoints(planner) -> None:
    scenario = GridScenario(
        width=3,
        height=3,
        starts=((0, 0),),
        goals=((2, 2),),
        walls=((1, 1),),
    )

    with pytest.raises(ValueError, match="out of bounds"):
        planner(scenario, (-1, 0), (2, 2))
    with pytest.raises(ValueError, match="is a wall"):
        planner(scenario, (0, 0), (1, 1))
    with pytest.raises(TypeError, match="must be a"):
        planner(scenario, [0, 0], (2, 2))  # type: ignore[arg-type]


def test_dfs_returns_a_valid_path_without_promising_the_shortest_path() -> None:
    scenario = GridScenario(
        width=4,
        height=4,
        starts=((0, 0),),
        goals=((3, 0),),
    )

    bfs = bfs_path(scenario, scenario.starts[0], scenario.goals[0])
    dfs = dfs_path(scenario, scenario.starts[0], scenario.goals[0])

    assert len(bfs) - 1 == 3
    assert len(dfs) > len(bfs)
    assert dfs[0] == scenario.starts[0]
    assert dfs[-1] == scenario.goals[0]
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in pairwise(dfs)
    )


@pytest.mark.parametrize(
    "planner",
    [astar_path, dijkstra_path, bfs_path, dfs_path],
)
def test_planners_resolve_ties_in_a_deterministic_direction_order(
    planner,
) -> None:
    scenario = GridScenario(
        width=3,
        height=3,
        starts=((1, 1),),
        goals=((0, 2),),
    )
    expected = ((1, 1), (0, 1), (0, 2))

    assert planner(scenario, (1, 1), (0, 2)) == expected
    assert planner(scenario, (1, 1), (0, 2)) == expected
