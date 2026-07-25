# deep-learning-robotics

Fast, reproducible 2D robotics environments for
[`deep-learning-core`](https://github.com/Blazkowiz47/dl-core).

## Install

```bash
pip install deep-learning-robotics
```

Version `0.0.2` requires `deep-learning-core>=0.0.28,<0.1`.

## What's New in 0.0.2?

- environment setup, action decoding, simulation advancement, classical
  planners, animation output, and episode summaries now keep one-off logic
  inline for a more direct implementation
- public environments, planners, rendering utilities, and episode-manager
  behavior remain unchanged

## What's New in 0.0.1?

- validated grid scenarios with walls, actor starts, and per-actor goals
- preallocated batched world state for position, velocity, acceleration,
  reached goals, path length, makespan, and sum of costs
- simultaneous exclusive-cell physics covering boundaries, walls, vertex
  conflicts, edge swaps, and moves into stationary actors
- scalar and native vector Gymnasium environments registered as
  `robotics_mapf` and `robotics_mapf_vector`
- centralized joint actions compatible with dl-core DQN and PPO
- semantic channel observations containing walls, actors, goals, velocity, and
  acceleration
- headless RGB rendering plus direct GIF and MP4 episode output
- exact A*, Dijkstra, and BFS utilities for static single-agent shortest paths,
  plus deterministic DFS for reachability and debugging
- a `robotics` episode manager for collision, completion, makespan,
  sum-of-costs, path-length, trajectory, and media artifacts

## Environment Configuration

Import `dl_robotics` once to register its environments, then use normal
dl-core configuration:

```yaml
environment:
  name: robotics_mapf_vector
  num_envs: 16
  scenario:
    name: crossing
    width: 7
    height: 7
    max_steps: 40
    walls: [[3, 1], [3, 5]]
    starts: [[1, 1], [5, 5]]
    goals: [[5, 5], [1, 1]]
  rewards:
    step: -0.01
    progress: 0.1
    collision: -0.25
    goal: 1.0
    success: 5.0
  render:
    cell_size: 48
    show_grid: true

episode_managers:
  robotics:
    capture_phases: [evaluation]
    capture_every_n_episodes: 1
    max_captured_episodes: 20
    media_format: both
    fps: 8
```

Each actor chooses one of `stay`, `up`, `right`, `down`, or `left`. The
centralized environment encodes all actor choices into one
`Discrete(5 ** num_agents)` joint action, with actor zero stored in the least
significant base-5 digit. This is intentionally aimed at small cooperative
MAPF problems; larger or decentralized systems should use a future multi-agent
policy API instead of an exponentially growing joint action.

The image observation is suitable for DQN and PPO. dl-core's tabular
Q-learning trainer requires a `Discrete` observation space, so it is not
compatible with this first image-observation environment.

The observation is a float32 tensor with shape `[7, height, width]`: walls,
actor identity, goal identity, row/column velocity, and row/column acceleration.
Episode info exposes `is_success`, collision counts, reached agents, makespan,
sum of costs, and total path length for episode managers and experiment
tracking. `collisions` and its typed variants describe the latest step;
`episode_collisions` and its typed variants retain the episode totals.

## Rendering and Episode Artifacts

`environment.render()` returns RGB `uint8` arrays without opening a display:
`[height, width, 3]` for the scalar environment and
`[num_envs, height, width, 3]` for the vector environment.

The `robotics` episode manager includes dl-core's standard episode metrics and
trajectory capture, so it should be used in place of the `standard` manager.
For selected phases and episode intervals it stores the complete compressed
trajectory and optionally a GIF, MP4, or both. It also emits
`robotics/collisions`, typed collision counts, reached fraction, makespan,
sum of costs, and path length through normal callback and tracker flows.

Media files can also be created directly:

```python
from dl_robotics import write_animation

write_animation("episode.gif", frames, fps=8)
write_animation("episode.mp4", frames, fps=8)
```

## Interaction Rules

`GridWorldBatch` owns numerical state, while `InteractionRule` owns how proposed
movements interact. `ExclusiveCellRule` provides MAPF-safe defaults. A custom
rule instance can be supplied as `environment.interaction_rule` when an
environment is created programmatically, without changing scenario definitions
or RL adapters. Serializable rule registries are planned for a later release.

The first version uses vectorized geometry and preallocated state arrays, with
small per-world conflict-resolution loops where agent dependencies require
them. It does not model continuous rigid-body dynamics, ROS, Gazebo, or 3D
simulation.

## Shortest-Path Baselines

Use A* for efficient exact planning on the unit-cost grid, or Dijkstra when a
heuristic-free reference is useful:

```python
from dl_robotics import (
    GridScenario,
    astar_path,
    bfs_path,
    dfs_path,
    dijkstra_path,
)

scenario = GridScenario(
    width=5,
    height=5,
    starts=((0, 0), (4, 4)),
    goals=((4, 4), (0, 0)),
    walls=((1, 2), (3, 2)),
)

astar = astar_path(scenario, scenario.starts[0], scenario.goals[0])
dijkstra = dijkstra_path(scenario, scenario.starts[1], scenario.goals[1])
bfs = bfs_path(scenario, scenario.starts[0], scenario.goals[0])
dfs = dfs_path(scenario, scenario.starts[1], scenario.goals[1])
```

Paths include both endpoints and use four-direction movement around static
walls. Their move count is therefore `len(path) - 1`. A*, Dijkstra, and BFS
return shortest paths on this unweighted grid. DFS returns the first
depth-first route and does not guarantee optimality. Traversal ties use the
fixed up, right, down, left order. The exact planners provide per-agent lower
bounds and deterministic evaluation baselines; independently planned paths can
still have vertex or edge conflicts and are not, by themselves, a multi-agent
path-finding solver.
