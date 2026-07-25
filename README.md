# deep-learning-robotics

Fast, reproducible 2D robotics environments for
[`deep-learning-core`](https://github.com/Blazkowiz47/dl-core).

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
