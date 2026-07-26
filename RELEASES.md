# deep-learning-robotics Release History

The main README shows only the latest release. This page preserves all
published release notes.

## 0.0.3

- `dl-init --with-robotics` now adds `deep-learning-robotics`, a runnable
  `configs/robotics.yaml`, and organized `environments`, `rules`, `scenarios`,
  `callbacks`, and `episode_managers` packages to a normal dl-core experiment
- `dl-robotics add environment|rule|scenario NAME` creates robotics-specific
  local components without replacing dl-core's existing generators for models,
  trainers, callbacks, and episode managers
- interaction rules can be selected by registered name or YAML mapping while
  existing programmatic `InteractionRule` instances remain supported

## 0.0.2

- environment setup, action decoding, simulation advancement, classical
  planners, animation output, and episode summaries now keep one-off logic
  inline for a more direct implementation
- public environments, planners, rendering utilities, and episode-manager
  behavior remain unchanged

## 0.0.1

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
