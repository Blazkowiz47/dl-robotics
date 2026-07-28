# deep-learning-robotics Release History

The main README shows only the latest release. This page preserves all
published release notes.

## 0.0.5

- all researcher override points use public names, including observation
  `build()`, environment `build_info()`, and renderer marker hooks
- custom actor and goal markers affect media and optimized resized model
  observations consistently
- renderers accept configurable palettes and expose `actor_color()` for
  dynamic research encodings
- redundant one-use public/private method pairs were removed from environments,
  world physics, scenarios, and scaffold integration

## 0.0.4

- registered observation builders make semantic tensors or shape-controlled
  RGB model/replay inputs a public environment API
- registered renderers independently control environment frames and episode
  media with solid actor and hollow goal circles, squares, or triangles
- resized RGB builders cache static geometry and draw directly at the requested
  resolution, keeping markers visible without large intermediate images
- documentation separates model/replay observations from visual artifacts and
  includes the complete data-flow graph and extension examples

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
