"""Fast 2D robotics environments for deep-learning-core."""

from .cli import create_robotics_component
from .environment import GridMAPFEnvironment, GridMAPFVectorEnvironment
from .episode_manager import RoboticsEpisodeManager
from .observations import (
    OBSERVATION_BUILDER_REGISTRY,
    GridObservationBuilder,
    RenderedGridObservationBuilder,
    SemanticGridObservationBuilder,
    make_observation_builder,
    register_observation_builder,
)
from .planning import (
    GridPath,
    PathNotFoundError,
    astar_path,
    bfs_path,
    dfs_path,
    dijkstra_path,
)
from .rendering import (
    GRID_RENDERER_REGISTRY,
    GridRenderer,
    make_grid_renderer,
    register_grid_renderer,
    write_animation,
)
from .rules import (
    INTERACTION_RULE_REGISTRY,
    make_interaction_rule,
    register_interaction_rule,
)
from .scenario import Coordinate, GridScenario
from .world import ExclusiveCellRule, GridWorldBatch, InteractionRule, StepEvents

__version__ = "0.0.3"

__all__ = [
    "GRID_RENDERER_REGISTRY",
    "INTERACTION_RULE_REGISTRY",
    "OBSERVATION_BUILDER_REGISTRY",
    "Coordinate",
    "ExclusiveCellRule",
    "GridMAPFEnvironment",
    "GridMAPFVectorEnvironment",
    "GridObservationBuilder",
    "GridPath",
    "GridRenderer",
    "GridScenario",
    "GridWorldBatch",
    "InteractionRule",
    "PathNotFoundError",
    "RenderedGridObservationBuilder",
    "RoboticsEpisodeManager",
    "SemanticGridObservationBuilder",
    "StepEvents",
    "__version__",
    "astar_path",
    "bfs_path",
    "create_robotics_component",
    "dfs_path",
    "dijkstra_path",
    "make_grid_renderer",
    "make_interaction_rule",
    "make_observation_builder",
    "register_grid_renderer",
    "register_interaction_rule",
    "register_observation_builder",
    "write_animation",
]
