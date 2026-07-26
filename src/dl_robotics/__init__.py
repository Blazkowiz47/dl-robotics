"""Fast 2D robotics environments for deep-learning-core."""

from .cli import create_robotics_component
from .environment import GridMAPFEnvironment, GridMAPFVectorEnvironment
from .episode_manager import RoboticsEpisodeManager
from .planning import (
    GridPath,
    PathNotFoundError,
    astar_path,
    bfs_path,
    dfs_path,
    dijkstra_path,
)
from .rendering import GridRenderer, write_animation
from .rules import (
    INTERACTION_RULE_REGISTRY,
    make_interaction_rule,
    register_interaction_rule,
)
from .scenario import Coordinate, GridScenario
from .world import ExclusiveCellRule, GridWorldBatch, InteractionRule, StepEvents

__version__ = "0.0.2"

__all__ = [
    "INTERACTION_RULE_REGISTRY",
    "Coordinate",
    "ExclusiveCellRule",
    "GridMAPFEnvironment",
    "GridMAPFVectorEnvironment",
    "GridPath",
    "GridRenderer",
    "GridScenario",
    "GridWorldBatch",
    "InteractionRule",
    "PathNotFoundError",
    "RoboticsEpisodeManager",
    "StepEvents",
    "__version__",
    "astar_path",
    "bfs_path",
    "create_robotics_component",
    "dfs_path",
    "dijkstra_path",
    "make_interaction_rule",
    "register_interaction_rule",
    "write_animation",
]
