"""Robotics project scaffolding for dl-init."""

from __future__ import annotations

import argparse
from pathlib import Path

from dl_core.init_extensions import InitExtension, ScaffoldContext


class RoboticsInitExtension(InitExtension):
    """Add robotics dependencies, folders, and configuration to a project."""

    name = "robotics"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register robotics scaffold flags."""
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--with-robotics",
            dest="with_robotics",
            action="store_true",
            default=None,
            help="Include robotics environments and MAPF configuration.",
        )
        group.add_argument(
            "--without-robotics",
            dest="with_robotics",
            action="store_false",
            default=None,
            help="Exclude robotics scaffolding when dl-robotics is installed.",
        )

    def is_enabled(
        self,
        args: argparse.Namespace,
        discovered_extensions: dict[str, InitExtension],
    ) -> bool:
        """Enable robotics scaffolding when explicitly requested."""
        del discovered_extensions
        return self.selection_state(args) is True

    def apply(self, context: ScaffoldContext) -> None:
        """Add robotics project files to the generated scaffold."""
        context.add_dependency("deep-learning-robotics>=0.0.6,<0.1")
        context.append_bootstrap_import("import dl_robotics  # noqa: F401")
        context.append_bootstrap_import("import rules  # noqa: F401")
        context.append_bootstrap_import("import scenarios  # noqa: F401")
        context.append_readme_note(
            "Robotics support is enabled. Start with `configs/robotics.yaml`, "
            "and use `dl-robotics add environment|rule|scenario NAME` for "
            "project-specific components."
        )
        for package_name, docstring in (
            ("callbacks", "Experiment callback extensions."),
            ("environments", "Robotics environment extensions."),
            ("episode_managers", "Robotics episode summary extensions."),
            ("rules", "Actor interaction rule extensions."),
            ("scenarios", "Reusable robotics scenario definitions."),
        ):
            relative_path = Path("src") / package_name / "__init__.py"
            if relative_path not in context.files:
                context.set_file(relative_path, f'"""{docstring}"""\n')
        context.set_file(
            Path("configs") / "robotics.yaml",
            """seed: 2026
deterministic: true

runtime:
  output_dir: artifacts
  log_level: INFO

experiment:
  description: "Centralized DQN on a two-agent MAPF task"

accelerator:
  type: cpu

environment:
  name: robotics_mapf_vector
  num_envs: 4
  scenario:
    name: two_agent_crossing
    width: 5
    height: 5
    max_steps: 20
    starts: [[0, 0], [4, 4]]
    goals: [[4, 4], [0, 0]]
    walls: [[1, 2], [3, 2]]
  rewards:
    step: -0.01
    progress: 0.10
    collision: -0.25
    goal: 1.0
    success: 5.0
  render:
    cell_size: 32
    show_grid: true
    actor_shape: circle
    goal_shape: square

evaluation_environment:
  name: robotics_mapf
  scenario:
    name: two_agent_crossing
    width: 5
    height: 5
    max_steps: 20
    starts: [[0, 0], [4, 4]]
    goals: [[4, 4], [0, 0]]
    walls: [[1, 2], [3, 2]]
  rewards:
    step: -0.01
    progress: 0.10
    collision: -0.25
    goal: 1.0
    success: 5.0
  render:
    cell_size: 32
    show_grid: true
    actor_shape: circle
    goal_shape: square

models:
  q_network:
    name: dqn_mlp
    hidden_sizes: [128, 128]

optimizers:
  name: adam
  lr: 0.001

trainer:
  dqn:
    total_timesteps: 10000
    max_episode_steps: 20
    evaluation_frequency: 1000
    evaluation_episodes: 5
    checkpoint_frequency: 5000
    gamma: 0.99
    buffer_size: 10000
    batch_size: 64
    learning_starts: 500
    train_frequency: 4
    gradient_steps: 1
    target_update_frequency: 500
    double_dqn: true
    epsilon_start: 1.0
    epsilon_end: 0.05
    epsilon_decay_steps: 8000

episode_managers:
  robotics:
    capture_phases: [evaluation]
    capture_every_n_episodes: 1
    max_captured_episodes: 5
    media_format: gif
    fps: 8
    cell_size: 32

callbacks:
  local_metric_tracker:
    log_frequency: 10
""",
        )
