"""Tests for the robotics dl-init extension."""

from __future__ import annotations

import argparse
from pathlib import Path

from dl_core.init_extensions import ProjectNames, ScaffoldContext

from dl_robotics.init_extension import RoboticsInitExtension


def test_robotics_init_extension_updates_scaffold(tmp_path: Path) -> None:
    """The extension should add a runnable, organized robotics project layer."""
    context = ScaffoldContext(
        target_dir=tmp_path,
        templates_dir=tmp_path,
        project=ProjectNames(
            project_name="demo",
            project_slug="demo",
            component_name="demo",
            dataset_name="demo",
            dataset_class_name="DemoDataset",
            model_name="resnet_example",
            model_class_name="ResNetExample",
            trainer_name="demo",
            trainer_class_name="DemoTrainer",
        ),
        files={
            Path("pyproject.toml"): (
                "[project]\n"
                "dependencies = [\n"
                '    "deep-learning-core",\n'
                "]\n"
            ),
            Path("README.md"): "# demo\n",
            Path("src") / "bootstrap.py": (
                '"""Project bootstrap hooks for local component loading."""\n'
            ),
        },
        enabled_extensions={"robotics"},
    )

    RoboticsInitExtension().apply(context)

    assert (
        '"deep-learning-robotics>=0.0.4,<0.1"'
        in context.get_file("pyproject.toml")
    )
    bootstrap = context.get_file(Path("src") / "bootstrap.py")
    assert "import dl_robotics" in bootstrap
    assert "import rules" in bootstrap
    assert "import scenarios" in bootstrap
    assert "robotics_mapf_vector" in context.get_file(
        Path("configs") / "robotics.yaml"
    )
    for package_name in (
        "callbacks",
        "environments",
        "episode_managers",
        "rules",
        "scenarios",
    ):
        assert Path("src") / package_name / "__init__.py" in context.files


def test_robotics_init_extension_requires_explicit_selection() -> None:
    """Installing the package alone should not alter every dl-core scaffold."""
    extension = RoboticsInitExtension()

    assert not extension.is_enabled(argparse.Namespace(), {"robotics": extension})
    assert extension.is_enabled(
        argparse.Namespace(with_robotics=True),
        {"robotics": extension},
    )
