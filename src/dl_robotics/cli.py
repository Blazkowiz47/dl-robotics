"""Command-line helpers for robotics experiment projects."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from dl_core.project import find_local_component_root_dir, find_project_root


def create_robotics_component(
    component_type: str,
    name: str,
    *,
    root_dir: str = ".",
    force: bool = False,
) -> Path:
    """Create a robotics-specific component in a dl-core experiment."""
    normalized_type = component_type.strip().lower().replace("-", "_")
    component_types = {
        "environment": ("environments", "Environment"),
        "rule": ("rules", "Rule"),
        "scenario": ("scenarios", "Scenario"),
    }
    if normalized_type not in component_types:
        supported = ", ".join(component_types)
        raise ValueError(
            f"Unsupported robotics component '{component_type}'. "
            f"Supported components: {supported}"
        )

    project_root = find_project_root(Path(root_dir).resolve())
    if project_root is None:
        raise FileNotFoundError(
            "Could not find a dl-core experiment repository. Run this command "
            "inside a repository created by dl-init or pass --root-dir."
        )

    normalized_name = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip()),
    ).strip("_").lower()
    if not normalized_name:
        raise ValueError("Component name must contain an alphanumeric character")
    if normalized_name[0].isdigit():
        normalized_name = f"robotics_{normalized_name}"
    class_name = "".join(
        part[:1].upper() + part[1:]
        for part in normalized_name.split("_")
        if part
    )

    package_name, class_suffix = component_types[normalized_type]
    package_dir = find_local_component_root_dir(project_root) / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    component_path = package_dir / f"{normalized_name}.py"
    if component_path.exists() and not force:
        raise FileExistsError(f"Component already exists: {component_path}")

    if normalized_type == "environment":
        symbol_name = f"{class_name}{class_suffix}"
        source = f'''"""Local robotics environment."""

from dl_core.core import register_environment
from dl_robotics import GridMAPFEnvironment


@register_environment("{normalized_name}")
class {symbol_name}(GridMAPFEnvironment):
    """Customize GridMAPFEnvironment for this experiment."""
'''
    elif normalized_type == "rule":
        symbol_name = f"{class_name}{class_suffix}"
        source = f'''"""Local actor interaction rule."""

from dl_robotics import ExclusiveCellRule, register_interaction_rule


@register_interaction_rule("{normalized_name}")
class {symbol_name}(ExclusiveCellRule):
    """Customize exclusive-cell conflict handling for this experiment."""
'''
    else:
        symbol_name = f"make_{normalized_name}_scenario"
        source = f'''"""Local robotics scenario."""

from dl_robotics import GridScenario


def {symbol_name}() -> GridScenario:
    """Create the {normalized_name.replace("_", " ")} scenario."""
    return GridScenario(
        width=8,
        height=8,
        starts=((0, 0),),
        goals=((7, 7),),
        walls=(),
        max_steps=100,
        name="{normalized_name}",
    )
'''

    component_path.write_text(source, encoding="utf-8")
    init_path = package_dir / "__init__.py"
    init_content = (
        init_path.read_text(encoding="utf-8")
        if init_path.exists()
        else f'"""Local {package_name.replace("_", " ")}."""\n'
    )
    import_line = f"from .{normalized_name} import {symbol_name}"
    if import_line not in init_content:
        init_content = f"{init_content.rstrip()}\n\n{import_line}\n"
    init_path.write_text(init_content, encoding="utf-8")
    return component_path


def main(argv: Sequence[str] | None = None) -> None:
    """Generate a robotics-specific experiment component."""
    parser = argparse.ArgumentParser(
        description="Add a robotics component to a dl-core experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_parser = subparsers.add_parser(
        "add",
        help="Add a robotics environment, interaction rule, or scenario.",
    )
    add_parser.add_argument(
        "component_type",
        choices=("environment", "rule", "scenario"),
    )
    add_parser.add_argument("name")
    add_parser.add_argument("--root-dir", default=".")
    add_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing component with the same name.",
    )
    args = parser.parse_args(argv)

    component_path = create_robotics_component(
        args.component_type,
        args.name,
        root_dir=args.root_dir,
        force=args.force,
    )
    print(f"Created {component_path}")
