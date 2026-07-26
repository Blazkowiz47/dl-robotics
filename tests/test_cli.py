"""Tests for robotics component generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dl_robotics.cli import create_robotics_component, main


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create the minimum experiment structure required by the generator."""
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = \"demo\"\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize(
    ("component_type", "name", "package_name", "expected_text"),
    [
        (
            "environment",
            "Warehouse",
            "environments",
            '@register_environment("warehouse")',
        ),
        (
            "rule",
            "Priority",
            "rules",
            '@register_interaction_rule("priority")',
        ),
        (
            "scenario",
            "Narrow Crossing",
            "scenarios",
            "def make_narrow_crossing_scenario",
        ),
    ],
)
def test_create_robotics_component(
    project_root: Path,
    component_type: str,
    name: str,
    package_name: str,
    expected_text: str,
) -> None:
    """Each supported component should have a readable local scaffold."""
    component_path = create_robotics_component(
        component_type,
        name,
        root_dir=str(project_root),
    )

    assert component_path.parent == project_root / "src" / package_name
    assert expected_text in component_path.read_text(encoding="utf-8")
    assert component_path.stem in (
        project_root / "src" / package_name / "__init__.py"
    ).read_text(encoding="utf-8")


def test_create_robotics_component_preserves_existing_file(
    project_root: Path,
) -> None:
    """Existing local components should require an explicit force flag."""
    create_robotics_component("rule", "priority", root_dir=str(project_root))

    with pytest.raises(FileExistsError):
        create_robotics_component(
            "rule",
            "priority",
            root_dir=str(project_root),
        )


def test_main_generates_component_directly(
    project_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI entry point should parse and execute without a wrapper layer."""
    main(["add", "scenario", "crossing", "--root-dir", str(project_root)])

    assert (project_root / "src" / "scenarios" / "crossing.py").exists()
    assert "Created" in capsys.readouterr().out
