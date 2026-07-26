"""Registration and construction of actor interaction rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dl_core.core import ComponentRegistry

from .world import ExclusiveCellRule, InteractionRule

INTERACTION_RULE_REGISTRY = ComponentRegistry("Interaction rule")
INTERACTION_RULE_REGISTRY.register_class("exclusive_cell", ExclusiveCellRule)
INTERACTION_RULE_REGISTRY.register_class("exclusive", ExclusiveCellRule)


def register_interaction_rule(names: str | list[str]):
    """Register an interaction rule class under one or more names."""
    def decorator(rule_class: type[InteractionRule]) -> type[InteractionRule]:
        if not isinstance(rule_class, type) or not issubclass(
            rule_class,
            InteractionRule,
        ):
            raise TypeError(
                "Registered interaction rules must inherit InteractionRule"
            )
        INTERACTION_RULE_REGISTRY.register(names)(rule_class)
        return rule_class

    return decorator


def make_interaction_rule(
    config: str | Mapping[str, Any] | InteractionRule | None,
) -> InteractionRule:
    """Create an interaction rule from a name, mapping, or existing instance."""
    if config is None:
        return ExclusiveCellRule()
    if isinstance(config, InteractionRule):
        return config
    if isinstance(config, str):
        rule_name = config
        rule_config: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        rule_name = config.get("name")
        if not isinstance(rule_name, str) or not rule_name:
            raise ValueError(
                "interaction_rule.name must be a non-empty string"
            )
        rule_config = {
            key: value for key, value in config.items() if key != "name"
        }
    else:
        raise TypeError(
            "interaction_rule must be a registered name, mapping, or "
            "InteractionRule instance"
        )

    registered_rules = INTERACTION_RULE_REGISTRY.registered_items()
    if rule_name not in registered_rules:
        raise NotImplementedError(
            f"Interaction rule '{rule_name}' not found. "
            f"Available interaction rules: {list(registered_rules)}"
        )
    rule_class = registered_rules[rule_name]
    if not issubclass(rule_class, InteractionRule):
        raise TypeError(
            f"Registered interaction rule '{rule_name}' must inherit "
            "InteractionRule"
        )
    return rule_class.from_config(rule_config)
