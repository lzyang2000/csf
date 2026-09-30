# SPDX-License-Identifier: Apache-2.0
"""The declared specification (`configs/rules.yaml`, `configs/safe_references.yaml`)."""
import dataclasses

import pytest

from csf.filter.config import (
    FilterConfig,
    Rule,
    SafeReferencePolicy,
    load_filter_config,
    load_safe_reference_policy,
)


@pytest.fixture(scope="module")
def cfg() -> FilterConfig:
    return load_filter_config()


@pytest.fixture(scope="module")
def policy_and_threshold():
    return load_safe_reference_policy()


# --------------------------------------------------------------- rules.yaml
def test_rules_declare_six_distinct_rules(cfg):
    assert len(cfg.rules) == 6
    assert all(isinstance(rule, Rule) for rule in cfg.rules)
    assert len({rule.name for rule in cfg.rules}) == 6
    assert all(rule.name and rule.unsafe and rule.protects for rule in cfg.rules)


def test_rules_protect_people_children_and_animals(cfg):
    assert set(cfg.protected_entities) == {"another person", "a child", "an animal"}


def test_rule_properties_follow_rule_order(cfg):
    assert cfg.unsafe_texts == [rule.unsafe for rule in cfg.rules]
    assert cfg.protected_entities == [rule.protects for rule in cfg.rules]


def test_gate_homes_are_declared(cfg):
    assert len(cfg.neutral_entities) >= 1
    assert len(cfg.benign_contrasts) >= len(cfg.rules), "at least one benign contrast per rule"
    assert not set(cfg.neutral_entities) & set(cfg.protected_entities)
    assert not set(cfg.benign_contrasts) & set(cfg.unsafe_texts)


def test_qp_constants(cfg):
    assert (cfg.rho, cfg.ridge, cfg.max_iters) == (1.0, 1e-2, 20)


def test_per_backbone_settings_are_not_read_from_yaml(cfg):
    """gamma and root_dims come from the backbone registry."""
    assert (cfg.gamma, cfg.root_dims) == (0.0, 0)
    assert cfg.pin_dims == [] and cfg.decoder_aware is False


def test_a_spec_without_rules_is_rejected(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("gate: {}\n")
    with pytest.raises(ValueError, match="no rules"):
        load_filter_config(path)


def test_optional_sections_fall_back_to_defaults(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("rules:\n  - {name: r, unsafe: u, protects: p}\n")
    cfg = load_filter_config(path)
    assert cfg.rules == [Rule(name="r", unsafe="u", protects="p")]
    assert cfg.neutral_entities == [] and cfg.benign_contrasts == []
    assert (cfg.rho, cfg.ridge, cfg.max_iters) == (1.0, 1e-2, 20)


# ---------------------------------------------------- safe_references.yaml
def test_safe_reference_policy(policy_and_threshold):
    policy, threshold = policy_and_threshold
    assert isinstance(policy, SafeReferencePolicy)
    assert policy.stand == "a person standing at ease"
    assert policy.min_similarity == pytest.approx(0.63)
    assert policy.locomotion_min_similarity == pytest.approx(0.65)
    assert threshold == pytest.approx(0.3)


def test_library_is_a_set_of_neutral_actions(policy_and_threshold):
    policy, _ = policy_and_threshold
    assert isinstance(policy.library, tuple) and len(policy.library) >= 4
    assert len(set(policy.library)) == len(policy.library)
    assert all(text == text.strip() and text for text in policy.library)
    assert policy.stand not in policy.library


def test_locomotion_is_a_subset_of_the_library(policy_and_threshold):
    policy, _ = policy_and_threshold
    assert policy.locomotion and set(policy.locomotion) <= set(policy.library)
    assert "a person walks slowly" in policy.locomotion


def test_policy_is_immutable(policy_and_threshold):
    policy, _ = policy_and_threshold
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.stand = "something else"


# ------------------------------------------------------------ with_backbone
def test_with_backbone_applies_overrides_to_a_copy(cfg):
    tuned = cfg.with_backbone(gamma=0.5, root_dims=5)
    assert (tuned.gamma, tuned.root_dims) == (0.5, 5)
    assert (cfg.gamma, cfg.root_dims) == (0.0, 0)
    assert tuned is not cfg


def test_with_backbone_shares_the_declared_lists(cfg):
    tuned = cfg.with_backbone(gamma=0.2)
    assert tuned.rules is cfg.rules
    assert tuned.benign_contrasts is cfg.benign_contrasts


def test_with_backbone_rejects_unknown_fields(cfg):
    with pytest.raises(AttributeError, match="gama"):
        cfg.with_backbone(gama=0.2)
