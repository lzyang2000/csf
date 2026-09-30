# SPDX-License-Identifier: Apache-2.0
"""The context gate (Eq. 5) with a stub text encoder."""
import dataclasses

import pytest

from csf.filter.config import Rule
from csf.filter.gate import GateDecision, rule_signals

# Axes: strike, kick, harm-animal, person, animal, object, walk, practice.
TABLE = {
    # unsafe descriptions
    "strike another person": [1, 0, 0, 0, 0, 0, 0, 0],
    "kick another person": [0, 1, 0, 0, 0, 0, 0, 0],
    "harm an animal": [0, 0, 1, 0, 0, 0, 0, 0],
    # protected and neutral entities
    "another person": [0, 0, 0, 1, 0, 0, 0, 0],
    "an animal": [0, 0, 0, 0, 1, 0, 0, 0],
    "an inanimate object": [0, 0, 0, 0, 0, 1, 0, 0],
    # benign contrasts
    "a person walks": [0, 0, 0, 0, 0, 0, 1, 0],
    "a person punches the air": [0.7, 0, 0, 0, 0, 0, 0, 0.7],
    # perceived entities
    "a man": [0, 0, 0, 0.95, 0, 0.1, 0, 0],
    "a dog": [0, 0, 0, 0, 0.95, 0.1, 0, 0],
    "a wall": [0, 0, 0, 0.1, 0, 0.95, 0, 0],
    # prompts
    "punch the man": [0.95, 0, 0, 0.2, 0, 0, 0, 0.1],
    "punch a sandbag": [0.6, 0, 0, 0, 0, 0.2, 0, 0.75],
    "a person walks forward": [0, 0, 0, 0, 0, 0, 1, 0.05],
}

UNSAFE = ["strike another person", "kick another person", "harm an animal"]
PROTECTED = ["another person", "another person", "an animal"]
NEUTRAL = ["an inanimate object"]
BENIGN = ["a person walks", "a person punches the air"]


@pytest.fixture()
def gate(make_encoder):
    encode = make_encoder(TABLE)

    def run(prompt, entities, benign=BENIGN):
        return rule_signals(encode, prompt, entities, PROTECTED, UNSAFE, NEUTRAL, benign)

    return run


def test_benign_prompt_in_an_empty_scene_fires_nothing(gate):
    assert gate("punch a sandbag", []) == ([False, False, False], [[], [], []])


def test_a_perceived_person_fires_only_the_person_rules(gate):
    assert gate("punch a sandbag", ["a man"]) == (
        [True, True, False], [["perception"], ["perception"], []])


def test_a_perceived_animal_fires_only_the_animal_rule(gate):
    assert gate("a person walks forward", ["a dog"]) == (
        [False, False, True], [[], [], ["perception"]])


def test_an_entity_nearest_a_neutral_home_fires_nothing(gate):
    assert gate("punch a sandbag", ["a wall"])[0] == [False, False, False]


def test_every_perceived_entity_is_checked(gate):
    assert gate("punch a sandbag", ["a wall", "a dog"])[0] == [False, False, True]
    assert gate("punch a sandbag", ["a man", "a dog"])[0] == [True, True, True]


def test_an_unsafe_prompt_fires_every_rule(gate):
    """The prompt signal: the command is nearest an unsafe description, not a benign contrast."""
    assert gate("punch the man", []) == (
        [True, True, True], [["prompt"], ["prompt"], ["prompt"]])


def test_both_signals_are_reported(gate):
    _mask, reasons = gate("punch the man", ["a man"])
    assert reasons == [["perception", "prompt"], ["perception", "prompt"], ["prompt"]]


def test_prompt_signal_needs_benign_contrasts(gate):
    assert gate("punch the man", [], benign=[])[0] == [False, False, False]


@pytest.mark.parametrize("prompt", ["", "   "])
def test_blank_prompt_never_fires_the_prompt_signal(gate, prompt):
    assert gate(prompt, [])[0] == [False, False, False]


def test_mask_holds_plain_bools(gate):
    mask, _ = gate("punch the man", ["a man"])
    assert all(type(on) is bool for on in mask)


def test_no_rules_gives_an_empty_mask(make_encoder):
    assert rule_signals(make_encoder(TABLE), "punch the man", ["a man"], [], [], NEUTRAL, BENIGN) == ([], [])


# ------------------------------------------------------------- GateDecision
RULES = [Rule(name=f"rule {k}", unsafe=u, protects=p) for k, (u, p) in enumerate(zip(UNSAFE, PROTECTED))]


def _decision(mask, reasons=None):
    return GateDecision(mask=mask, reasons=reasons, safe_reference="a person standing at ease",
                        prompt="punch the man", entities=("a man",))


def test_inactive_only_when_the_gate_ran_and_nothing_fired():
    assert _decision([False, False, False]).inactive
    assert not _decision([False, True, False]).inactive
    assert not _decision(None).inactive, "an unavailable gate enforces every rule"


def test_active_rules():
    assert _decision([True, False, True]).active_rules(RULES) == [RULES[0], RULES[2]]
    assert _decision(None).active_rules(RULES) == RULES


def test_as_record():
    record = _decision([True, False, True], [["perception"], [], ["prompt"]]).as_record(RULES)
    assert record["prompt"] == "punch the man"
    assert record["entities"] == ["a man"]
    assert record["safe_reference"] == "a person standing at ease"
    assert record["rules"] == [
        {"name": "rule 0", "protects": "another person", "active": True, "signals": ["perception"]},
        {"name": "rule 1", "protects": "another person", "active": False, "signals": []},
        {"name": "rule 2", "protects": "an animal", "active": True, "signals": ["prompt"]},
    ]


def test_as_record_without_an_encoder_marks_every_rule_active():
    record = _decision(None).as_record(RULES)
    assert [r["active"] for r in record["rules"]] == [True, True, True]
    assert all(r["signals"] == ["unavailable"] for r in record["rules"])


def test_decision_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _decision([True]).mask = None
