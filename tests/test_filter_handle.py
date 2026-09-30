# SPDX-License-Identifier: Apache-2.0
"""The filter handles: the per-segment gate decision and the adapter installation."""
import zlib

import pytest
import torch

from csf.filter.config import FilterConfig, Rule, SafeReferencePolicy
from csf.filter.handle import AdapterFilterHandle, FilterHandle, attach_adapter_filter
from csf.filter.qp import filter_correction

STAND = "a person standing at ease"
WALK = "a person walks slowly"
CLAP = "a person claps in a neutral way"

# Axes: strike, harm-animal, person, animal, object, walk, clap, practice.
TABLE = {
    "strike another person": [1, 0, 0, 0, 0, 0, 0, 0],
    "harm an animal": [0, 1, 0, 0, 0, 0, 0, 0],
    "another person": [0, 0, 1, 0, 0, 0, 0, 0],
    "an animal": [0, 0, 0, 1, 0, 0, 0, 0],
    "an inanimate object": [0, 0, 0, 0, 1, 0, 0, 0],
    WALK: [0, 0, 0, 0, 0, 1, 0, 0],
    CLAP: [0, 0, 0, 0, 0, 0, 1, 0],
    STAND: [0, 0, 0, 0, 0, 0.1, 0.1, 0],
    "a person punches the air": [0.7, 0, 0, 0, 0, 0, 0, 0.7],
    "a man": [0, 0, 0.95, 0, 0.1, 0, 0, 0],
    "a wall": [0, 0, 0.1, 0, 0.95, 0, 0, 0],
    "punch the man": [0.95, 0, 0.2, 0, 0, 0, 0, 0.1],
    "a person walks forward": [0, 0, 0, 0, 0, 1, 0, 0.05],
}

RULES = [Rule(name="strike a person", unsafe="strike another person", protects="another person"),
         Rule(name="harm an animal", unsafe="harm an animal", protects="an animal")]
POLICY = SafeReferencePolicy(library=(WALK, CLAP), stand=STAND, min_similarity=0.6)
FRAMES, DIM, STEPS = 6, 5, 4


def _cfg(**overrides) -> FilterConfig:
    values = dict(rules=RULES, neutral_entities=["an inanimate object"],
                  benign_contrasts=["a person punches the air"])
    values.update(overrides)
    return FilterConfig(**values)


@pytest.fixture()
def encoder(make_encoder):
    return make_encoder(TABLE)


# ------------------------------------------------------------ FilterHandle
def test_without_an_encoder_every_rule_is_enforced_against_the_stand():
    handle = FilterHandle(_cfg(), POLICY)
    decision = handle.decide("punch the man")
    assert decision.mask is None and decision.reasons is None
    assert decision.safe_reference == STAND
    assert not decision.inactive
    assert handle.decisions == [decision]


def test_perception_decides_the_active_rules(encoder):
    handle = FilterHandle(_cfg(), POLICY, encoder.batch)
    handle.set_entities(["a man"])
    decision = handle.decide("a person walks forward")
    assert decision.mask == [True, False]
    assert decision.reasons == [["perception"], []]
    assert decision.safe_reference == WALK
    assert decision.entities == ("a man",)


def test_benign_prompt_in_an_empty_scene_is_inactive(encoder):
    handle = FilterHandle(_cfg(), POLICY, encoder.batch)
    handle.set_entities(["a wall"])
    assert handle.decide("a person walks forward").inactive


def test_unsafe_prompt_fires_by_prompt_and_tracks_the_stand(encoder):
    decision = FilterHandle(_cfg(), POLICY, encoder.batch).decide("punch the man")
    assert decision.mask == [True, True]
    assert decision.reasons == [["prompt"], ["prompt"]]
    assert decision.safe_reference == STAND, "no library action is close enough"


def test_library_actions_serve_as_benign_contrasts(encoder):
    handle = FilterHandle(_cfg(benign_contrasts=[]), POLICY, encoder.batch)
    assert handle.decide("punch the man").mask == [True, True]
    assert handle.decide("a person walks forward").inactive


def test_pinned_safe_reference_wins(encoder):
    handle = FilterHandle(_cfg(), POLICY, encoder.batch)
    handle.pin_safe_references({"punch the man": CLAP})
    assert handle.decide("punch the man").safe_reference == CLAP
    assert handle.decide("Punch the man.").safe_reference == CLAP, "sanitised text is pinned too"
    assert handle.decide("punch the man").mask == [True, True], "pinning does not change the gate"
    handle.pin_safe_references(None)
    assert handle.decide("punch the man").safe_reference == STAND


def test_set_entities_drops_blank_labels():
    handle = FilterHandle(_cfg(), POLICY)
    handle.set_entities([" a man ", "", "   "])
    assert handle.entities == ["a man"]
    handle.set_entities(None)
    assert handle.entities == []


def test_embeddings_are_cached_by_sanitised_text(encoder):
    handle = FilterHandle(_cfg(), POLICY, encoder.batch)
    first = handle.encode("punch the man")
    second = handle.encode("  Punch the man.")
    assert (first == second).all()
    assert encoder.batch_calls == [["Punch the man."]]


# ------------------------------------------------------ AdapterFilterHandle
class FakeAdapterModel:
    """Implements the adapter protocol of `csf.backbones.base` and records every call."""

    root_dims = 2

    def __init__(self):
        self.calls = []
        self.feat_requests = []

    def reference_feat(self, text, num_frames, num_steps):
        self.feat_requests.append((text, num_frames, num_steps))
        g = torch.Generator().manual_seed(zlib.crc32(text.encode()))
        return torch.randn(num_frames, DIM, generator=g)

    def set_filter_context(self, unsafe, safe, cfg, active_mask=None):
        self.calls.append(("set", unsafe, safe, cfg, active_mask))

    def clear_filter_context(self):
        self.calls.append(("clear",))

    def set_filter_segments(self, per_prompt):
        self.calls.append(("segments", per_prompt))


@pytest.fixture()
def adapter(encoder):
    model = FakeAdapterModel()
    handle = attach_adapter_filter(model, _cfg(), policy=POLICY, encoder=encoder.batch)
    handle.set_entities(["a man"])
    return model, handle


def test_adapter_handle_takes_root_dims_from_the_model():
    cfg = _cfg()
    handle = AdapterFilterHandle(FakeAdapterModel(), cfg, POLICY)
    assert handle.cfg.root_dims == 2
    assert cfg.root_dims == 0, "the caller's config is not modified"


def test_active_segment_installs_its_references(adapter):
    model, handle = adapter
    with handle.filtering(["punch the man"], [FRAMES], STEPS):
        (kind, unsafe, safe, cfg, mask), segments = model.calls
        assert kind == "set" and segments == ("segments", None)
    assert unsafe.shape == (len(RULES), 1, FRAMES, DIM)
    assert safe.shape == (1, FRAMES, DIM)
    for k, rule in enumerate(RULES):
        assert torch.equal(unsafe[k, 0], model.reference_feat(rule.unsafe, FRAMES, STEPS))
    assert torch.equal(safe[0], model.reference_feat(STAND, FRAMES, STEPS))
    assert cfg is handle.cfg and mask == [True, True]
    assert model.calls[-2:] == [("clear",), ("segments", None)]


def test_inactive_segment_generates_unfiltered(adapter):
    model, handle = adapter
    handle.set_entities([])
    with handle.filtering(["a person walks forward"], [FRAMES], STEPS):
        assert model.calls == [("clear",), ("segments", None)]
    assert handle.decisions[0].inactive
    assert model.feat_requests == [], "no reference is generated for an inactive segment"


def test_multi_prompt_schedule_installs_one_context_per_segment(adapter):
    model, handle = adapter
    handle.set_entities([])
    prompts = ["a person walks forward", "punch the man"]
    with handle.filtering(prompts, [FRAMES, FRAMES], STEPS):
        segments = model.calls[1][1]
    assert model.calls[0] == ("clear",), "the first segment is inactive"
    assert set(segments) == set(prompts)
    assert segments["a person walks forward"] == (None, None, None, None)
    unsafe, safe, cfg, mask = segments["punch the man"]
    assert unsafe.shape == (len(RULES), 1, FRAMES, DIM) and mask == [True, True]
    assert [d.prompt for d in handle.decisions] == prompts


def test_reference_features_are_generated_once(adapter):
    model, handle = adapter
    with handle.filtering(["punch the man", "a person walks forward"], [FRAMES, FRAMES], STEPS):
        pass
    assert len(model.feat_requests) == len(set(model.feat_requests))
    texts = {text for text, _, _ in model.feat_requests}
    assert texts == {rule.unsafe for rule in RULES} | {STAND, WALK}


def test_filter_is_removed_even_if_generation_fails(adapter):
    model, handle = adapter
    with pytest.raises(RuntimeError):
        with handle.filtering(["punch the man"], [FRAMES], STEPS):
            raise RuntimeError("generation failed")
    assert model.calls[-2:] == [("clear",), ("segments", None)]


def test_installed_context_drives_the_qp(adapter):
    """The references a handle installs are exactly what `filter_correction` consumes."""
    model, handle = adapter
    with handle.filtering(["punch the man"], [FRAMES], STEPS):
        _, unsafe, safe, cfg, mask = model.calls[0]
    x_hat = unsafe[0].clone()
    delta, info = filter_correction(x_hat, safe, cfg, x_unsafe=unsafe, active_mask=mask)
    assert info["active"]
    assert torch.equal(delta[..., :2], torch.zeros_like(delta[..., :2]))
    assert delta[..., 2:].abs().max() > 0
