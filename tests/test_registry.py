# SPDX-License-Identifier: Apache-2.0
"""The backbone registry and descriptors (Table I), without loading any model."""
import dataclasses
import subprocess
import sys
from types import SimpleNamespace

import pytest

from csf.backbones import registry
from csf.backbones.base import (
    ADAPTER_ATTACH,
    BackboneSpec,
    FilterDefaults,
    PerSegmentFilterMixin,
    shared_text_encoder,
)
from csf.filter.config import FilterConfig
from csf.paths import BACKBONE_CONFIGS_DIR

NAMES = ["Kimodo-G1-RP-v1", "ECHO-G1-v1", "MotionHiFlow-SMPL-v1", "ARDY-G1-v1"]


def test_exactly_the_four_paper_backbones_in_order():
    assert registry.all_names() == NAMES
    assert list(registry.SPECS) == NAMES


def test_families():
    assert [registry.get_spec(n).family for n in NAMES] == ["kimodo", "echo", "motionhiflow", "ardy"]


@pytest.mark.parametrize(("name", "gamma"), [
    ("Kimodo-G1-RP-v1", 0.0),
    ("ECHO-G1-v1", 0.2),
    ("MotionHiFlow-SMPL-v1", 0.5),
    ("ARDY-G1-v1", 0.2),
])
def test_table_one_filter_settings(name, gamma):
    defaults = registry.get_spec(name).filter
    assert defaults.gamma == gamma
    assert defaults.decoder_aware is False


def test_filter_defaults_apply_cleanly_to_the_filter_config():
    base = FilterConfig()
    for name in NAMES:
        defaults = registry.get_spec(name).filter
        cfg = base.with_backbone(**defaults.as_overrides())
        assert cfg.gamma == defaults.gamma


def test_text_encoder_placement():
    shared = {n: registry.get_spec(n).shared_text_encoder for n in NAMES}
    assert shared == {"Kimodo-G1-RP-v1": True, "ECHO-G1-v1": False,
                      "MotionHiFlow-SMPL-v1": False, "ARDY-G1-v1": True}


def test_mesh_modes():
    assert registry.get_spec("MotionHiFlow-SMPL-v1").mesh_mode == "smplx_skin"
    assert all(registry.get_spec(n).mesh_mode == "g1_stl" for n in NAMES if "SMPL" not in n)


def test_filter_attach_points():
    kimodo = registry.get_spec("Kimodo-G1-RP-v1")
    assert kimodo.attach == ("csf.backbones.kimodo.filtered_model", "attach_kimodo_filter")
    assert all(registry.get_spec(n).attach == ADAPTER_ATTACH for n in NAMES[1:])
    assert callable(kimodo.attach_filter)
    assert callable(registry.get_spec("ECHO-G1-v1").attach_filter)


def test_config_files():
    assert registry.get_spec("Kimodo-G1-RP-v1").config_path() is None
    for name in NAMES[1:]:
        spec = registry.get_spec(name)
        assert spec.config_path() == BACKBONE_CONFIGS_DIR / spec.config_file
        assert spec.config_file.endswith(".yaml")


def test_builders_are_lazy_callables():
    assert all(callable(registry.get_spec(n).build) for n in NAMES)


def test_available_names_are_an_ordered_subset():
    available = registry.available_names()
    assert available == [n for n in NAMES if n in available]


def test_unknown_name_lists_the_registered_ones():
    with pytest.raises(KeyError, match="ECHO-G1-v1"):
        registry.get_spec("nope")


def test_specs_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        registry.get_spec("ECHO-G1-v1").filter = FilterDefaults()


def test_importing_the_registry_does_not_import_torch():
    code = ("import sys, csf.backbones.registry as r; "
            "assert r.all_names(); "
            "assert 'torch' not in sys.modules, 'the registry must stay import-light'")
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------ availability
def _spec(config_file: str = "", **fields) -> BackboneSpec:
    return BackboneSpec(name="Test-v1", family="test", config_file=config_file,
                        build=lambda spec, ctx: None, **fields)


def test_spec_without_config_or_checkpoints_is_available():
    assert _spec().available()


def test_spec_with_a_missing_config_is_unavailable(tmp_path):
    assert not _spec(config_file=str(tmp_path / "missing.yaml")).available()


def test_spec_needs_its_checkpoint_directories(tmp_path):
    ckpt = tmp_path / "ckpt"
    config = tmp_path / "model.yaml"
    config.write_text(f"ckpt_dir: {ckpt}\n")
    spec = _spec(config_file=str(config), ckpt_config_keys=("ckpt_dir",))
    assert not spec.available()
    ckpt.mkdir()
    assert spec.available()
    assert not _spec(config_file=str(config), ckpt_config_keys=("other_dir",)).available()


def test_spec_needs_its_required_paths():
    assert not _spec(required_paths=("does/not/exist.py",)).available()
    assert _spec(required_paths=("csf/__init__.py",)).available()


def test_shared_text_encoder_reads_the_load_context():
    encoder = object()
    assert shared_text_encoder(SimpleNamespace(text_encoder=encoder)) is encoder
    assert shared_text_encoder(SimpleNamespace()) is None


# ------------------------------------------------------ per-segment mixin
class _Model(PerSegmentFilterMixin):
    def __init__(self):
        self.calls = []

    def set_filter_context(self, unsafe, safe, cfg, active_mask=None):
        self.calls.append(("set", unsafe, safe, cfg, active_mask))

    def clear_filter_context(self):
        self.calls.append(("clear",))


def test_each_segment_installs_its_own_context():
    model = _Model()
    cfg = FilterConfig()
    model.set_filter_segments({"punch": ("U", "S", cfg, [True]),
                               "walk": (None, None, None, None)})
    model._activate_filter_for("punch")
    model._activate_filter_for("walk")
    model._activate_filter_for("unknown")
    assert model.calls == [("set", "U", "S", cfg, [True]), ("clear",), ("clear",)]


def test_without_segments_the_installed_context_is_kept():
    model = _Model()
    model._activate_filter_for("punch")
    model.set_filter_segments(None)
    model._activate_filter_for("punch")
    assert model.calls == []
