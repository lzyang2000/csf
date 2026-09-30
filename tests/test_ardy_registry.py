# SPDX-License-Identifier: Apache-2.0
"""ARDY's registry entry, config and bundle builder (GPU-free, no checkpoint)."""
from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from csf.backbones import registry
from csf.backbones.base import ADAPTER_ATTACH
from csf.paths import BACKBONE_CONFIGS_DIR, REPO_ROOT


def test_ardy_g1_is_the_only_ardy_backbone():
    assert [n for n, s in registry.SPECS.items() if s.family == "ardy"] == ["ARDY-G1-v1"]


def test_ardy_spec_matches_table_1():
    spec = registry.get_spec("ARDY-G1-v1")
    assert spec.filter.gamma == 0.2
    assert not spec.filter.decoder_aware
    assert spec.shared_text_encoder, "ARDY conditions on the shared LLM2Vec"
    assert spec.mesh_mode == "g1_stl"
    assert spec.attach == ADAPTER_ATTACH
    assert spec.config_file == "ardy.yaml"
    assert spec.required_paths == ("third_party/ardy/ardy/__init__.py",)


def test_ardy_config_names_the_g1_checkpoint():
    with open(BACKBONE_CONFIGS_DIR / "ardy.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["model_name"] == "g1"
    assert cfg["num_denoising_steps"] == 10
    assert cfg["cfg_weight"] == 2.0


def test_ardy_is_available_when_the_submodule_is_checked_out():
    spec = registry.get_spec("ARDY-G1-v1")
    present = (REPO_ROOT / spec.required_paths[0]).exists()
    assert spec.available() == present


def test_the_builder_gets_the_yaml_and_the_shared_encoder(monkeypatch):
    import csf.backbones.ardy.model as ardy_model

    seen = {}

    def fake_build(cfg_dict, device, text_encoder=None):
        seen.update(cfg=cfg_dict, device=device, text_encoder=text_encoder)
        return "bundle"

    monkeypatch.setattr(ardy_model, "build_ardy_bundle", fake_build)
    ctx = SimpleNamespace(device="cuda:0", text_encoder="llm2vec")
    assert registry.load_backbone("ARDY-G1-v1", ctx) == "bundle"
    assert seen["cfg"]["model_name"] == "g1"
    assert seen["device"] == "cuda:0" and seen["text_encoder"] == "llm2vec"


def test_building_without_the_shared_encoder_fails_before_loading():
    pytest.importorskip("torch")
    from csf.backbones.ardy.model import build_ardy_bundle

    with pytest.raises(ValueError, match="text_encoder"):
        build_ardy_bundle({"model_name": "g1"}, "cpu", text_encoder=None)


def test_the_ardy_package_import_is_light():
    code = ("import sys, csf.backbones.ardy; "
            "assert 'torch' not in sys.modules and 'ardy' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)
