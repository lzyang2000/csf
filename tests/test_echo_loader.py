# SPDX-License-Identifier: Apache-2.0
"""ECHO's opt.txt parser and the error paths of the pipeline loader (no checkpoint needed)."""
from __future__ import annotations

import sys
import textwrap
import types

import pytest

from csf.backbones.echo.opt import parse_opt_txt


def test_parse_opt_txt_coerces_types(tmp_path):
    path = tmp_path / "opt.txt"
    path.write_text(textwrap.dedent("""\
        ------------ Options -------------
        dim_pose: 38
        fps: 50
        diffusion_steps: 1000
        no_ema: False
        dim_mults: [2, 2, 2, 2]
        lr: 0.0002
        beta_schedule: linear
        -------------- End ----------------
    """))
    opt = parse_opt_txt(str(path))
    assert opt.dim_pose == 38 and isinstance(opt.dim_pose, int)
    assert opt.fps == 50
    assert opt.no_ema is False
    assert list(opt.dim_mults) == [2, 2, 2, 2]
    assert opt.lr == pytest.approx(2e-4)
    assert opt.beta_schedule == "linear"


def test_parse_opt_txt_fills_robotv2_defaults(tmp_path):
    path = tmp_path / "opt.txt"
    path.write_text("dim_pose: 38\n")
    opt = parse_opt_txt(str(path))
    assert (opt.joints_num, opt.max_motion_length, opt.fps) == (30, 490, 50)


def test_load_echo_pipeline_missing_ckpt_raises(tmp_path):
    from csf.backbones.echo.loader import load_echo_pipeline

    cfg = {"variant": "lite", "ckpt_dir": str(tmp_path / "nope"), "data_root": str(tmp_path),
           "mujoco_xml": "x.xml", "num_denoising_steps": 10, "diffuser_name": "dpmsolver",
           "fps": 50}
    with pytest.raises(FileNotFoundError, match="fetch_echo_ckpts"):
        load_echo_pipeline(cfg, device="cpu")


def test_relative_paths_resolve_against_the_repo_root(tmp_path, monkeypatch):
    from csf.backbones.echo.loader import resolve_repo_path
    from csf.paths import REPO_ROOT

    monkeypatch.chdir(tmp_path)
    assert resolve_repo_path("checkpoints/echo") == REPO_ROOT / "checkpoints/echo"
    assert resolve_repo_path(tmp_path) == tmp_path


def test_echo_import_isolation_evicts_and_restores():
    """A cached module under one of ECHO's generic names is hidden inside, restored after."""
    from csf.backbones.echo.loader import _echo_import_isolation

    outer = types.ModuleType("models")
    sys.modules["models"] = outer
    try:
        with _echo_import_isolation():
            assert sys.modules.get("models") is not outer
            sys.modules["models.echo_only"] = types.ModuleType("models.echo_only")
        assert sys.modules.get("models") is outer
        assert "models.echo_only" not in sys.modules
    finally:
        if sys.modules.get("models") is outer:
            del sys.modules["models"]
