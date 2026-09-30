# SPDX-License-Identifier: Apache-2.0
"""Load ECHO's `DiffusePipeline` from a checkpoint directory.

ECHO's generator directory is not an installable package: it exposes
generically named top-level modules (`models`, `utils`, `options`, ...) that
collide with packages already importable in the environment.
`_echo_import_isolation` evicts those names from `sys.modules` while ECHO is
imported and restores them afterwards, so neither side shadows the other.
`DiffusePipeline.__init__` also opens `config/diffuser_params.yaml` relative
to the working directory, so construction runs inside the generator root.

Config keys (`configs/backbones/echo.yaml`): `ckpt_dir` (holds `opt.txt` and
`model/latest.tar`), `data_root` (holds `Mean_38d.npy` / `Std_38d.npy`),
`diffuser_name` and `num_denoising_steps`.  Relative paths are resolved
against the repository root, never the working directory.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from csf.paths import REPO_ROOT, THIRD_PARTY_DIR

from .opt import parse_opt_txt

ECHO_GENERATOR = THIRD_PARTY_DIR / "ECHO_CODE" / "generator"

#: Top-level module names defined by ECHO's generator directory.
_ECHO_COLLIDING_ROOTS = (
    "models", "utils", "options", "datasets", "trainers", "eval", "motion_loader", "tools",
)


def resolve_repo_path(path: str | Path) -> Path:
    """`path` as given if absolute, else relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


@contextmanager
def _echo_generator_cwd():
    """Run inside ECHO's generator root (it opens config files by relative path)."""
    original = os.getcwd()
    try:
        os.chdir(str(ECHO_GENERATOR))
        yield
    finally:
        os.chdir(original)


@contextmanager
def _echo_import_isolation():
    """Import ECHO's generically named modules without shadowing, or being shadowed by, ours.

    Cached modules under ECHO's top-level names are removed for the duration and
    restored afterwards; modules ECHO imported meanwhile are dropped.  Objects
    already bound from ECHO (classes, functions) keep working after the exit.
    """
    saved: dict[str, Any] = {}
    for name in list(sys.modules):
        if name.split(".")[0] in _ECHO_COLLIDING_ROOTS:
            saved[name] = sys.modules.pop(name)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in _ECHO_COLLIDING_ROOTS and name not in saved:
                sys.modules.pop(name)
        sys.modules.update(saved)


def _ensure_echo_on_path() -> None:
    """Put ECHO's generator root on `sys.path` (idempotent)."""
    root = str(ECHO_GENERATOR)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_echo_pipeline(cfg_dict: dict[str, Any], device: str
                       ) -> tuple[Any, np.ndarray, np.ndarray, SimpleNamespace]:
    """Build ECHO's pipeline and read its normalisation statistics.

    Args:
        cfg_dict: The ECHO backbone config (see the module docstring).
        device: Torch device string, e.g. ``"cuda:0"``.

    Returns:
        ``(pipeline, mean, std, opt)``: the `DiffusePipeline` ready for
        inference, the 38-d mean and std, and the parsed `opt.txt`.

    Raises:
        FileNotFoundError: `opt.txt` or `model/latest.tar` is missing.
        RuntimeError: The pipeline could not be constructed.
    """
    import torch  # noqa: PLC0415

    ckpt_dir = resolve_repo_path(cfg_dict["ckpt_dir"])
    data_root = resolve_repo_path(cfg_dict["data_root"])
    opt_path = ckpt_dir / "opt.txt"
    model_dir = ckpt_dir / "model"
    ckpt_path = model_dir / "latest.tar"
    for required in (opt_path, ckpt_path):
        if not required.exists():
            raise FileNotFoundError(
                f"ECHO checkpoint file not found: {required}\n"
                "Run scripts/fetch_echo_ckpts.sh to download the checkpoints."
            )

    opt = parse_opt_txt(str(opt_path))
    opt.model_dir = str(model_dir)
    opt.data_root = str(data_root)

    with _echo_import_isolation():
        _ensure_echo_on_path()
        import models as echo_models  # type: ignore[import]  # noqa: PLC0415
        from models.gaussian_diffusion import DiffusePipeline  # type: ignore[import]  # noqa: PLC0415
        from utils.model_load import load_model_weights  # type: ignore[import]  # noqa: PLC0415

        model = echo_models.build_models(opt)
        load_model_weights(model, str(ckpt_path), use_ema=not getattr(opt, "no_ema", False),
                           device=device)
        try:
            with _echo_generator_cwd():
                pipeline = DiffusePipeline(
                    opt=opt,
                    model=model,
                    diffuser_name=cfg_dict["diffuser_name"],
                    num_inference_steps=cfg_dict["num_denoising_steps"],
                    device=device,
                    torch_dtype=torch.float32,
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to construct ECHO's DiffusePipeline: {exc}") from exc

    mean = np.load(str(data_root / "Mean_38d.npy"))
    std = np.load(str(data_root / "Std_38d.npy"))
    return pipeline, mean, std, opt
