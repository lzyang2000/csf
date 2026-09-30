# SPDX-License-Identifier: Apache-2.0
"""Load MotionHiFlow's VAE and flow DiT from their checkpoint directories.

MotionHiFlow's `load_model` instantiates Hydra configs whose targets live in
its top-level package `src`, so `third_party/MotionHiFlow` must be on
`sys.path` and its configs' relative paths must resolve against the
MotionHiFlow root.  `src` is a generic name, so `_mhf_import_isolation` keeps
MotionHiFlow's `src` from shadowing, or being shadowed by, another cached one.

Config keys (`configs/backbones/motionhiflow.yaml`): `vae_dir` and `dit_dir`
(each holds `config.yaml` and `checkpoints/net_best_fid.tar`) and `dataset`,
which selects the HumanML3D normalisation statistics that
`scripts/fetch_motionhiflow_ckpts.sh` places under
`third_party/MotionHiFlow/deps/evaluators/<dataset>/Comp_v6_KLD005/meta/`.
Relative paths are resolved against the repository root.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np

from csf.paths import REPO_ROOT, THIRD_PARTY_DIR

MHF_ROOT = THIRD_PARTY_DIR / "MotionHiFlow"

_EVAL_META = "Comp_v6_KLD005"
_EVAL_MEAN = {ds: MHF_ROOT / "deps" / "evaluators" / ds / _EVAL_META / "meta" / "mean.npy"
              for ds in ("t2m", "kit")}
_EVAL_STD = {ds: MHF_ROOT / "deps" / "evaluators" / ds / _EVAL_META / "meta" / "std.npy"
             for ds in ("t2m", "kit")}

_MHF_COLLIDING_ROOTS = ("src",)


@contextmanager
def _mhf_root_cwd():
    """Run inside the MotionHiFlow root (its Hydra configs use relative paths)."""
    original = os.getcwd()
    try:
        os.chdir(str(MHF_ROOT))
        yield
    finally:
        os.chdir(original)


@contextmanager
def _mhf_import_isolation():
    """Import MotionHiFlow's `src` package without disturbing another cached `src`.

    Cached `src` modules are removed for the duration and restored afterwards;
    modules MotionHiFlow imported meanwhile are dropped.  Objects already bound
    from MotionHiFlow keep working after the exit.
    """
    saved: dict[str, Any] = {}
    for name in list(sys.modules):
        if name.split(".")[0] in _MHF_COLLIDING_ROOTS:
            saved[name] = sys.modules.pop(name)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in _MHF_COLLIDING_ROOTS and name not in saved:
                sys.modules.pop(name)
        sys.modules.update(saved)


def _ensure_mhf_on_path() -> None:
    """Put the MotionHiFlow root on `sys.path` (idempotent)."""
    root = str(MHF_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _check_model_dir(dir_path: Path, label: str, ckpt_name: str = "net_best_fid") -> None:
    """Raise FileNotFoundError unless `dir_path` holds config.yaml and its checkpoint."""
    for required in (dir_path, dir_path / "config.yaml",
                     dir_path / "checkpoints" / f"{ckpt_name}.tar"):
        if not required.exists():
            raise FileNotFoundError(
                f"MotionHiFlow {label} file not found: {required}\n"
                "Run scripts/fetch_motionhiflow_ckpts.sh to download the checkpoints."
            )


def load_mhf_models(cfg_dict: dict[str, Any], device: str
                    ) -> tuple[Any, Any, Callable[[np.ndarray], np.ndarray]]:
    """Build MotionHiFlow's VAE and flow DiT, and the feature denormaliser.

    Args:
        cfg_dict: The MotionHiFlow backbone config (see the module docstring).
        device: Torch device string, e.g. ``"cuda:0"``.

    Returns:
        ``(vae, flow, inv_transform)``: both models in eval mode on `device`,
        and ``inv_transform(x) = x * std + mean`` for 263-d numpy features.

    Raises:
        FileNotFoundError: A model directory, its config or checkpoint, or the
            normalisation statistics are missing.
        ValueError: Unknown `dataset`.
        RuntimeError: MotionHiFlow could not be imported or a model not loaded.
    """
    vae_dir = _resolve(cfg_dict["vae_dir"])
    dit_dir = _resolve(cfg_dict["dit_dir"])
    dataset = cfg_dict.get("dataset", "t2m")
    _check_model_dir(vae_dir, label="VAE")
    _check_model_dir(dit_dir, label="DiT")

    with _mhf_import_isolation():
        _ensure_mhf_on_path()
        try:
            from src.utils import load_model as mhf_load_model  # type: ignore[import]  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"Failed to import MotionHiFlow ({exc}); install the backbone extras: "
                "pip install -e '.[mhf]'"
            ) from exc
        loaded = {}
        for label, path in (("flow DiT", dit_dir), ("VAE", vae_dir)):
            try:
                with _mhf_root_cwd():
                    loaded[label] = mhf_load_model(str(path))
            except Exception as exc:
                raise RuntimeError(f"Failed to load MotionHiFlow {label} from {path}: {exc}") from exc

    flow = loaded["flow DiT"].to(device).eval()
    vae = loaded["VAE"].to(device).eval()

    if dataset not in _EVAL_MEAN:
        raise ValueError(f"Unknown dataset {dataset!r}; supported: {list(_EVAL_MEAN)}")
    for required in (_EVAL_MEAN[dataset], _EVAL_STD[dataset]):
        if not required.exists():
            raise FileNotFoundError(
                f"MotionHiFlow normalisation statistics not found: {required}\n"
                "Run scripts/fetch_motionhiflow_ckpts.sh to download them."
            )
    mean = np.load(str(_EVAL_MEAN[dataset]))
    std = np.load(str(_EVAL_STD[dataset]))

    def inv_transform(data: np.ndarray) -> np.ndarray:
        return data * std + mean

    return vae, flow, inv_transform
