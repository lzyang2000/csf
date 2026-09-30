# SPDX-License-Identifier: Apache-2.0
"""Backbone descriptors and the adapter protocol.

A `BackboneSpec` is everything the app needs to know about one generator, as
data: how to build it, how to draw its skeleton, whether it conditions on the
shared LLM2Vec text encoder, its per-backbone filter settings (Table I), and
how to tell that its checkpoints are on disk.

This module imports no torch, viser or model code, so the registry that
instantiates the specs is cheap to import from the app and from tests.  Heavy
imports live inside each spec's `build` callable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from csf.paths import BACKBONE_CONFIGS_DIR, REPO_ROOT

#: The filter attach shared by the adapter families (ECHO, MotionHiFlow, ARDY).
ADAPTER_ATTACH = ("csf.filter.handle", "attach_adapter_filter")


def shared_text_encoder(ctx: object):
    """The shared LLM2Vec encoder a load context holds, or None."""
    return getattr(ctx, "text_encoder", None)


def ensure_text_encoder(ctx: object, device: str | None = None):
    """The shared LLM2Vec text encoder, loaded onto `device` (default `ctx.device`) if absent.

    Kimodo and ARDY condition on it; for every backbone, the context gate and
    the safe-reference selection compare texts with it.  The same presets as
    Kimodo's loader: a text-encoder service when one is reachable, else local.
    """
    encoder = shared_text_encoder(ctx)
    if encoder is not None:
        return encoder
    from kimodo.model.load_model import (  # noqa: PLC0415
        DEFAULT_TEXT_ENCODER_URL,
        _select_text_encoder_conf,
        get_env_var,
        instantiate_from_dict,
    )

    device = str(device or ctx.device)
    conf = _select_text_encoder_conf(get_env_var("TEXT_ENCODER_URL", DEFAULT_TEXT_ENCODER_URL))
    if "device" in conf:
        conf["device"] = device
    ctx.text_encoder = instantiate_from_dict(conf)
    if hasattr(ctx, "text_encoder_device"):
        ctx.text_encoder_device = device
    return ctx.text_encoder


class FilterAdapter(Protocol):
    """What an adapter model exposes to CSF (documentation only).

    Attributes:
        motion_rep: `.fps`, `.inverse(latent, is_normalized=)`.
        skeleton: A kimodo-compatible skeleton.
        root_dims: Leading channels of the filtered space that are explicit
            root motion (excluded from margin and correction); 0 for none.
    """

    motion_rep: object
    skeleton: object
    root_dims: int

    def reference_feat(self, text: str, num_frames: int, num_steps: int) -> object:
        """The unfiltered prediction for `text`, in the space the filter edits."""
        ...

    def set_filter_context(self, unsafe, safe, cfg, active_mask=None) -> None:
        """Install the per-step filter for these unsafe/safe references."""
        ...

    def clear_filter_context(self) -> None:
        """Revert to plain, unfiltered generation."""
        ...

    def set_filter_segments(self, per_prompt) -> None:
        """Per-segment contexts ``{prompt: (unsafe, safe, cfg, mask)}``; None clears."""
        ...


@dataclass(frozen=True)
class FilterDefaults:
    """Per-backbone filter settings (Table I).

    Attributes:
        gamma: Safe-reference tracking strength.
        decoder_aware: Use the decoder-aware latent path of Eqs. 7-8.
    """

    gamma: float = 0.0
    decoder_aware: bool = False

    def as_overrides(self) -> dict:
        return {
            "gamma": self.gamma,
            "decoder_aware": self.decoder_aware,
        }


@dataclass(frozen=True)
class BackboneSpec:
    """One registered generator.

    Attributes:
        name: Dropdown name, e.g. ``"ECHO-G1-v1"``.
        family: ``"kimodo" | "echo" | "motionhiflow" | "ardy"``.
        config_file: YAML under `configs/backbones/` ("" for none).
        build: ``(spec, ctx) -> kimodo.demo.state.ModelBundle``; `ctx` carries
            ``.device`` and the shared ``.text_encoder``.
        mesh_mode: How the viewer draws it: ``"g1_stl"`` or ``"smplx_skin"``.
        shared_text_encoder: True when generation conditions on the shared
            LLM2Vec (it must then stay on the GPU); False for backbones with
            their own CLIP, which lets the app move LLM2Vec to the CPU.
        filter: Per-backbone filter settings (Table I).
        attach: ``(module, attribute)`` of the function that attaches the
            filter handle, imported on first use.
        ckpt_config_keys: Config keys naming directories that must exist.
        required_paths: Repo-relative paths that must exist.
        description: One line for the GUI.
    """

    name: str
    family: str
    config_file: str
    build: Callable[["BackboneSpec", object], object]
    mesh_mode: str = "g1_stl"
    shared_text_encoder: bool = True
    filter: FilterDefaults = FilterDefaults()
    attach: tuple[str, str] = ADAPTER_ATTACH
    ckpt_config_keys: tuple[str, ...] = ()
    required_paths: tuple[str, ...] = ()
    description: str = ""

    @property
    def attach_filter(self) -> Callable[..., object]:
        """``attach(model, cfg, *, policy, encoder=None) -> handle`` (lazy import)."""
        import importlib  # noqa: PLC0415

        module, attribute = self.attach
        return getattr(importlib.import_module(module), attribute)

    def config_path(self) -> Path | None:
        return BACKBONE_CONFIGS_DIR / self.config_file if self.config_file else None

    def available(self) -> bool:
        """True when this backbone can be loaded here (config and checkpoints present)."""
        cfg_path = self.config_path()
        if cfg_path is not None and not cfg_path.exists():
            return False
        for rel in self.required_paths:
            if not (REPO_ROOT / rel).exists():
                return False
        if not self.ckpt_config_keys:
            return True
        try:
            import yaml  # noqa: PLC0415

            with open(cfg_path, encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle) or {}
            for key in self.ckpt_config_keys:
                path = Path(str(cfg[key]))
                if not path.is_absolute():
                    path = REPO_ROOT / path
                if not path.exists():
                    return False
        except Exception:  # noqa: BLE001 - any config problem means unavailable
            return False
        return True


class PerSegmentFilterMixin:
    """Per-segment filter contexts for a multi-prompt schedule.

    Adapters generate a schedule as one sampling run per prompt segment, so each
    segment is filtered with the gate decision for its own text.  Mix into a
    model and call `_activate_filter_for(text)` at the top of each segment.
    """

    def set_filter_segments(self, per_prompt) -> None:
        self._filter_per_prompt = dict(per_prompt or {})

    def _activate_filter_for(self, text: str) -> None:
        per_prompt = getattr(self, "_filter_per_prompt", None)
        if not per_prompt:
            return
        unsafe, safe, cfg, mask = per_prompt.get(text, (None, None, None, None))
        if unsafe is None or cfg is None:
            self.clear_filter_context()
            return
        self.set_filter_context(unsafe, safe, cfg, active_mask=mask)
