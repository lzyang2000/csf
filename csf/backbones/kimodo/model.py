# SPDX-License-Identifier: Apache-2.0
"""Load Kimodo-G1 from kimodo's Hugging Face registry into a `ModelBundle`.

Kimodo loads the shared LLM2Vec text encoder; the builder publishes it back
onto the load context so ARDY and the filter's gate reuse the same instance
instead of loading a second 8B model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from csf.backbones.base import shared_text_encoder

if TYPE_CHECKING:
    from kimodo.demo.state import ModelBundle

    from csf.backbones.base import BackboneSpec


def build_kimodo_bundle(spec: "BackboneSpec", ctx: object) -> "ModelBundle":
    """Load `spec.name` (e.g. "Kimodo-G1-RP-v1") on `ctx.device`."""
    from kimodo.demo.embedding_cache import CachedTextEncoder  # noqa: PLC0415
    from kimodo.demo.state import ModelBundle  # noqa: PLC0415
    from kimodo.model.load_model import load_model  # noqa: PLC0415
    from kimodo.model.registry import MODEL_NAMES, resolve_model_name  # noqa: PLC0415

    short_key = resolve_model_name(spec.name, "Kimodo")
    if short_key not in MODEL_NAMES:
        raise ValueError(f"Unknown Kimodo model {spec.name!r}; released: {list(MODEL_NAMES)}")

    # kimodo's loader stores the device in an OmegaConf dict: pass a string.
    device = str(ctx.device)
    shared = shared_text_encoder(ctx)
    print(f"[csf] loading {spec.name} ({short_key}) on {device}...")
    model = load_model(modelname=short_key, device=device, text_encoder=shared)

    if hasattr(model, "text_encoder"):
        if shared is None:
            ctx.text_encoder = model.text_encoder
        if not isinstance(model.text_encoder, CachedTextEncoder):
            # Keyed by kimodo's short key: that is its prompt-cache directory.
            model.text_encoder = CachedTextEncoder(model.text_encoder, model_name=short_key)

    return ModelBundle(
        model=model,
        motion_rep=model.motion_rep,
        skeleton=model.motion_rep.skeleton,
        model_fps=model.motion_rep.fps,
    )
