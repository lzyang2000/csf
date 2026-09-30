# SPDX-License-Identifier: Apache-2.0
"""The four generators of the paper, one `BackboneSpec` each (Table I).

| Backbone     | Prediction space        | gamma | Root channels  |
|--------------|-------------------------|-------|----------------|
| Kimodo-G1    | 417-d motion feature    | 0     | excluded (5)   |
| ECHO         | 38-d motion feature     | 0.2   | not separable  |
| MotionHiFlow | VAE latent              | 0.5   | not separable  |
| ARDY         | hybrid token (20 + 128) | 0.2   | excluded (20)  |

Root exclusion is declared by each model (`root_dims`); gamma lives here.  Adding a backbone is one adapter package, one
YAML under `configs/backbones/` and one entry below.

No heavy imports at module level: builders import their model stacks inside.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from csf.backbones.base import BackboneSpec, FilterDefaults, ensure_text_encoder, shared_text_encoder

if TYPE_CHECKING:
    from kimodo.demo.state import ModelBundle


def _load_cfg(spec: BackboneSpec) -> dict:
    import yaml  # noqa: PLC0415

    with open(spec.config_path(), encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_kimodo(spec: BackboneSpec, ctx: object) -> "ModelBundle":
    from csf.backbones.kimodo import model  # noqa: PLC0415

    return model.build_kimodo_bundle(spec, ctx)


def _build_echo(spec: BackboneSpec, ctx: object) -> "ModelBundle":
    from csf.backbones.echo import model  # noqa: PLC0415

    return model.build_echo_bundle(_load_cfg(spec), ctx.device, text_encoder=shared_text_encoder(ctx))


def _build_mhf(spec: BackboneSpec, ctx: object) -> "ModelBundle":
    from csf.backbones.motionhiflow import model  # noqa: PLC0415

    return model.build_mhf_bundle(_load_cfg(spec), ctx.device, text_encoder=shared_text_encoder(ctx))


def _build_ardy(spec: BackboneSpec, ctx: object) -> "ModelBundle":
    from csf.backbones.ardy import model  # noqa: PLC0415

    # ARDY conditions on the same LLM2Vec as Kimodo.
    return model.build_ardy_bundle(_load_cfg(spec), ctx.device, text_encoder=ensure_text_encoder(ctx))


_SPECS: tuple[BackboneSpec, ...] = (
    BackboneSpec(
        name="Kimodo-G1-RP-v1",
        family="kimodo",
        config_file="",
        build=_build_kimodo,
        mesh_mode="g1_stl",
        shared_text_encoder=True,
        filter=FilterDefaults(gamma=0.0),
        attach=("csf.backbones.kimodo.filtered_model", "attach_kimodo_filter"),
        description="DDIM diffusion over a 417-d G1 motion feature, 30 fps.",
    ),
    BackboneSpec(
        name="ECHO-G1-v1",
        family="echo",
        config_file="echo.yaml",
        build=_build_echo,
        mesh_mode="g1_stl",
        shared_text_encoder=False,
        filter=FilterDefaults(gamma=0.2),
        ckpt_config_keys=("ckpt_dir",),
        description="Diffusion with DPM-Solver over a 38-d G1 motion feature, 50 fps.",
    ),
    BackboneSpec(
        name="MotionHiFlow-SMPL-v1",
        family="motionhiflow",
        config_file="motionhiflow.yaml",
        build=_build_mhf,
        mesh_mode="smplx_skin",
        shared_text_encoder=False,
        filter=FilterDefaults(gamma=0.5),
        ckpt_config_keys=("vae_dir", "dit_dir"),
        description="Flow matching in a VAE latent, HumanML3D SMPL body, 20 fps.",
    ),
    BackboneSpec(
        name="ARDY-G1-v1",
        family="ardy",
        config_file="ardy.yaml",
        build=_build_ardy,
        mesh_mode="g1_stl",
        shared_text_encoder=True,
        filter=FilterDefaults(gamma=0.2),
        required_paths=("third_party/ardy/ardy/__init__.py",),
        description="Windowed autoregressive diffusion over hybrid tokens, G1, 25 fps.",
    ),
)

SPECS: dict[str, BackboneSpec] = {spec.name: spec for spec in _SPECS}


def all_names() -> list[str]:
    """Every registered backbone, in dropdown order."""
    return list(SPECS)


def available_names() -> list[str]:
    """The registered backbones whose configs and checkpoints are present."""
    return [name for name, spec in SPECS.items() if spec.available()]


def get_spec(name: str) -> BackboneSpec:
    """The spec for `name`.

    Raises:
        KeyError: Unknown name (the message lists the registered ones).
    """
    try:
        return SPECS[name]
    except KeyError:
        raise KeyError(f"Unknown backbone {name!r}; registered: {', '.join(SPECS)}") from None


def load_backbone(name: str, ctx: object) -> "ModelBundle":
    """Build the bundle for `name` (heavy imports happen here)."""
    spec = get_spec(name)
    return spec.build(spec, ctx)
