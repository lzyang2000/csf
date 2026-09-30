# SPDX-License-Identifier: Apache-2.0
"""ARDY-G1 as a CSF adapter model.

ARDY (nv-tlabs/ardy, vendored at third_party/ardy) is windowed autoregressive
diffusion over hybrid tokens: per token of ``num_frames_per_token`` frames, the
explicit normalized root motion (20 channels on ARDY-G1) followed by an FSQ
body-latent embedding (128).  `ArdyModel` wraps a loaded ARDY model in the
kimodo generation interface the app uses and implements the adapter protocol
of `csf.backbones.base`:

* `ArdyModel.reference_feat` runs an unfiltered rollout for a text and returns
  its hybrid-token sequence [T_tok, D], the space the filter edits;
* `ArdyModel.set_filter_context` installs a
  `csf.backbones.ardy.filter_proxy.FilteredDenoiser` over ARDY's denoiser, so
  filtering happens inside ARDY's own sampling loop on each window's clean
  token prediction, before ARDY commits and requantizes the window (Eq. 9).
  Filtered and plain generation run the same code path;
* `ArdyModel.root_dims` is the width of the explicit root block, which the
  filter leaves to the generator.

ARDY conditions on the shared LLM2Vec text encoder (the same checkpoint as
Kimodo), which therefore stays on the model's device.  Text classifier-free
guidance is supported; constraint conditioning is not exposed.
"""
from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from csf.backbones.base import PerSegmentFilterMixin
from csf.paths import THIRD_PARTY_DIR

if TYPE_CHECKING:
    from kimodo.demo.state import ModelBundle

_ARDY_ROOT = THIRD_PARTY_DIR / "ardy"

#: Longest autoregressive window (history + generation) ARDY attends over, in
#: seconds; the default history budget keeps every window within it.
_ATTENTION_WINDOW_SECONDS = 10


def ensure_ardy_importable() -> None:
    """Put the vendored ARDY package (third_party/ardy) on `sys.path`."""
    path = str(_ARDY_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _to_numpy_dict(d: dict) -> dict:
    return {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in d.items()}


class ArdyModel(PerSegmentFilterMixin):
    """Kimodo-interface adapter over ARDY's autoregressive diffusion model.

    Args:
        ardy_model: A loaded ``ardy.model.ardy_model.Ardy`` (or any object
            exposing ``__call__``, ``hybrid``, ``motion_rep``, ``denoiser``,
            ``num_frames_per_token``, ``gen_horizon_len`` and ``diffusion``).
        skeleton: Kimodo's ``G1Skeleton34`` (same bone order as ARDY's), so
            the viewer draws the G1 mesh.
        text_encoder: The shared LLM2Vec encoder ARDY conditions on.
        device: Torch device.
        fps: Frame rate of the checkpoint.
        cfg_weight: Default text classifier-free guidance weight.
        history_frames: History budget per autoregressive window, in frames (a
            multiple of ``num_frames_per_token``); None keeps the full history.
        default_denoising_steps: Steps used when the caller has no better
            value; defaults to the checkpoint's trained step count.

    Attributes:
        motion_rep, skeleton, text_encoder, device, fps, default_denoising_steps.
    """

    def __init__(
        self,
        ardy_model,
        skeleton,
        text_encoder,
        device,
        fps: int = 25,
        cfg_weight: float = 2.0,
        history_frames: Optional[int] = None,
        default_denoising_steps: Optional[int] = None,
    ) -> None:
        self._ardy = ardy_model
        self.motion_rep = ardy_model.motion_rep
        self.skeleton = skeleton
        self.text_encoder = text_encoder
        self.device = device
        self.fps: int = int(fps)
        self._cfg_weight = float(cfg_weight)
        self._history_frames = history_frames
        self.default_denoising_steps = int(default_denoising_steps or self.max_denoising_steps)

        # The installed filter proxy and the denoiser it wraps (None: plain generation).
        self._proxy = None
        self._inner_denoiser = None

    @property
    def max_denoising_steps(self) -> int:
        """The checkpoint's trained step count, which sampling cannot exceed."""
        return int(self._ardy.diffusion.num_base_steps)

    # ------------------------------------------------------------ filter protocol
    @property
    def root_dims(self) -> int:
        """Leading channels of the hybrid token that are explicit root motion (20).

        The filter excludes them from the margin and the correction.  Each
        window conditions on the committed tokens of the previous one, so an
        edited root would make the next window resume from a root state it did
        not predict; only the FSQ body latent carries the semantic edit.
        """
        return int(getattr(self._ardy.denoiser, "nframe_root_dim", 0) or 0)

    def set_filter_context(self, unsafe, safe, cfg, active_mask=None) -> None:
        """Install the filter over ARDY's denoiser.

        Args:
            unsafe: Unsafe references from `reference_feat`, [K, 1, T_tok, D].
            safe: Safe reference from `reference_feat`, [1, T_tok, D].
            cfg: The `FilterConfig` in force.
            active_mask: bool[K] from the context gate; None enforces every rule.

        ``unsafe=None`` clears the context.  A multi-prompt schedule installs
        one context per segment through `set_filter_segments`.
        """
        self.clear_filter_context()
        if unsafe is None:
            return
        from .filter_proxy import FilteredDenoiser  # noqa: PLC0415

        self._inner_denoiser = self._ardy.denoiser
        self._proxy = FilteredDenoiser(
            self._inner_denoiser, unsafe, safe, cfg, active_mask=active_mask,
        )
        self._ardy.denoiser = self._proxy

    def clear_filter_context(self) -> None:
        """Restore ARDY's own denoiser (plain generation)."""
        if self._inner_denoiser is not None:
            self._ardy.denoiser = self._inner_denoiser
        self._proxy = None
        self._inner_denoiser = None

    @contextlib.contextmanager
    def _unfiltered(self):
        """Suspend an installed filter for the duration of the block."""
        proxy, inner = self._proxy, self._inner_denoiser
        if proxy is None:
            yield
            return
        self._ardy.denoiser, self._proxy = inner, None
        try:
            yield
        finally:
            self._ardy.denoiser, self._proxy = proxy, proxy

    def reference_feat(self, text: str, num_frames: int, num_steps: int) -> torch.Tensor:
        """The unfiltered hybrid-token sequence [T_tok, D] for `text`.

        Runs an unfiltered rollout and encodes its normalized motion with
        ARDY's own tokenizer, so references live in exactly the space the
        filter edits.  The tokenizer folds ``num_frames_per_token`` frames per
        token, so a frame count that is not a multiple of it is cropped to the
        last full token.
        """
        with self._unfiltered():
            motion = self._generate([text], int(num_frames), int(num_steps))  # [1, T, D_motion]
        fpt = int(self._ardy.num_frames_per_token)
        motion = motion[:, :(motion.shape[1] // fpt) * fpt]
        with torch.no_grad():
            tokens, _ = self._ardy.hybrid.get_hybrid_motion_from_explicit(
                motion=motion,
                motion_len=torch.tensor([motion.shape[1]], device=motion.device),
                motion_pad_mask=torch.ones(
                    1, motion.shape[1], device=motion.device, dtype=torch.bool
                ),
            )  # [1, T_tok, D_hybrid]
        return tokens.squeeze(0).float()

    # ------------------------------------------------------------------ sampling
    def _text_cfg_weight(self, cfg_weight) -> float:
        """The text guidance weight.

        Kimodo passes ``[text, constraint]`` pairs.  This adapter never
        conditions on constraints, and a nonzero constraint weight would still
        add a constraint term to ARDY's guidance combine, so a pair collapses
        to its text weight.
        """
        if cfg_weight is None:
            return self._cfg_weight
        if isinstance(cfg_weight, (tuple, list)):
            return float(cfg_weight[0]) if len(cfg_weight) else self._cfg_weight
        return float(cfg_weight)

    def _generate(
        self,
        texts: List[str],
        num_frames: int,
        num_denoising_steps: Optional[int],
        *,
        first_heading_angle=None,
        cfg_weight=None,
        cfg_type=None,
    ) -> torch.Tensor:
        """One autoregressive rollout per text: normalized motion features [B, T, D]."""
        ensure_ardy_importable()
        from ardy.motion_rep.tools import length_to_mask  # noqa: PLC0415

        B = len(texts)
        nf = int(num_frames)
        steps = int(num_denoising_steps) if num_denoising_steps else self.max_denoising_steps
        steps = max(1, min(steps, self.max_denoising_steps))

        if self._proxy is not None:
            self._proxy.reset()  # fresh rollout: realign windows with the references

        lengths = torch.tensor([nf] * B, device=self.device)
        if first_heading_angle is None:
            first_heading_angle = torch.zeros(B, device=self.device)
        with torch.no_grad():
            return self._ardy(
                texts,
                nf,
                num_denoising_steps=steps,
                pad_mask=length_to_mask(lengths),
                first_heading_angle=first_heading_angle,
                motion_mask=None,
                observed_motion=None,
                cfg_weight=self._text_cfg_weight(cfg_weight),
                cfg_type=cfg_type,
                progress_bar=lambda it: it,
                crop_history_length=self._history_frames,
            )

    def _multiprompt(
        self,
        prompts: List[str],
        num_frames: List[int],
        num_denoising_steps: Optional[int],
        *,
        cfg_weight=None,
        cfg_type=None,
        num_transition_frames: int = 5,
    ) -> torch.Tensor:
        """One rollout per segment, stitched by a linear cross-fade.

        Each segment is its own sampling run, so it is filtered with the
        context of its own text (`PerSegmentFilterMixin`).  Segments are
        blended over ``num_transition_frames`` frames in normalized motion
        space.

        Returns:
            Normalized motion features [T_total, D].
        """
        segments: list[torch.Tensor] = []
        for text, nf in zip(prompts, num_frames):
            self._activate_filter_for(text)
            segment = self._generate(
                [text], int(nf), num_denoising_steps, cfg_weight=cfg_weight, cfg_type=cfg_type,
            )
            segments.append(segment[0])  # [T, D]

        parts: list[torch.Tensor] = [segments[0]]
        for seg_b in segments[1:]:
            seg_a = parts[-1]
            n = min(num_transition_frames, seg_a.shape[0], seg_b.shape[0])
            if n < 1:
                parts.append(seg_b)
                continue
            alpha = torch.linspace(1.0, 0.0, n, device=seg_a.device).unsqueeze(1)
            parts[-1] = seg_a[:-n]
            parts.append(alpha * seg_a[-n:] + (1.0 - alpha) * seg_b[:n])
            parts.append(seg_b[n:])
        return torch.cat(parts, dim=0)

    def __call__(
        self,
        prompts: Union[str, List[str]],
        num_frames: Union[int, List[int]],
        num_denoising_steps: Optional[int],
        *,
        multi_prompt: bool = False,
        constraint_lst=None,
        cfg_weight=None,
        cfg_type=None,
        num_samples: Optional[int] = None,
        return_numpy: bool = False,
        first_heading_angle=None,
        num_transition_frames: int = 5,
        **_unused,
    ) -> dict:
        """Generate motion in kimodo's output format.

        Args:
            prompts: One text, or one text per segment (``multi_prompt=True``)
                or per batch row.
            num_frames: Frames per segment (multi-prompt) or for every row.
            num_denoising_steps: Steps per window, capped at the trained count.
            multi_prompt: Generate the prompts as consecutive segments of one
                motion; otherwise as a batch of equal-length rollouts.
            constraint_lst: Ignored (no constraint conditioning).
            cfg_weight: Text guidance weight, or kimodo's ``[text, constraint]``
                pair; None uses the configured default.
            cfg_type: Passed through to ARDY's guidance.
            num_samples: Ignored (one sample per prompt).
            return_numpy: Return numpy arrays instead of tensors.
            first_heading_angle: Initial heading [B] for batched generation.
            num_transition_frames: Cross-fade length between segments.

        Returns:
            ARDY's decoded motion: ``posed_joints`` [B, T, J, 3],
            ``global_rot_mats`` / ``local_rot_mats`` [B, T, J, 3, 3],
            ``root_positions`` / ``smooth_root_pos`` [B, T, 3], ``foot_contacts``
            and ``global_root_heading``; B = 1 for a multi-prompt schedule.
        """
        if multi_prompt:
            prompts = [prompts] if isinstance(prompts, str) else list(prompts)
            if isinstance(num_frames, int):
                num_frames = [num_frames] * len(prompts)
            motion = self._multiprompt(
                prompts, list(num_frames), num_denoising_steps,
                cfg_weight=cfg_weight, cfg_type=cfg_type,
                num_transition_frames=num_transition_frames,
            ).unsqueeze(0)
        else:
            texts = [prompts] if isinstance(prompts, str) else list(prompts)
            if isinstance(num_frames, (list, tuple)):
                if len(set(num_frames)) > 1:
                    raise ValueError(
                        f"batched generation needs equal num_frames, got {list(num_frames)}; "
                        "use multi_prompt=True for segments of different lengths"
                    )
                num_frames = num_frames[0]
            motion = self._generate(
                texts, int(num_frames), num_denoising_steps,
                first_heading_angle=first_heading_angle,
                cfg_weight=cfg_weight, cfg_type=cfg_type,
            )  # [B, T, D]

        with torch.no_grad():
            output = self.motion_rep.inverse(motion, is_normalized=True)
        return _to_numpy_dict(output) if return_numpy else output


def default_history_frames(fps: int, num_frames_per_token: int, gen_horizon_len: int) -> int:
    """The longest history that keeps each window within the trained attention span.

    A window is history plus ``gen_horizon_len`` generated frames; both are
    whole tokens, and the total may not exceed `_ATTENTION_WINDOW_SECONDS`.
    This is the rule ARDY's own scripts use (196 frames for ARDY-G1 at 25 fps).
    """
    fpt = int(num_frames_per_token)
    max_window = (int(_ATTENTION_WINDOW_SECONDS * fps) // fpt) * fpt
    return ((max_window - int(gen_horizon_len)) // fpt) * fpt


def build_ardy_bundle(cfg_dict: dict, device, text_encoder=None) -> "ModelBundle":
    """Load the released ARDY-G1 checkpoint into a `ModelBundle`.

    Weights download from Hugging Face on first use.  The shared LLM2Vec
    encoder is required: ARDY conditions on it, and reusing the loaded
    instance avoids a second 8B encoder.

    Args:
        cfg_dict: ``configs/backbones/ardy.yaml`` (``model_name``,
            ``num_denoising_steps``, ``cfg_weight``, optional ``history_frames``).
        device: Torch device.
        text_encoder: The shared LLM2Vec encoder.
    """
    if text_encoder is None:
        raise ValueError(
            "build_ardy_bundle requires the shared LLM2Vec text_encoder: ARDY "
            "conditions generation on it."
        )
    ensure_ardy_importable()
    from ardy.model.load_model import load_model  # noqa: PLC0415
    from kimodo.demo.state import ModelBundle  # noqa: PLC0415
    from kimodo.skeleton.registry import build_skeleton  # noqa: PLC0415

    model_name = str(cfg_dict.get("model_name", "g1"))
    ardy_model = load_model(model_name, device=device, text_encoder=text_encoder)
    ardy_model.eval()

    if type(ardy_model.skeleton).__name__ != "G1Skeleton34":
        raise ValueError(
            f"ARDY checkpoint {model_name!r} uses {type(ardy_model.skeleton).__name__}; "
            "only the G1 checkpoint (ARDY-G1-RP-25FPS-Horizon52) is supported"
        )
    # Kimodo's G1 skeleton has ARDY's bone order and drives the G1 mesh.
    skeleton = build_skeleton(34)
    fps = int(round(float(ardy_model.motion_rep.fps)))

    history_frames = cfg_dict.get("history_frames")
    if history_frames is None:
        history_frames = default_history_frames(
            fps, ardy_model.num_frames_per_token, ardy_model.gen_horizon_len,
        )

    steps = cfg_dict.get("num_denoising_steps")
    model = ArdyModel(
        ardy_model=ardy_model,
        skeleton=skeleton,
        text_encoder=text_encoder,
        device=device,
        fps=fps,
        cfg_weight=float(cfg_dict.get("cfg_weight", 2.0)),
        history_frames=int(history_frames),
        default_denoising_steps=int(steps) if steps is not None else None,
    )
    return ModelBundle(
        model=model,
        motion_rep=ardy_model.motion_rep,
        skeleton=skeleton,
        model_fps=float(fps),
    )
