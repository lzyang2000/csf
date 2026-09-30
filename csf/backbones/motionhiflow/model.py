# SPDX-License-Identifier: Apache-2.0
"""MotionHiFlow behind kimodo's generation interface, with the CSF adapter protocol.

`MHFModel` wraps MotionHiFlow's flow DiT and VAE so the app drives it like a
Kimodo model: ``model(prompts, num_frames, num_denoising_steps,
multi_prompt=True, ...)`` returns kimodo's motion dict on the 22-joint SMPL-X
body.  As in MotionHiFlow's `gen_t2m.py`, the DiT samples a latent at a quarter
of the frame rate (``frames // 4`` tokens) and the VAE decodes it to
normalised 263-d HumanML3D features.  MotionHiFlow conditions on its own CLIP
with a fixed guidance scale and has no constraint or heading inputs, so
kimodo's conditioning keyword arguments are accepted and ignored.

Filtering (Sec. III-E): the filtered space is the clean VAE latent.  Once
`set_filter_context` installs the unsafe and safe references, generation runs
`filtered_flow_generate`, which applies Eq. 3 to the clean latent at every
Euler step (`csf_adapter`), or, with ``cfg.decoder_aware``, the decoder-aware
iteration of Eqs. 7-8 (`decoded_margin`).  The references come from
`reference_feat`, MotionHiFlow's own unfiltered clean latent for a text.  A
multi-prompt schedule is generated one run per segment and cross-faded, and
each segment is filtered with the gate decision for its own text.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from csf.backbones.base import PerSegmentFilterMixin

if TYPE_CHECKING:
    from kimodo.demo.state import ModelBundle


def _to_numpy_dict(d: dict) -> dict:
    return {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def _crossfade(segments: Sequence[Tensor], num_transition_frames: int) -> Tensor:
    """Concatenate ``[T_i, D]`` segments, blending each seam linearly over n frames."""
    out = [segments[0]]
    for nxt in segments[1:]:
        prev = out[-1]
        n = min(num_transition_frames, prev.shape[0], nxt.shape[0])
        if n < 1:
            out.append(nxt)
            continue
        alpha = torch.linspace(1.0, 0.0, n, device=prev.device).unsqueeze(1)
        out[-1] = prev[:-n]
        out.append(alpha * prev[-n:] + (1.0 - alpha) * nxt[:n])
        out.append(nxt[n:])
    return torch.cat(out, dim=0)


def _latent_of(result):
    """`FlowModel.generate` returns ``(latent, lengths)``; accept a bare latent too."""
    return result[0] if isinstance(result, tuple) else result


class MHFModel(PerSegmentFilterMixin):
    """MotionHiFlow with kimodo's generation interface and the CSF adapter protocol.

    Args:
        vae: MotionHiFlow's VAE (``decode(latent) -> [B, T, 263]``).
        flow: MotionHiFlow's flow DiT (``generate(text, m_lengths, time_steps,
            cond_scale) -> (latent [B, T // 4, 6, 16], lengths)``).
        inv_transform: ``x * std + mean`` for 263-d numpy features.
        motion_rep: `MHFMotionRep`.
        skeleton: kimodo's `SMPLXSkeleton22`.
        text_encoder: The shared LLM2Vec encoder, or None.  CSF's gate and
            safe-reference selection use it; MotionHiFlow conditions on CLIP.
        device: Torch device string.
        cond_scale: Classifier-free guidance scale.
        time_steps: Default flow-matching step count.

    Attributes:
        root_dims: 0: the VAE latent has no separable root block (Table I).
        default_denoising_steps: The trained step count (`time_steps`).
    """

    root_dims: int = 0

    def __init__(self, vae, flow, inv_transform: Callable, motion_rep, skeleton, text_encoder,
                 device: str, cond_scale: float = 4.5, time_steps: int = 12) -> None:
        self._vae = vae
        self._flow = flow
        self._inv_transform = inv_transform
        self.motion_rep = motion_rep
        self.skeleton = skeleton
        self.text_encoder = text_encoder
        self.device = device
        self.fps: int = motion_rep.fps
        self.cond_scale = cond_scale
        self.time_steps = time_steps
        self.default_denoising_steps = int(time_steps)
        self._latent_callback = None               # None => MotionHiFlow's native sampling

    # ------------------------------------------------------------ CSF protocol
    def reference_feat(self, text: str, num_frames: int, num_steps: int) -> Tensor:
        """MotionHiFlow's unfiltered clean latent for `text`: ``[T // 4, 6, 16]``.

        Sampled with the native sampler, so any installed filter does not
        touch it.
        """
        return self._generate_latent([text], int(num_frames), int(num_steps))[0]

    def set_filter_context(self, unsafe, safe, cfg, active_mask=None) -> None:
        """Filter every Euler step against these latent references (None clears).

        Args:
            unsafe: Unsafe latent references ``[K, 1, T, 6, 16]``.
            safe: Safe latent reference ``[1, T, 6, 16]``.
            cfg: The `FilterConfig` in force; ``cfg.decoder_aware`` selects the
                decoder-aware path of Eqs. 7-8 over the latent QP.
            active_mask: The gate's rule selection; None enforces every rule.
        """
        if unsafe is None:
            self._latent_callback = None
            return
        if getattr(cfg, "decoder_aware", False):
            from .decoded_margin import make_decoded_step_callback  # noqa: PLC0415

            self._latent_callback = make_decoded_step_callback(self._vae, unsafe, safe, cfg,
                                                               active_mask=active_mask)
        else:
            from .csf_adapter import make_latent_step_callback  # noqa: PLC0415

            self._latent_callback = make_latent_step_callback(unsafe, safe, cfg,
                                                              active_mask=active_mask)

    def clear_filter_context(self) -> None:
        """Revert to MotionHiFlow's native sampling."""
        self._latent_callback = None

    # -------------------------------------------------------------- generation
    def _token_lengths(self, batch: int, max_frames: int) -> Tensor:
        """Latent lengths for `max_frames` output frames (``frames // 4``, as gen_t2m.py)."""
        return torch.tensor([max(int(max_frames) // 4, 1)] * batch, dtype=torch.long,
                            device=self.device)

    def _generate_latent(self, texts: List[str], max_frames: int,
                         num_denoising_steps: Optional[int] = None) -> Tensor:
        """Native (unfiltered) clean latents ``[B, T // 4, 6, 16]``."""
        steps = self.time_steps if num_denoising_steps is None else num_denoising_steps
        with torch.no_grad():
            result = self._flow.generate(text=texts, m_lengths=self._token_lengths(len(texts), max_frames),
                                         time_steps=steps, cond_scale=self.cond_scale)
        return _latent_of(result)

    def _generate(self, texts: List[str], max_frames: int,
                  num_denoising_steps: Optional[int] = None, **_ignored) -> Tensor:
        """Sample and decode ``[B, T, 263]`` normalised features, one clip per text."""
        steps = self.time_steps if num_denoising_steps is None else num_denoising_steps
        lengths = self._token_lengths(len(texts), max_frames)
        with torch.no_grad():
            if self._latent_callback is not None:
                from .filtered_sampler import filtered_flow_generate  # noqa: PLC0415

                result = filtered_flow_generate(self._flow, text=texts, m_lengths=lengths,
                                                time_steps=steps, cond_scale=self.cond_scale,
                                                step_callback=self._latent_callback)
            else:
                result = self._flow.generate(text=texts, m_lengths=lengths, time_steps=steps,
                                             cond_scale=self.cond_scale)
            return self._vae.decode(_latent_of(result))

    def _decode(self, motion: Tensor) -> dict:
        """Denormalise one ``[T, 263]`` clip and decode it to kimodo's motion dict."""
        denorm = self._inv_transform(motion.detach().cpu().numpy().astype(np.float32))
        return self.motion_rep.inverse(denorm, is_normalized=False, return_numpy=False)

    def _multiprompt(self, prompts: List[str], num_frames: Union[int, List[int]],
                     num_denoising_steps: Optional[int] = None, *, return_numpy: bool = False,
                     num_transition_frames: int = 5, **_ignored) -> dict:
        """Generate the segments in order and cross-fade them in normalised space.

        Each segment is its own sampling run, filtered with its own context
        when per-segment contexts are installed.  Returns a batched ``[1, T, ...]``
        dict, as kimodo's multi-prompt generation does.
        """
        if isinstance(num_frames, int):
            num_frames = [num_frames] * len(prompts)
        segments = []
        for text, frames in zip(prompts, num_frames):
            self._activate_filter_for(text)
            segments.append(self._generate([text], frames, num_denoising_steps)[0])
        stitched = _crossfade(segments, num_transition_frames)
        output = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                  for k, v in self._decode(stitched).items()}
        return _to_numpy_dict(output) if return_numpy else output

    def __call__(self, prompts: Union[str, List[str]], num_frames: Union[int, List[int]],
                 num_denoising_steps: Optional[int] = None, *, multi_prompt: bool = False,
                 return_numpy: bool = False, num_transition_frames: int = 5, **_ignored) -> dict:
        """Generate motion and return kimodo's motion dict.

        Input handling follows kimodo: one prompt and one length give an
        unbatched ``[T, ...]`` dict; a list of prompts or lengths gives a batch
        (equal lengths only); ``multi_prompt=True`` treats the prompts as a
        sequential schedule and returns a batched ``[1, T, ...]`` dict.

        Returns:
            ``posed_joints``, ``global_rot_mats``, ``local_rot_mats``,
            ``root_positions``, ``smooth_root_pos``, ``foot_contacts`` and
            ``global_root_heading``.
        """
        if multi_prompt:
            prompts = [prompts] if isinstance(prompts, str) else prompts
            return self._multiprompt(prompts, num_frames, num_denoising_steps,
                                     return_numpy=return_numpy,
                                     num_transition_frames=num_transition_frames)

        squeeze = False
        if isinstance(prompts, list) and isinstance(num_frames, list):
            assert len(prompts) == len(num_frames), "Number of prompts must match number of num_frames."
            texts, lengths = prompts, num_frames
        elif isinstance(prompts, list):
            texts, lengths = prompts, [num_frames] * len(prompts)
        elif isinstance(num_frames, list):
            texts, lengths = [prompts] * len(num_frames), num_frames
        else:
            squeeze = True
            texts, lengths = [prompts], [int(num_frames)]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"MHFModel batched generation requires equal num_frames across prompts; got "
                f"{lengths}. Use separate calls or multi_prompt for variable lengths."
            )

        motion = self._generate(texts, max(lengths), num_denoising_steps)
        if squeeze:
            output = self._decode(motion[0])
        else:
            outputs = [self._decode(motion[i, :lengths[i]]) for i in range(len(texts))]
            output = {k: torch.stack([o[k] for o in outputs], dim=0) for k in outputs[0]}
        return _to_numpy_dict(output) if return_numpy else output


def build_mhf_bundle(cfg_dict: dict, device: str, text_encoder=None) -> "ModelBundle":
    """Load MotionHiFlow from `configs/backbones/motionhiflow.yaml` settings into a `ModelBundle`.

    Args:
        cfg_dict: The MotionHiFlow backbone config (`vae_dir`, `dit_dir`,
            `cond_scale`, `time_steps`, `fps`, `joints_num`, `dataset`).
        device: Torch device string.
        text_encoder: The shared LLM2Vec encoder, or None.
    """
    from kimodo.demo.state import ModelBundle  # noqa: PLC0415
    from kimodo.skeleton.definitions import SMPLXSkeleton22  # noqa: PLC0415

    from .loader import load_mhf_models  # noqa: PLC0415
    from .motion_rep import MHFMotionRep  # noqa: PLC0415

    vae, flow, inv_transform = load_mhf_models(cfg_dict, device)
    fps = int(cfg_dict.get("fps", 20))
    time_steps = int(cfg_dict.get("time_steps", 12))
    # HumanML3D's 22-joint order is SMPL-X's body order; kimodo's skeleton
    # carries the rest pose and names the SMPL-X mesh renderer needs.
    skeleton = SMPLXSkeleton22(load=True)
    motion_rep = MHFMotionRep(joints_num=int(cfg_dict.get("joints_num", 22)), fps=fps,
                              rest_joints=skeleton.neutral_joints,
                              joint_parents=skeleton.joint_parents)
    model = MHFModel(
        vae=vae, flow=flow, inv_transform=inv_transform, motion_rep=motion_rep,
        skeleton=skeleton, text_encoder=text_encoder, device=device,
        cond_scale=float(cfg_dict.get("cond_scale", 4.5)), time_steps=time_steps,
    )
    return ModelBundle(model=model, motion_rep=motion_rep, skeleton=skeleton, model_fps=float(fps))
