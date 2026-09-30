# SPDX-License-Identifier: Apache-2.0
"""ECHO behind kimodo's generation interface, with the CSF adapter protocol.

`EchoModel` wraps ECHO's `DiffusePipeline` so the app drives it like a Kimodo
model: ``model(prompts, num_frames, num_denoising_steps, multi_prompt=True,
...)`` returns kimodo's motion dict on the 34-joint G1.  ECHO conditions on its
own CLIP text encoder and has no constraint, heading or guidance-weight
inputs, so kimodo's conditioning keyword arguments are accepted and ignored.

Filtering (Sec. III-E, direct motion space): ECHO predicts the clean,
normalised 38-d feature at every denoising step.  Once `set_filter_context`
installs the unsafe and safe references, generation runs
`filtered_generate_batch`, which applies Eq. 3 to each estimate before the
scheduler update.  The references come from `reference_feat`, ECHO's own
unfiltered prediction for a text in that same space.  A multi-prompt schedule
is generated one denoising run per segment and cross-faded, and each segment
is filtered with the gate decision for its own text (`set_filter_segments`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

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


class EchoModel(PerSegmentFilterMixin):
    """ECHO with kimodo's generation interface and the CSF adapter protocol.

    Args:
        pipeline: ECHO's `DiffusePipeline`.
        mean: The 38-d normalisation mean.
        std: The 38-d normalisation std.
        motion_rep: `EchoMotionRep`.
        skeleton: kimodo's `G1Skeleton34`.
        text_encoder: The shared LLM2Vec encoder, or None.  CSF's gate and
            safe-reference selection use it; ECHO itself conditions on CLIP.
        device: Torch device string.
        default_denoising_steps: ECHO's trained DPM-Solver step count (its
            schedule degrades when heavily over-stepped).

    Attributes:
        root_dims: 0: the 38-d feature has no separable leading root block, so
            the whole vector is filtered (Table I).
    """

    root_dims: int = 0

    def __init__(self, pipeline, mean: np.ndarray, std: np.ndarray, motion_rep, skeleton,
                 text_encoder, device: str, default_denoising_steps: int = 10) -> None:
        self._pipeline = pipeline
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.asarray(std, dtype=np.float32)
        self.motion_rep = motion_rep
        self.skeleton = skeleton
        self.text_encoder = text_encoder
        self.device = device
        self.fps: int = motion_rep.fps
        self.default_denoising_steps = int(default_denoising_steps)
        self._step_callback = None                 # None => ECHO's native sampling

    # ------------------------------------------------------------ CSF protocol
    def reference_feat(self, text: str, num_frames: int, num_steps: int) -> Tensor:
        """ECHO's unfiltered clean prediction for `text`: normalised ``[T, 38]``.

        Any installed filter is suspended for the call, since references must
        come from the unmodified generator.
        """
        saved, self._step_callback = self._step_callback, None
        try:
            out = self._generate([text], int(num_frames), int(num_steps))
        finally:
            self._step_callback = saved
        return out[0]

    def set_filter_context(self, unsafe, safe, cfg, active_mask=None) -> None:
        """Filter every denoising step against these references (None clears).

        Args:
            unsafe: Unsafe references ``[K, 1, T, 38]`` (or any shape
                `csf_adapter.broadcast_unsafe` accepts).
            safe: Safe reference ``[1, T, 38]``.
            cfg: The `FilterConfig` in force.
            active_mask: The gate's rule selection; None enforces every rule.
        """
        if unsafe is None:
            self._step_callback = None
            return
        from .csf_adapter import make_step_callback  # noqa: PLC0415

        self._step_callback = make_step_callback(unsafe, safe, cfg, active_mask=active_mask)

    def clear_filter_context(self) -> None:
        """Revert to ECHO's native sampling."""
        self._step_callback = None

    # -------------------------------------------------------------- generation
    def _denormalize(self, x: Tensor) -> Tensor:
        mean = torch.from_numpy(self._mean).to(x.device, x.dtype)
        std = torch.from_numpy(self._std).to(x.device, x.dtype)
        return x * std + mean

    def _generate(self, texts: List[str], max_frames: int,
                  num_denoising_steps: Optional[int] = None, **_ignored) -> Tensor:
        """Sample ``[B, max_frames, 38]`` normalised motion, one clip per text.

        The step count is a pipeline attribute in ECHO, so a per-call value is
        written onto the pipeline before sampling.
        """
        if num_denoising_steps is not None:
            self._pipeline.num_inference_steps = int(num_denoising_steps)
        m_lens = torch.as_tensor([int(max_frames)] * len(texts), dtype=torch.long)
        if self._step_callback is not None:
            from .filtered_pipeline import filtered_generate_batch  # noqa: PLC0415

            return filtered_generate_batch(self._pipeline, texts, m_lens,
                                           step_callback=self._step_callback)
        # DiffusePipeline.generate returns one [T_i, 38] clip per caption.
        return torch.stack(self._pipeline.generate(texts, m_lens), dim=0)

    def _decode(self, motion: Tensor) -> dict:
        """Denormalise one ``[T, 38]`` clip and decode it to kimodo's motion dict."""
        return self.motion_rep.inverse(self._denormalize(motion), is_normalized=False,
                                       return_numpy=False)

    def _multiprompt(self, prompts: List[str], num_frames: Union[int, List[int]],
                     num_denoising_steps: Optional[int] = None, *, return_numpy: bool = False,
                     num_transition_frames: int = 5, **_ignored) -> dict:
        """Generate the segments in order and cross-fade them in normalised space.

        Each segment is its own denoising run, filtered with its own context
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
                f"EchoModel batched generation requires equal num_frames across prompts; got "
                f"{lengths}. Use separate calls or multi_prompt for variable lengths."
            )

        motion = self._generate(texts, max(lengths), num_denoising_steps)
        if squeeze:
            output = self._decode(motion[0])
        else:
            outputs = [self._decode(motion[i]) for i in range(len(texts))]
            output = {k: torch.stack([o[k] for o in outputs], dim=0) for k in outputs[0]}
        return _to_numpy_dict(output) if return_numpy else output


def build_echo_bundle(cfg_dict: dict, device: str, text_encoder=None) -> "ModelBundle":
    """Load ECHO from `configs/backbones/echo.yaml` settings into a `ModelBundle`.

    Args:
        cfg_dict: The ECHO backbone config (`ckpt_dir`, `data_root`,
            `mujoco_xml`, `num_denoising_steps`, `diffuser_name`, `fps`).
        device: Torch device string.
        text_encoder: The shared LLM2Vec encoder, or None.
    """
    from kimodo.demo.state import ModelBundle  # noqa: PLC0415
    from kimodo.skeleton.registry import build_skeleton  # noqa: PLC0415

    from .loader import load_echo_pipeline, resolve_repo_path  # noqa: PLC0415
    from .motion_rep import EchoMotionRep  # noqa: PLC0415

    pipeline, mean, std, _opt = load_echo_pipeline(cfg_dict, device)
    fps = int(cfg_dict.get("fps", 50))
    # kimodo's G1Skeleton34, so the app skins it with the G1 meshes.
    skeleton = build_skeleton(34)
    xml = cfg_dict.get("mujoco_xml")
    motion_rep = EchoMotionRep(skeleton, fps=fps,
                               mujoco_xml=str(resolve_repo_path(xml)) if xml else None)
    model = EchoModel(
        pipeline=pipeline, mean=mean, std=std, motion_rep=motion_rep, skeleton=skeleton,
        text_encoder=text_encoder, device=device,
        default_denoising_steps=int(cfg_dict.get("num_denoising_steps", 10)),
    )
    return ModelBundle(model=model, motion_rep=motion_rep, skeleton=skeleton, model_fps=float(fps))
