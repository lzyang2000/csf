# SPDX-License-Identifier: Apache-2.0
"""ECHO's denoising loop with a hook on the output estimate.

ECHO's diffusion model predicts the clean sample (``prediction_type =
"sample"``), so each forward pass yields the output estimate x̂₀ directly, in
the normalised 38-d feature space.  `filtered_generate_batch` reproduces
`DiffusePipeline.generate` / `generate_batch` (noise draw, scheduler
timesteps, cached text embedding, classifier-free-guidance branch) and passes
every prediction through ``step_callback(x0_hat, t, step_idx)`` before the
scheduler update, so the sampler advances from the filtered estimate.  With
no callback it is ECHO's native loop.

It takes any pipeline-like object (``model``, ``scheduler``, ``device``,
``torch_dtype``, ``num_inference_steps``), so it runs on the pipeline the
loader built and needs no ECHO import of its own.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

StepCallback = Callable[[Tensor, Tensor, int], Tensor]


def filtered_generate_batch(pipeline, caption: list[str], m_lens: Tensor,
                            step_callback: Optional[StepCallback] = None) -> Tensor:
    """Sample one batch with ECHO's scheduler, filtering x̂₀ at every step.

    Args:
        pipeline: ECHO's `DiffusePipeline` (or anything with the same fields).
        caption: B text prompts.
        m_lens: ``[B]`` motion lengths in frames.
        step_callback: ``(x0_hat [B, T, 38], t, step_idx) -> x0_hat``, or None.

    Returns:
        The normalised motion ``[B, max(m_lens), 38]``.
    """
    model = pipeline.model
    if hasattr(model, "eval"):
        model.eval()                                   # as DiffusePipeline.generate does
    B = len(caption)
    T = int(m_lens.max())
    sample = torch.randn((B, T, model.input_feats), device=pipeline.device,
                         dtype=pipeline.torch_dtype)

    pipeline.scheduler.set_timesteps(pipeline.num_inference_steps, pipeline.device)
    timesteps = [torch.tensor([t] * B, device=pipeline.device).long()
                 for t in pipeline.scheduler.timesteps]
    enc_text = model.encode_text(caption, pipeline.device)

    for i, t in enumerate(timesteps):
        with torch.no_grad():
            if getattr(model, "cond_mask_prob", 0) > 0:
                predict = model.forward_with_cfg(sample, t, enc_text=enc_text, lengths=m_lens)
            else:
                predict = model(sample, t, enc_text=enc_text, lengths=m_lens)
        if step_callback is not None:
            predict = step_callback(predict, t, i)
        sample = pipeline.scheduler.step(predict, t[0], sample).prev_sample
    return sample
