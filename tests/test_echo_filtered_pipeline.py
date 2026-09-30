# SPDX-License-Identifier: Apache-2.0
"""ECHO's denoising loop with the x̂₀ hook, on a mock pipeline (no ECHO import, no GPU)."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from csf.backbones.echo.filtered_pipeline import filtered_generate_batch


class MockScheduler:
    """Identity step: the next sample is the (filtered) prediction."""

    def __init__(self, n=3):
        self.timesteps = torch.arange(n)

    def set_timesteps(self, n, device=None):
        pass

    def step(self, predict, t, sample):
        return SimpleNamespace(prev_sample=predict)


class MockModel:
    """Predicts sample + 1, without classifier-free guidance."""

    input_feats = 38
    cond_mask_prob = 0

    def __init__(self):
        self.eval_calls = 0

    def encode_text(self, caption, device):
        return torch.zeros(len(caption), 77, 256)

    def __call__(self, sample, t, enc_text=None, lengths=None):
        return sample + 1.0

    def eval(self):
        self.eval_calls += 1
        return self


def _pipeline(model=None):
    return SimpleNamespace(model=model or MockModel(), scheduler=MockScheduler(), device="cpu",
                           torch_dtype=torch.float32, num_inference_steps=3)


def test_callback_invoked_per_step_and_applied():
    calls = []

    def cb(x0, t, i):
        calls.append(i)
        return x0 * 0.0

    out = filtered_generate_batch(_pipeline(), ["walk"], torch.tensor([4]), step_callback=cb)
    assert calls == [0, 1, 2]
    assert torch.allclose(out, torch.zeros_like(out))


def test_no_callback_runs_the_native_loop():
    out = filtered_generate_batch(_pipeline(), ["walk"], torch.tensor([4]))
    assert out.shape == (1, 4, 38)


def test_callback_sees_batch_shape():
    shapes = []

    def cb(x0, t, i):
        shapes.append(tuple(x0.shape))
        return x0

    out = filtered_generate_batch(_pipeline(), ["walk", "run"], torch.tensor([5, 5]), step_callback=cb)
    assert shapes == [(2, 5, 38)] * 3
    assert out.shape == (2, 5, 38)


def test_classifier_free_guidance_branch_is_used():
    class CfgModel(MockModel):
        cond_mask_prob = 0.1

        def forward_with_cfg(self, sample, t, enc_text=None, lengths=None):
            return torch.full_like(sample, 7.0)

    out = filtered_generate_batch(_pipeline(CfgModel()), ["walk"], torch.tensor([3]))
    assert torch.allclose(out, torch.full_like(out, 7.0))


def test_model_is_put_in_eval_mode():
    model = MockModel()
    filtered_generate_batch(_pipeline(model), ["walk"], torch.tensor([3]))
    assert model.eval_calls == 1
