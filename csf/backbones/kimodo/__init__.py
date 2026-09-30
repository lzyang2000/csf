# SPDX-License-Identifier: Apache-2.0
"""Kimodo-G1 (DDIM diffusion over a 417-d motion feature).

Modules: `model` (bundle builder), `filtered_model` (the filter handle, which
wraps Kimodo's per-segment sampler), `filtered_generate` (the step-matched
filtered sampling loop).
"""
