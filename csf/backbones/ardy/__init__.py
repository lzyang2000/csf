# SPDX-License-Identifier: Apache-2.0
"""ARDY-G1 (windowed autoregressive diffusion over hybrid root + FSQ tokens).

Modules: `model` (the adapter model and bundle builder), `filter_proxy` (the
denoiser wrapper that filters each window's clean token prediction, Eq. 9).
"""
