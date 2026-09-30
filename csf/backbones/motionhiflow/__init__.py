# SPDX-License-Identifier: Apache-2.0
"""MotionHiFlow (flow matching in a VAE latent, HumanML3D SMPL body, 20 fps).

Modules: `model` (the adapter model and bundle builder), `csf_adapter` (the
per-step filter on the clean latent), `decoded_margin` (the optional
decoder-aware path, Eqs. 7-8), `filtered_sampler` (MotionHiFlow's pyramid
Euler loop with an output-estimate hook), `loader` (checkpoint loading) and
`motion_rep` (263-d HumanML3D features to kimodo motion dict).
"""
