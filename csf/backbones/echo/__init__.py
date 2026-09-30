# SPDX-License-Identifier: Apache-2.0
"""ECHO (diffusion with DPM-Solver over a 38-d G1 motion feature, 50 fps).

Modules: `model` (the adapter model and bundle builder), `csf_adapter` (the
per-step filter on ECHO's output estimate), `filtered_pipeline` (ECHO's
denoising loop with that hook), `loader` (checkpoint loading), `motion_rep`
(38-d feature to kimodo motion dict), `fk` (MuJoCo forward kinematics of the
G1) and `opt` (ECHO's `opt.txt` parser).
"""
