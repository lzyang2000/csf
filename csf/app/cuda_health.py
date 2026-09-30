# SPDX-License-Identifier: Apache-2.0
"""Exit on a poisoned CUDA context instead of serving an error loop.

A device-side assert leaves the CUDA context unusable for the rest of the
process, so every later generation would fail the same way.  Exiting lets a
supervisor (systemd, Docker, a shell loop) bring the server back.
"""
from __future__ import annotations

import os
import sys


def check_cuda_health(app: object) -> bool:
    """Probe the context; exit the process once it is found poisoned."""
    import torch  # noqa: PLC0415

    if torch.device(app.device).type == "cpu":
        return True
    try:
        torch.tensor([1.0], device=app.device) + torch.tensor([1.0], device=app.device)
        return True
    except RuntimeError as exc:
        if "device-side assert" in str(exc) or "CUDA error" in str(exc):
            if app.cuda_healthy:
                app.cuda_healthy = False
                print("[csf] FATAL: the CUDA context is corrupted; exiting.")
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
            return False
        raise
