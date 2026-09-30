# SPDX-License-Identifier: Apache-2.0
"""Runtime handles that attach CSF to a loaded generator.

A handle owns what one filtered generation needs besides the generator: the
rules, the perceived scene entities, the safe-reference choice per prompt
segment and the gate decision per segment.  Two mechanisms install the filter:

* `AdapterFilterHandle` (ECHO, MotionHiFlow, ARDY): the generator exposes a
  per-step hook on its output estimate; the handle builds the unsafe and safe
  references once in that prediction space and installs them per segment.
* `csf.backbones.kimodo.filtered_model.KimodoFilterHandle`: Kimodo constructs
  step-matched references inside its own sampling loop.

Both take the same gate decision (`FilterHandle.decide`).
"""
from __future__ import annotations

import contextlib
import copy
from typing import Callable

import numpy as np
import torch

from .config import FilterConfig, SafeReferencePolicy
from .gate import GateDecision, rule_signals
from .references import select_from_library


def _identity(text: str) -> str:
    return text


def text_sanitizer() -> Callable[[str], str]:
    """Kimodo's prompt normaliser, so texts are encoded exactly as generation does."""
    try:
        from kimodo.sanitize import sanitize_text  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - kimodo is always installed in practice
        return _identity
    return sanitize_text


class FilterHandle:
    """State shared by both filter mechanisms.

    Args:
        cfg: The `FilterConfig` in force (already carrying the backbone's
            settings).
        policy: The safe-reference library.
        encoder: ``encoder(texts) -> (feat[B, 1, d], mask)``, the generator's
            text encoder (LLM2Vec).  None disables gating (every rule is then
            enforced) and selection (the stand is used).
    """

    def __init__(self, cfg: FilterConfig, policy: SafeReferencePolicy, encoder=None) -> None:
        self.cfg = cfg
        self.policy = policy
        self._sanitize = text_sanitizer()
        self._encoder = encoder
        self._cache: dict[str, np.ndarray] = {}
        self.entities: list[str] = []
        self._pins: dict[str, str] = {}
        self.decisions: list[GateDecision] = []

    # ------------------------------------------------------------ text space
    @property
    def has_encoder(self) -> bool:
        return self._encoder is not None

    def encode(self, text: str) -> np.ndarray:
        """1-D embedding of `text` (sanitised the way generation sees it)."""
        key = self._sanitize(text)
        if key not in self._cache:
            feat, _ = self._encoder([key])
            feat = feat[:, None] if feat.dim() == 2 else feat
            self._cache[key] = feat[0, 0].detach().float().cpu().numpy()
        return self._cache[key]

    # ---------------------------------------------------------------- inputs
    def set_entities(self, entities) -> None:
        """The perceived scene entity labels, e.g. ``["a person"]``; [] = none."""
        self.entities = [str(e).strip() for e in (entities or []) if str(e).strip()]

    def pin_safe_references(self, pins: dict[str, str] | None) -> None:
        """Fix the safe reference per segment text, ``{prompt: safe_text}``."""
        self._pins = {}
        for text, ref in (pins or {}).items():
            self._pins[text] = ref
            self._pins[self._sanitize(text)] = ref

    # -------------------------------------------------------------- decision
    def safe_reference_for(self, prompt: str) -> str:
        """The pinned safe reference, else the nearest library action."""
        pinned = self._pins.get(prompt)
        if pinned is not None:
            return pinned
        if not self.has_encoder:
            return self.policy.stand
        return select_from_library(self.encode, prompt, self.policy)

    def decide(self, prompt: str) -> GateDecision:
        """The gate decision for one segment (Eq. 5), recorded in `decisions`."""
        prompt = prompt or ""
        if self.has_encoder:
            mask, reasons = rule_signals(
                self.encode, prompt, self.entities,
                self.cfg.protected_entities, self.cfg.unsafe_texts,
                self.cfg.neutral_entities,
                list(self.cfg.benign_contrasts) + list(self.policy.library),
            )
        else:
            mask, reasons = None, None
        decision = GateDecision(
            mask=mask, reasons=reasons, safe_reference=self.safe_reference_for(prompt),
            prompt=prompt, entities=tuple(self.entities),
        )
        self.decisions.append(decision)
        return decision


class AdapterFilterHandle(FilterHandle):
    """Drives a generator that exposes a per-step output-estimate hook.

    The model implements the adapter protocol of `csf.backbones.base`:
    ``reference_feat(text, num_frames, num_steps)`` returns the generator's
    unfiltered prediction for a text in the space the filter edits, and
    ``set_filter_context`` / ``clear_filter_context`` / ``set_filter_segments``
    install the references.  ``root_dims`` on the model names its explicit
    root channels.
    """

    def __init__(self, model, cfg: FilterConfig, policy: SafeReferencePolicy, encoder=None) -> None:
        cfg = copy.copy(cfg)
        cfg.root_dims = int(getattr(model, "root_dims", 0) or 0)
        super().__init__(cfg, policy, encoder)
        self._model = model
        self._feats: dict = {}

    def _feat(self, text: str, num_frames: int, steps: int) -> torch.Tensor:
        key = (text, int(num_frames), int(steps))
        if key not in self._feats:
            self._feats[key] = self._model.reference_feat(text, int(num_frames), int(steps))
        return self._feats[key]

    def _context_for(self, prompt: str, num_frames: int, steps: int):
        """``(unsafe, safe, cfg, mask)`` for one segment, or Nones when inactive."""
        decision = self.decide(prompt)
        if decision.inactive:
            return None, None, None, None
        unsafe = torch.stack(
            [self._feat(t, num_frames, steps).unsqueeze(0) for t in self.cfg.unsafe_texts], 0
        )
        safe = self._feat(decision.safe_reference, num_frames, steps).unsqueeze(0)
        return unsafe, safe, self.cfg, decision.mask

    def clear(self) -> None:
        self._model.clear_filter_context()
        self._model.set_filter_segments(None)

    @contextlib.contextmanager
    def filtering(self, prompts: list, durations: list, steps: int):
        """Install the filter for one generation of this prompt schedule."""
        self.decisions = []
        contexts = {
            text: self._context_for(text, int(frames), steps)
            for text, frames in zip(prompts, durations)
        }
        unsafe, safe, cfg, mask = contexts[prompts[0]]
        try:
            if unsafe is None:
                self._model.clear_filter_context()
            else:
                self._model.set_filter_context(unsafe, safe, cfg, active_mask=mask)
            self._model.set_filter_segments(contexts if len(prompts) > 1 else None)
            yield self
        finally:
            self.clear()


def attach_adapter_filter(model, cfg: FilterConfig, *, policy: SafeReferencePolicy,
                          encoder=None) -> AdapterFilterHandle:
    """`BackboneSpec.attach_filter` for the adapter families."""
    return AdapterFilterHandle(model, cfg, policy, encoder=encoder)

