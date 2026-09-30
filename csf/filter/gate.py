# SPDX-License-Identifier: Apache-2.0
"""The semantic context gate (Sec. III-D, Eq. 5).

The margin identifies a behaviour; the gate decides which rules the QP
enforces.  Both signals compare texts with the generator's own text encoder by
nearest neighbour, so no fixed vocabulary is needed:

* perception: rule i fires when a detected entity label is closest to the
  entity rule i protects (rather than to a neutral entity);
* prompt: the context-sensitive rules fire when the command is closest to an
  unsafe description (rather than to a benign contrast).

    g_i = g_i^perception  OR  g_i^prompt,        A = { i : g_i = 1 }.

Everything takes an ``encode(text) -> 1-D array`` callable, so it is testable
with a stub encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

_EPS = 1e-8

Encode = Callable[[str], np.ndarray]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + _EPS)


def rule_signals(encode: Encode, prompt: str, entities, protected, unsafe_texts,
                 neutral_entities, benign_contrasts):
    """Evaluate Eq. 5.

    Args:
        encode: Text encoder, ``encode(text) -> 1-D vector``.
        prompt: The command being generated.
        entities: Perceived scene entity labels, e.g. ``["a person"]``.
        protected: The protected entity of each rule (length K).
        unsafe_texts: The unsafe description of each rule (length K).
        neutral_entities: Benign homes for scene entities.
        benign_contrasts: Benign homes for commands.

    Returns:
        ``(mask, reasons)``: mask[k] is g_k, and reasons[k] lists the signals
        that fired for rule k ("perception" and/or "prompt").
    """
    K = len(unsafe_texts)
    perception = [False] * K
    protected_set = [c for c in dict.fromkeys(protected) if c]
    if entities and protected_set:
        homes = protected_set + list(neutral_entities)
        home_emb = {c: _unit(encode(c)) for c in homes}
        for entity in entities:
            e = _unit(encode(entity))
            nearest = max(homes, key=lambda c: float(e @ home_emb[c]))
            for k in range(K):
                if protected[k] == nearest:
                    perception[k] = True

    prompt_fires = False
    if prompt and prompt.strip() and K and benign_contrasts:
        candidates = list(unsafe_texts) + list(benign_contrasts)
        p = _unit(encode(prompt))
        scores = [float(p @ _unit(encode(t))) for t in candidates]
        prompt_fires = int(np.argmax(scores)) < K

    reasons = []
    for k in range(K):
        fired = []
        if perception[k]:
            fired.append("perception")
        if prompt_fires and protected[k]:
            fired.append("prompt")
        reasons.append(fired)
    return [bool(r) for r in reasons], reasons


@dataclass(frozen=True)
class GateDecision:
    """The gate's decision for one prompt segment.

    Attributes:
        mask: g_k per rule, or None when no encoder was available (every rule
            is then enforced: fail safe).
        reasons: The firing signals per rule (None with `mask`).
        safe_reference: The text of the safe reference this segment tracks.
        prompt: The segment's command.
        entities: The perceived entities the decision used.
    """

    mask: Optional[list]
    reasons: Optional[list]
    safe_reference: str
    prompt: str
    entities: tuple

    @property
    def inactive(self) -> bool:
        """True when the gate ran and no rule fired: the segment passes through."""
        return self.mask is not None and not any(self.mask)

    def active_rules(self, rules) -> list:
        """The rules in A (every rule when the gate was unavailable)."""
        if self.mask is None:
            return list(rules)
        return [rule for rule, on in zip(rules, self.mask) if on]

    def as_record(self, rules) -> dict:
        """A plain-data record of this decision, for logs and the GUI."""
        reasons = self.reasons or [["unavailable"]] * len(rules)
        return {
            "prompt": self.prompt,
            "entities": list(self.entities),
            "safe_reference": self.safe_reference,
            "rules": [
                {"name": rule.name, "protects": rule.protects,
                 "active": bool(self.mask is None or self.mask[k]),
                 "signals": list(reasons[k])}
                for k, rule in enumerate(rules)
            ],
        }
