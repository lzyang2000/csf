# SPDX-License-Identifier: Apache-2.0
"""The declared safety specification and the filter's knobs.

`rules.yaml` holds the K natural-language rules (Sec. III-B): each pairs an
unsafe behaviour with the entity it protects.  The gate section lists the
benign homes the context gate compares against (Sec. III-D), and the filter
section holds the QP constants shared by every backbone (Eq. 3, Prop. 1).

`safe_references.yaml` holds the library of neutral actions the safe reference
x_safe is chosen from, with a neutral stand as the fallback.

Per-backbone settings (tracking strength gamma, which leading channels are
explicit root motion, Table I) are not read from YAML; they come from the
backbone registry and are written onto a copy of this config at generation
time.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from csf.paths import CONFIGS_DIR

RULES_YAML = "rules.yaml"
SAFE_REFERENCES_YAML = "safe_references.yaml"


@dataclass(frozen=True)
class Rule:
    """One declared rule: an unsafe behaviour and the entity it protects."""

    name: str
    unsafe: str
    protects: str


@dataclass
class FilterConfig:
    """Everything the filter reads while it runs.

    Attributes:
        rules: The K declared rules.
        neutral_entities: Benign homes a perceived scene entity can land on
            instead of a protected entity (perception signal, Sec. III-D).
        benign_contrasts: Benign homes a command can land on instead of an
            unsafe description (prompt signal, Sec. III-D).  The safe-reference
            library is appended to these at runtime.
        rho: Barrier contraction rate rho_t in Eq. 3 (1 imposes the safe set in
            one step).
        gamma: Safe-reference tracking weight gamma in Eq. 3 (0 is the
            minimum-norm barrier-only correction).
        ridge: Ridge epsilon of the dual (Prop. 1).
        max_iters: Coordinate-ascent sweeps of the dual solve.
        root_dims: Number of LEADING channels that are explicit root motion.
            They are excluded from the margin and from the correction (the
            projector M of Eq. 6).  0 filters every channel.
        stand_speed_threshold: Root speed (m/s) below which a clip, or a frame
            of a clip, counts as in place, so its safe reference is the stand.
        decoder_aware: Evaluate the margin after the pretrained decoder
            (Eqs. 7-8) instead of in the latent.  Implemented for MotionHiFlow;
            off by default, as in the benchmark.
        decoder_iters: Iteration budget N of Eq. 8.
        decoder_step: Step size eta of Eq. 8.
        pin_dims: Channels pinned to the unfiltered root path during sampling
            (filled at runtime by the Kimodo backbone).
    """

    rules: list[Rule] = field(default_factory=list)
    neutral_entities: list[str] = field(default_factory=list)
    benign_contrasts: list[str] = field(default_factory=list)
    rho: float = 1.0
    gamma: float = 0.0
    ridge: float = 1e-2
    max_iters: int = 20
    root_dims: int = 0
    stand_speed_threshold: float = 0.3
    decoder_aware: bool = False
    decoder_iters: int = 8
    decoder_step: float = 0.5
    pin_dims: list[int] = field(default_factory=list)

    @property
    def unsafe_texts(self) -> list[str]:
        """The unsafe-behaviour descriptions, in rule order."""
        return [rule.unsafe for rule in self.rules]

    @property
    def protected_entities(self) -> list[str]:
        """The protected entity of each rule, in rule order."""
        return [rule.protects for rule in self.rules]

    def with_backbone(self, **overrides) -> "FilterConfig":
        """A copy with per-backbone settings applied (the shared lists stay shared)."""
        cfg = copy.copy(self)
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"FilterConfig has no field {key!r}")
            setattr(cfg, key, value)
        return cfg


@dataclass(frozen=True)
class SafeReferencePolicy:
    """Where the safe reference x_safe comes from (Sec. III-B).

    Attributes:
        library: Neutral actions the text encoder selects from.
        stand: The neutral stand used as the fallback.
        min_similarity: Below this cosine to the nearest library action, the
            stand is used instead.
        locomotion: The locomotion subset of the library, used when the
            unfiltered clip actually travels.
        locomotion_min_similarity: Similarity floor for that locomotion pick.
    """

    library: tuple[str, ...]
    stand: str
    min_similarity: float = 0.0
    locomotion: tuple[str, ...] = ()
    locomotion_min_similarity: float = 0.65


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_filter_config(path: Path | None = None) -> FilterConfig:
    """Read `configs/rules.yaml` (or `path`) into a `FilterConfig`."""
    data = _read_yaml(path or CONFIGS_DIR / RULES_YAML)
    rules = [
        Rule(name=str(r["name"]), unsafe=str(r["unsafe"]), protects=str(r["protects"]))
        for r in data.get("rules", [])
    ]
    if not rules:
        raise ValueError("rules.yaml declares no rules")
    gate = data.get("gate", {}) or {}
    qp = data.get("filter", {}) or {}
    return FilterConfig(
        rules=rules,
        neutral_entities=[str(x) for x in gate.get("neutral_entities", [])],
        benign_contrasts=[str(x) for x in gate.get("benign_contrasts", [])],
        rho=float(qp.get("rho", 1.0)),
        ridge=float(qp.get("ridge", 1e-2)),
        max_iters=int(qp.get("max_iters", 20)),
    )


def load_safe_reference_policy(path: Path | None = None) -> tuple[SafeReferencePolicy, float]:
    """Read `configs/safe_references.yaml`.

    Returns:
        The policy, and the stand-speed threshold (m/s) it declares.
    """
    data = _read_yaml(path or CONFIGS_DIR / SAFE_REFERENCES_YAML)
    policy = SafeReferencePolicy(
        library=tuple(str(x).strip() for x in data["library"]),
        stand=str(data["stand"]).strip(),
        min_similarity=float(data.get("min_similarity", 0.0)),
        locomotion=tuple(str(x).strip() for x in data.get("locomotion", [])),
        locomotion_min_similarity=float(data.get("locomotion_min_similarity", 0.65)),
    )
    return policy, float(data.get("stand_speed_threshold", 0.3))
