# SPDX-License-Identifier: Apache-2.0
"""Example scenes (`configs/examples.yaml`)."""
from __future__ import annotations

from dataclasses import dataclass

from csf.paths import CONFIGS_DIR

EXAMPLES_YAML = CONFIGS_DIR / "examples.yaml"


@dataclass(frozen=True)
class Example:
    """One scene: backbone, prompt schedule, seed and perceived entities."""

    name: str
    prompts: tuple[tuple[str, float], ...]
    seed: int = 0
    backbone: str | None = None
    person: bool = False
    entities: tuple[str, ...] = ()
    runtime: bool = False
    note: str = ""

    @property
    def scene_entities(self) -> list[str]:
        """What perception reports for this scene (the person included)."""
        return (["a person"] if self.person else []) + list(self.entities)


def load_examples(path=EXAMPLES_YAML) -> list[Example]:
    import yaml  # noqa: PLC0415

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return [
        Example(
            name=str(e["name"]),
            prompts=tuple((str(t), float(s)) for t, s in e["prompts"]),
            seed=int(e.get("seed", 0)),
            backbone=e.get("backbone"),
            person=bool(e.get("person", False)),
            entities=tuple(str(x) for x in e.get("entities", []) or []),
            runtime=bool(e.get("runtime", False)),
            note=str(e.get("note", "") or ""),
        )
        for e in data.get("examples", [])
    ]
