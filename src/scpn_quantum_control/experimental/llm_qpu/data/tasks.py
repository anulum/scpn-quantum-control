# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — synthetic independent-target memory task
"""Generate deterministic associative text prompts without quantum labels."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..contracts import canonical_bytes

_COLORS = ("amber", "blue", "copper", "jade")
_TEMPLATES = (
    "Reference: {pairs}. Query: Which color belongs to card {query}? Answer:",
    "Study this table: {pairs}. Give the color of card {query}. Answer:",
)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One prefix-only question and its separate classically generated target."""

    sample_id: str
    source_id: str
    group_id: str
    variant_id: int
    prefix_text: str
    target_text: str
    source_kind: str = "synthetic_classical"
    target_origin: str = "classical_generator"

    def __post_init__(self) -> None:
        """Refuse answer suffixes and unknown provenance in generated records."""
        if not all(
            type(value) is str and value and len(value) <= 1024
            for value in (self.sample_id, self.source_id, self.group_id, self.prefix_text)
        ):
            raise ValueError("memory record identity or prefix is invalid")
        if type(self.variant_id) is not int or self.variant_id not in range(len(_TEMPLATES)):
            raise ValueError("unknown memory prompt variant")
        if (
            self.target_text not in _COLORS
            or self.source_kind != "synthetic_classical"
            or self.target_origin != "classical_generator"
        ):
            raise ValueError("memory target must come from classical generator")
        if not self.prefix_text.endswith("Answer:"):
            raise ValueError("memory prefix must end before generated answer")

    def public_wire(self) -> dict[str, object]:
        """Return the answer-free input record handed to an LLM worker."""
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "group_id": self.group_id,
            "variant_id": self.variant_id,
            "prefix_text": self.prefix_text,
            "source_kind": self.source_kind,
            "target_origin": self.target_origin,
        }


def generate_memory_records(*, seed: int, source_count: int = 128) -> tuple[MemoryRecord, ...]:
    """Generate two paraphrases per source from an independent integer seed."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("memory generator seed must be nonnegative signed 64-bit")
    if type(source_count) is not int or not 3 <= source_count <= 4096:
        raise ValueError("memory source count must be between 3 and 4096")
    records: list[MemoryRecord] = []
    for source_index in range(source_count):
        source_id = f"memory-{source_index:04d}"
        color_order = list(_COLORS)
        color_order.sort(
            key=lambda color: hashlib.sha256(
                canonical_bytes({"seed": seed, "source": source_id, "color": color})
            ).digest()
        )
        cards = tuple(f"{source_index:04d}-{slot}" for slot in range(4))
        queried_slot = (
            hashlib.sha256(
                canonical_bytes({"seed": seed, "source": source_id, "selection": "query"})
            ).digest()[0]
            % 4
        )
        pairs = "; ".join(
            f"card {card} = {color}" for card, color in zip(cards, color_order, strict=True)
        )
        for variant_id, template in enumerate(_TEMPLATES):
            records.append(
                MemoryRecord(
                    sample_id=f"{source_id}-v{variant_id}",
                    source_id=source_id,
                    group_id=source_id,
                    variant_id=variant_id,
                    prefix_text=template.format(pairs=pairs, query=cards[queried_slot]),
                    target_text=color_order[queried_slot],
                )
            )
    return tuple(records)
