# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — temporal key-value research task
"""Generate independent-label temporal retrieval scenes with paired histories."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..contracts import ArtifactHeader, TaskSpec, canonical_bytes
from ..contracts.wire import TASK_SCHEMA

_COLORS = ("amber", "blue", "copper", "jade")
_KEY_COUNT = 12
_UPDATE_COUNT = 24
_CRITICAL_UPDATE = 8
_TASK_ID = "synthetic-temporal-key-value-v1"
_OBJECTIVE = "Predict the latest color of a queried key after ordered updates"
_GROUP_DEFINITION = "Counterfactual histories and phrasings of one registry share a source group"


def _choice(seed: int, source: int, role: str, count: int) -> int:
    payload = canonical_bytes({"seed": seed, "source": source, "role": role})
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % count


@dataclass(frozen=True, slots=True)
class TemporalRecord:
    """One answer-free prompt and its separately generated target."""

    sample_id: str
    source_id: str
    group_id: str
    scene_id: int
    variant_id: int
    prefix_text: str
    target_text: str
    source_kind: str = "synthetic_classical"
    target_origin: str = "classical_generator"

    def __post_init__(self) -> None:
        """Refuse invalid provenance, oversized prompts and answer suffixes."""
        if any(
            type(value) is not str or not value or len(value) > 256
            for value in (self.sample_id, self.source_id, self.group_id)
        ):
            raise ValueError("temporal record identity is invalid")
        if type(self.scene_id) is not int or self.scene_id not in (0, 1):
            raise ValueError("temporal scene ID must identify one paired history")
        if type(self.variant_id) is not int or self.variant_id not in (0, 1):
            raise ValueError("temporal variant ID must identify one phrasing")
        if (
            type(self.prefix_text) is not str
            or not self.prefix_text.endswith("Answer:")
            or len(self.prefix_text) > 8192
        ):
            raise ValueError("temporal prefix must be bounded and answer-free")
        if (
            self.target_text not in _COLORS
            or self.source_kind != "synthetic_classical"
            or self.target_origin != "classical_generator"
        ):
            raise ValueError("temporal label must have classical generator provenance")

    def public_wire(self) -> dict[str, object]:
        """Expose the model input and identities without the target label."""
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "group_id": self.group_id,
            "scene_id": self.scene_id,
            "variant_id": self.variant_id,
            "prefix_text": self.prefix_text,
            "source_kind": self.source_kind,
            "target_origin": self.target_origin,
        }


def generate_temporal_records(*, seed: int, source_count: int = 128) -> tuple[TemporalRecord, ...]:
    """Generate paired scenes with the same late context but different targets."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("temporal generator seed must be nonnegative signed 64-bit")
    if type(source_count) is not int or not 3 <= source_count <= 3584:
        raise ValueError("temporal source count must be between 3 and 3584")
    records: list[TemporalRecord] = []
    for source in range(source_count):
        source_id = f"temporal-{source:04d}"
        keys = tuple(f"{source:04d}-{slot:02d}" for slot in range(_KEY_COUNT))
        query_slot = _choice(seed, source, "query", _KEY_COUNT)
        initial = tuple(
            _COLORS[_choice(seed, source, f"initial-{slot}", len(_COLORS))]
            for slot in range(_KEY_COUNT)
        )
        first_color = _COLORS[_choice(seed, source, "first-critical-color", len(_COLORS))]
        alternate_colors = tuple(color for color in _COLORS if color != first_color)
        second_color = alternate_colors[
            _choice(seed, source, "second-critical-color", len(alternate_colors))
        ]
        distractor_slots = tuple(slot for slot in range(_KEY_COUNT) if slot != query_slot)
        shared_updates = tuple(
            (
                distractor_slots[
                    _choice(seed, source, f"update-key-{step}", len(distractor_slots))
                ],
                _COLORS[_choice(seed, source, f"update-color-{step}", len(_COLORS))],
            )
            for step in range(_UPDATE_COUNT)
        )
        for scene_id, target in enumerate((first_color, second_color)):
            updates = list(shared_updates)
            updates[_CRITICAL_UPDATE] = (query_slot, target)
            for variant_id in (0, 1):
                if variant_id == 0:
                    opening = "Card registry. Later updates replace earlier colors."
                    facts = [
                        f"Initially card {key} is {color}."
                        for key, color in zip(keys, initial, strict=True)
                    ]
                    changes = [
                        f"Update {step + 1:02d}: card {keys[slot]} is now {color}."
                        for step, (slot, color) in enumerate(updates)
                    ]
                    question = f"Current color of card {keys[query_slot]}? Choose amber, blue, copper, or jade. Answer:"
                else:
                    opening = "Audit ledger. Apply every change in order; the latest value wins."
                    facts = [
                        f"Start: {key} -> {color}."
                        for key, color in zip(keys, initial, strict=True)
                    ]
                    changes = [
                        f"Change {step + 1:02d}: {keys[slot]} -> {color}."
                        for step, (slot, color) in enumerate(updates)
                    ]
                    question = f"Final value for {keys[query_slot]} among amber, blue, copper, jade? Answer:"
                prefix = "\n".join((opening, *facts, *changes, question))
                records.append(
                    TemporalRecord(
                        sample_id=f"{source_id}-s{scene_id}-v{variant_id}",
                        source_id=source_id,
                        group_id=source_id,
                        scene_id=scene_id,
                        variant_id=variant_id,
                        prefix_text=prefix,
                        target_text=target,
                    )
                )
    return tuple(records)


def build_temporal_task(*, base_repo_commit: str, implementation_revision: str) -> TaskSpec:
    """Freeze the temporal objective separately from the old four-card task."""
    label_digest = hashlib.sha256(canonical_bytes(list(_COLORS))).hexdigest()
    content = {
        "schema": TASK_SCHEMA,
        "object_kind": "task_spec",
        "task_id": _TASK_ID,
        "objective": _OBJECTIVE,
        "source_kind": "synthetic_classical",
        "target_origin": "classical_generator",
        "label_schema_digest": label_digest,
        "causal_cutoff": 4096,
        "primary_metric": "mean_log_loss",
        "group_definition": _GROUP_DEFINITION,
    }
    header = ArtifactHeader(
        object_kind="task_spec",
        content_digest=hashlib.sha256(canonical_bytes(content)).hexdigest(),
        parents=(label_digest,),
        base_repo_commit=base_repo_commit,
        implementation_revision=implementation_revision,
        execution_origin="offline_design",
        data_origin="synthetic_classical",
        claim_scope="design_only",
    )
    return TaskSpec(
        task_id=_TASK_ID,
        objective=_OBJECTIVE,
        source_kind="synthetic_classical",
        target_origin="classical_generator",
        label_schema_digest=label_digest,
        causal_cutoff=4096,
        primary_metric="mean_log_loss",
        group_definition=_GROUP_DEFINITION,
        header=header,
    )
