# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — temporal transfer research task
"""Generate paired histories whose answer crosses three ordered copies."""

from __future__ import annotations

import hashlib

from ..contracts import ArtifactHeader, TaskSpec, canonical_bytes
from ..contracts.wire import TASK_SCHEMA
from .temporal_task import TemporalRecord

_COLOURS = ("amber", "blue", "copper", "jade")
_KEY_COUNT = 12
_UPDATE_COUNT = 24
_CRITICAL_STEP = 3
_COPY_STEPS = (8, 13, 18)
_PRE_COPY_WRITES = {6: 1, 11: 2, 16: 3}
_TASK_ID = "synthetic-temporal-key-value-v2"
_OBJECTIVE = "Predict a key's final colour after ordered writes and value-at-time copies"
_GROUP_DEFINITION = "Counterfactual transfer histories and phrasings share one source group"


def _choice(seed: int, source: int, role: str, count: int) -> int:
    payload = canonical_bytes({"task": _TASK_ID, "seed": seed, "source": source, "role": role})
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % count


def generate_transfer_records(*, seed: int, source_count: int = 128) -> tuple[TemporalRecord, ...]:
    """Build two different targets from one early write and three later copies."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("transfer generator seed must be nonnegative signed 64-bit")
    if type(source_count) is not int or not 3 <= source_count <= 3584:
        raise ValueError("transfer source count must be between 3 and 3584")
    records: list[TemporalRecord] = []
    for source in range(source_count):
        source_id = f"transfer-{source:04d}"
        keys = tuple(f"{source:04d}-{slot:02d}" for slot in range(_KEY_COUNT))
        ranked_slots = sorted(
            range(_KEY_COUNT),
            key=lambda slot: _choice(seed, source, f"path-rank-{slot}", 2**32),
        )
        path = tuple(ranked_slots[:4])
        distractors = tuple(slot for slot in range(_KEY_COUNT) if slot not in path)
        initial = tuple(
            _COLOURS[_choice(seed, source, f"initial-{slot}", len(_COLOURS))]
            for slot in range(_KEY_COUNT)
        )
        first_colour = _COLOURS[_choice(seed, source, "critical-first", len(_COLOURS))]
        alternatives = tuple(colour for colour in _COLOURS if colour != first_colour)
        second_colour = alternatives[_choice(seed, source, "critical-second", len(alternatives))]
        shared: list[tuple[str, int, int | str]] = []
        for step in range(_UPDATE_COUNT):
            if step in _COPY_STEPS:
                hop = _COPY_STEPS.index(step)
                shared.append(("copy", path[hop + 1], path[hop]))
            elif step in _PRE_COPY_WRITES:
                slot = path[_PRE_COPY_WRITES[step]]
                colour = _COLOURS[_choice(seed, source, f"pre-copy-{step}", len(_COLOURS))]
                shared.append(("write", slot, colour))
            else:
                slot = distractors[_choice(seed, source, f"distractor-{step}", len(distractors))]
                colour = _COLOURS[_choice(seed, source, f"update-{step}", len(_COLOURS))]
                shared.append(("write", slot, colour))
        for scene_id, target in enumerate((first_colour, second_colour)):
            updates = list(shared)
            updates[_CRITICAL_STEP] = ("write", path[0], target)
            state = list(initial)
            for operation, slot, value in updates:
                if operation == "write" and type(value) is str:
                    state[slot] = value
                elif operation == "copy" and type(value) is int:
                    state[slot] = state[value]
                else:
                    raise ValueError("transfer update is not a write or copy")
            final_colour = state[path[3]]
            for variant_id in (0, 1):
                if variant_id == 0:
                    opening = (
                        "Card ledger. Process updates in order. A copy takes the source's "
                        "current colour and overwrites the destination."
                    )
                    facts = [
                        f"Initially card {key} has colour {colour}."
                        for key, colour in zip(keys, initial, strict=True)
                    ]
                    changes = [
                        (
                            f"Update {step + 1:02d}: set card {keys[slot]} to {value}."
                            if operation == "write"
                            else f"Update {step + 1:02d}: copy card {keys[int(value)]} into card {keys[slot]}."
                        )
                        for step, (operation, slot, value) in enumerate(updates)
                    ]
                    question = (
                        f"Final colour of card {keys[path[3]]}? "
                        "Choose amber, blue, copper, or jade. Answer:"
                    )
                else:
                    opening = (
                        "Ordered register. Assignment overwrites. value(source) means "
                        "the source's value at this step, not its initial value."
                    )
                    facts = [
                        f"Start: {key} -> {colour}."
                        for key, colour in zip(keys, initial, strict=True)
                    ]
                    changes = [
                        (
                            f"Step {step + 1:02d}: {keys[slot]} := {value}."
                            if operation == "write"
                            else f"Step {step + 1:02d}: {keys[slot]} := value({keys[int(value)]})."
                        )
                        for step, (operation, slot, value) in enumerate(updates)
                    ]
                    question = (
                        f"Final value for {keys[path[3]]} among amber, blue, copper, jade? Answer:"
                    )
                records.append(
                    TemporalRecord(
                        sample_id=f"{source_id}-s{scene_id}-v{variant_id}",
                        source_id=source_id,
                        group_id=source_id,
                        scene_id=scene_id,
                        variant_id=variant_id,
                        prefix_text="\n".join((opening, *facts, *changes, question)),
                        target_text=final_colour,
                    )
                )
    return tuple(records)


def build_transfer_task(*, base_repo_commit: str, implementation_revision: str) -> TaskSpec:
    """Freeze copy-at-time semantics under a new task identity."""
    label_digest = hashlib.sha256(canonical_bytes(list(_COLOURS))).hexdigest()
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
