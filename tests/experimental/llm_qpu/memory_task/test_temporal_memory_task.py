# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — temporal memory task acceptance
"""Check generated temporal prompts with an independent text-level oracle."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    SplitManifest,
    TaskSpec,
    canonical_bytes,
    validate_task_split,
)
from scpn_quantum_control.experimental.llm_qpu.data.tasks import build_memory_task
from scpn_quantum_control.experimental.llm_qpu.data.temporal_splits import (
    build_temporal_split,
    temporal_pilot_inputs,
)
from scpn_quantum_control.experimental.llm_qpu.data.temporal_task import (
    TemporalRecord,
    build_temporal_task,
    generate_temporal_records,
)

_BASE = "a" * 40
_REVISION = "b" * 40


def _oracle(prompt: str, variant_id: int) -> str:
    """Replay the natural-language ledger without using generator internals."""
    state: dict[str, str] = {}
    lines = prompt.splitlines()
    if variant_id == 0:
        initial_pattern = r"Initially card (\S+) is (\w+)\."
        update_pattern = r"Update \d+: card (\S+) is now (\w+)\."
        query_pattern = r"Current color of card (\S+)\?"
    else:
        initial_pattern = r"Start: (\S+) -> (\w+)\."
        update_pattern = r"Change \d+: (\S+) -> (\w+)\."
        query_pattern = r"Final value for (\S+) among"
    for line in lines:
        initial = re.fullmatch(initial_pattern, line)
        update = re.fullmatch(update_pattern, line)
        match = initial or update
        if match is not None:
            state[match.group(1)] = match.group(2)
    query = re.search(query_pattern, lines[-1])
    if query is None:
        raise ValueError("query is not readable independently")
    return state[query.group(1)]


def _split(records: tuple[TemporalRecord, ...]) -> tuple[TaskSpec, SplitManifest]:
    """Return the temporal task and its six-group split of the records."""
    task = build_temporal_task(base_repo_commit=_BASE, implementation_revision=_REVISION)
    split = build_temporal_split(
        task,
        records,
        seed=31,
        generation_seed=73,
        train_count=2,
        dev_count=2,
        test_count=2,
        base_repo_commit=_BASE,
        implementation_revision=_REVISION,
    )
    validate_task_split(task, split)
    return task, split


def test_independent_oracle_and_counterfactual_suffix() -> None:
    """An independent oracle reproduces every label, and each paired history differs in one early line only."""
    records = generate_temporal_records(seed=73, source_count=6)
    assert len(records) == 24
    assert records == generate_temporal_records(seed=73, source_count=6)
    for record in records:
        assert _oracle(record.prefix_text, record.variant_id) == record.target_text
        assert "target_text" not in record.public_wire()
    for source_index in range(6):
        for variant_id in (0, 1):
            pair = [
                record
                for record in records
                if record.source_id == f"temporal-{source_index:04d}"
                and record.variant_id == variant_id
            ]
            assert len(pair) == 2
            assert pair[0].target_text != pair[1].target_text
            left, right = (record.prefix_text.splitlines() for record in pair)
            assert len(left) == len(right)
            differences = [
                index
                for index, rows in enumerate(zip(left, right, strict=True))
                if rows[0] != rows[1]
            ]
            assert len(differences) == 1
            assert differences[0] < len(left) - 8
            assert left[-8:] == right[-8:]


def test_group_safe_split_and_answer_free_paired_pilot() -> None:
    """The split covers six groups, and the pilot returns answer-free paired dev histories in group order."""
    records = generate_temporal_records(seed=73, source_count=6)
    task, split = _split(records)
    assert len(set(split.train_groups + split.dev_groups + split.test_groups)) == 6
    pilot = temporal_pilot_inputs(task, records, split, group_count=2)
    assert len(pilot) == 4
    assert all("target_text" not in row for row in pilot)
    assert [row["group_id"] for row in pilot] == [
        group for group in split.dev_groups for _ in (0, 1)
    ]
    assert [row["scene_id"] for row in pilot] == [0, 1, 0, 1]


def test_label_and_source_leakage_refuse_split() -> None:
    """A changed label, a source in two groups and a prompt copied across groups each refuse the split."""
    records = generate_temporal_records(seed=73, source_count=6)
    replacement = "blue" if records[0].target_text != "blue" else "amber"
    with pytest.raises(ValueError, match="differ from the independent"):
        _split((replace(records[0], target_text=replacement), *records[1:]))
    with pytest.raises(ValueError, match="multiple groups"):
        _split((replace(records[0], group_id="foreign"), *records[1:]))
    with pytest.raises(ValueError, match="duplicated across"):
        _split(
            (
                *records,
                replace(
                    records[0],
                    sample_id="foreign-sample",
                    source_id="foreign-source",
                    group_id="foreign",
                ),
            )
        )


def test_temporal_record_refuses_qpu_label_and_answer_suffix() -> None:
    """A record refuses a QPU-made label, an answer inside its prefix, a third scene and a third phrasing."""
    record = generate_temporal_records(seed=73, source_count=3)[0]
    with pytest.raises(ValueError, match="classical generator"):
        replace(record, target_origin="qpu_output")
    with pytest.raises(ValueError, match="answer-free"):
        replace(record, prefix_text=f"{record.prefix_text} {record.target_text}")
    with pytest.raises(ValueError, match="paired history"):
        replace(record, scene_id=2)
    with pytest.raises(ValueError, match="phrasing"):
        replace(record, variant_id=2)


def test_pilot_refuses_rebound_prompt_and_non_dev_selection() -> None:
    """The pilot refuses an edited prompt and more groups than the dev split holds."""
    records = generate_temporal_records(seed=73, source_count=6)
    task, split = _split(records)
    changed = replace(
        records[0], prefix_text=records[0].prefix_text.replace("Card registry", "New registry")
    )
    with pytest.raises(ValueError, match="frozen temporal dataset"):
        temporal_pilot_inputs(task, (changed, *records[1:]), split, group_count=2)
    with pytest.raises(ValueError, match="exceeds dev groups"):
        temporal_pilot_inputs(task, records, split, group_count=3)


def test_generator_refuses_ambiguous_seed_and_too_few_groups() -> None:
    """The generator refuses a bool seed and fewer than three sources."""
    with pytest.raises(ValueError, match="seed"):
        generate_temporal_records(seed=True)
    with pytest.raises(ValueError, match="source count"):
        generate_temporal_records(seed=73, source_count=2)


def test_old_four_card_task_cannot_authorize_temporal_split() -> None:
    """The earlier four-card task cannot authorise a temporal split."""
    records = generate_temporal_records(seed=73, source_count=6)
    old_task = build_memory_task(base_repo_commit=_BASE, implementation_revision=_REVISION)
    with pytest.raises(ValueError, match="exact task"):
        build_temporal_split(
            old_task,
            records,
            seed=31,
            generation_seed=73,
            train_count=2,
            dev_count=2,
            test_count=2,
            base_repo_commit=_BASE,
            implementation_revision=_REVISION,
        )


def test_streamed_dataset_identity_above_canonical_wire_limit() -> None:
    """A 512-source dataset above the canonical wire limit still splits and pilots through its streamed identity."""
    records = generate_temporal_records(seed=73, source_count=512)
    task = build_temporal_task(base_repo_commit=_BASE, implementation_revision=_REVISION)
    split = build_temporal_split(
        task,
        records,
        seed=31,
        generation_seed=73,
        train_count=256,
        dev_count=128,
        test_count=128,
        base_repo_commit=_BASE,
        implementation_revision=_REVISION,
    )
    validate_task_split(task, split)
    assert len(split.train_groups + split.dev_groups + split.test_groups) == 512
    assert temporal_pilot_inputs(task, records, split, group_count=8)


def _build(records: Any, **changes: Any) -> SplitManifest:
    """Build the six-group temporal split of the records, with changed arguments."""
    arguments: dict[str, Any] = {
        "seed": 31,
        "generation_seed": 73,
        "train_count": 2,
        "dev_count": 2,
        "test_count": 2,
        "base_repo_commit": _BASE,
        "implementation_revision": _REVISION,
    }
    arguments.update(changes)
    task = build_temporal_task(base_repo_commit=_BASE, implementation_revision=_REVISION)
    return build_temporal_split(task, records, **arguments)


def test_temporal_split_refuses_bad_seeds_counts_and_inventories() -> None:
    """The builder refuses bad seeds and counts, and inventories that are empty or not unique."""
    records = generate_temporal_records(seed=73, source_count=6)
    flag: Any = True
    for changes, message in (
        ({"seed": -1}, "split seed must be a nonnegative signed 64-bit integer"),
        ({"seed": 2**63}, "split seed must be a nonnegative signed 64-bit integer"),
        ({"seed": flag}, "split seed must be a nonnegative signed 64-bit integer"),
        ({"dev_count": 0}, "every split needs a positive source count"),
        ({"test_count": flag}, "every split needs a positive source count"),
        ({"test_count": 1}, "split counts must cover every source group exactly"),
    ):
        with pytest.raises(ValueError, match=f"^{message}$"):
            _build(records, **changes)
    for inventory, message in (
        ((), "temporal inventory out of bounds"),
        ((records[0].public_wire(), *records[1:]), "temporal inventory contains an unvalidated"),
        ((records[0], records[0], *records[1:]), "duplicate temporal sample ID"),
    ):
        with pytest.raises(ValueError, match=f"^{message}"):
            _build(inventory)


def test_temporal_pilot_refuses_unsynthetic_splits_and_missing_pairs() -> None:
    """The pilot needs a synthetic frozen split and both scenes of every selected dev group."""
    records = generate_temporal_records(seed=73, source_count=6)
    task, split = _split(records)
    loose: Any = split.to_wire()
    owner = replace(split, header=replace(split.header, data_origin="owner_dataset"))
    for candidate in (loose, owner):
        with pytest.raises(ValueError, match="^temporal pilot requires a synthetic group split$"):
            temporal_pilot_inputs(task, records, candidate, group_count=2)
    foreign_dev = tuple(sorted(("aaa-foreign-group", *split.dev_groups[1:])))
    content = {name: value for name, value in split.to_wire().items() if name != "header"} | {
        "dev_groups": list(foreign_dev)
    }
    header = replace(
        split.header, content_digest=hashlib.sha256(canonical_bytes(content)).hexdigest()
    )
    foreign = replace(split, dev_groups=foreign_dev, header=header)
    with pytest.raises(ValueError, match="^paired temporal history is missing$"):
        temporal_pilot_inputs(task, records, foreign, group_count=2)
