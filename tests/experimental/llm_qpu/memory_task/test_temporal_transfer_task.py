# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — temporal transfer task acceptance
"""Replay transfer prompts independently and exercise split custody."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import validate_task_split
from scpn_quantum_control.experimental.llm_qpu.data.temporal_splits import (
    build_transfer_split,
    transfer_pilot_inputs,
)
from scpn_quantum_control.experimental.llm_qpu.data.temporal_task import build_temporal_task
from scpn_quantum_control.experimental.llm_qpu.data.temporal_transfer import (
    build_transfer_task,
    generate_transfer_records,
)

_BASE = "a" * 40
_REVISION = "b" * 40


def _oracle(prompt: str, variant_id: int) -> str:
    """Interpret visible assignments and copies without the task generator."""
    state: dict[str, str] = {}
    lines = prompt.splitlines()
    if variant_id == 0:
        initial_pattern = r"Initially card (\S+) has colour (\w+)\."
        write_pattern = r"Update \d+: set card (\S+) to (\w+)\."
        copy_pattern = r"Update \d+: copy card (\S+) into card (\S+)\."
        query_pattern = r"Final colour of card (\S+)\?"
    else:
        initial_pattern = r"Start: (\S+) -> (\w+)\."
        write_pattern = r"Step \d+: (\S+) := (\w+)\."
        copy_pattern = r"Step \d+: (\S+) := value\((\S+)\)\."
        query_pattern = r"Final value for (\S+) among"
    copy_count = 0
    for line in lines:
        initial = re.fullmatch(initial_pattern, line)
        write = re.fullmatch(write_pattern, line)
        copy = re.fullmatch(copy_pattern, line)
        if initial is not None:
            state[initial.group(1)] = initial.group(2)
        elif write is not None:
            state[write.group(1)] = write.group(2)
        elif copy is not None:
            if variant_id == 0:
                source, destination = copy.groups()
            else:
                destination, source = copy.groups()
            state[destination] = state[source]
            copy_count += 1
    if copy_count != 3:
        raise ValueError("the rendered task omitted a transfer")
    query = re.search(query_pattern, lines[-1])
    if query is None:
        raise ValueError("transfer query is not readable independently")
    return state[query.group(1)]


def test_transfer_oracle_and_counterfactual_history() -> None:
    records = generate_transfer_records(seed=83, source_count=8)
    assert records == generate_transfer_records(seed=83, source_count=8)
    assert len(records) == 32
    for record in records:
        assert _oracle(record.prefix_text, record.variant_id) == record.target_text
        assert "target_text" not in record.public_wire()
    for source_index in range(8):
        for variant_id in (0, 1):
            pair = [
                record
                for record in records
                if record.source_id == f"transfer-{source_index:04d}"
                and record.variant_id == variant_id
            ]
            assert len(pair) == 2
            assert pair[0].target_text != pair[1].target_text
            left, right = (record.prefix_text.splitlines() for record in pair)
            differences = [
                index
                for index, (first, second) in enumerate(zip(left, right, strict=True))
                if first != second
            ]
            assert differences == [16]
            assert left[-8:] == right[-8:]


def test_transfer_split_and_pilot_are_group_bound() -> None:
    records = generate_transfer_records(seed=83, source_count=8)
    task = build_transfer_task(base_repo_commit=_BASE, implementation_revision=_REVISION)
    split = build_transfer_split(
        task,
        records,
        seed=29,
        generation_seed=83,
        train_count=4,
        dev_count=2,
        test_count=2,
        base_repo_commit=_BASE,
        implementation_revision=_REVISION,
    )
    validate_task_split(task, split)
    assert len(set(split.train_groups + split.dev_groups + split.test_groups)) == 8
    pilot = transfer_pilot_inputs(task, records, split, group_count=2)
    assert len(pilot) == 4
    assert [row["group_id"] for row in pilot] == [
        group for group in split.dev_groups for _ in (0, 1)
    ]
    assert [row["scene_id"] for row in pilot] == [0, 1, 0, 1]
    assert all("target_text" not in row for row in pilot)
    replacement = "blue" if records[0].target_text != "blue" else "amber"
    with pytest.raises(ValueError, match="independent classical generator"):
        build_transfer_split(
            task,
            (replace(records[0], target_text=replacement), *records[1:]),
            seed=29,
            generation_seed=83,
            train_count=4,
            dev_count=2,
            test_count=2,
            base_repo_commit=_BASE,
            implementation_revision=_REVISION,
        )
    changed = replace(
        records[0], prefix_text=records[0].prefix_text.replace("Card ledger", "Changed ledger")
    )
    with pytest.raises(ValueError, match="frozen temporal dataset"):
        transfer_pilot_inputs(task, (changed, *records[1:]), split, group_count=2)


def test_other_task_and_invalid_seed_cannot_authorise_transfer() -> None:
    records = generate_transfer_records(seed=83, source_count=8)
    old_task = build_temporal_task(base_repo_commit=_BASE, implementation_revision=_REVISION)
    with pytest.raises(ValueError, match="exact task"):
        build_transfer_split(
            old_task,
            records,
            seed=29,
            generation_seed=83,
            train_count=4,
            dev_count=2,
            test_count=2,
            base_repo_commit=_BASE,
            implementation_revision=_REVISION,
        )
    with pytest.raises(ValueError, match="seed"):
        generate_transfer_records(seed=True)
    with pytest.raises(ValueError, match="source count"):
        generate_transfer_records(seed=83, source_count=2)
