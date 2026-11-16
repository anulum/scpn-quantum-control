# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental offline IQM compile (isolated IQM cohort only)
"""IQM payloads from the adapter, proved after deserialisation; runs where the IQM SDK is installed."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import replace
from typing import Any

import pytest
from _llm_qpu_dev_chain import _REVISIONS, _planned, _target

from scpn_quantum_control.experimental.llm_qpu.contracts import canonical_bytes
from scpn_quantum_control.experimental.llm_qpu.data.compile_records import _bundle
from scpn_quantum_control.experimental.llm_qpu.data.iqm_compile import (
    compile_iqm_experiment,
    iqm_architecture,
    iqm_backend,
    verify_iqm_experiment,
)
from scpn_quantum_control.experimental.llm_qpu.data.target_compiler import CompiledExperiment

pytest.importorskip("iqm.qiskit_iqm")

_IQM: dict[str, Any] = {
    "provider": "iqm",
    "device_name": "synthetic_iqm_line_9",
    "sdk_name": "iqm-client",
    "sdk_version": "35.0.3",
    "basis_gates": ("cz", "measure", "prx"),
}


def _compiled(**changes: Any) -> tuple[Any, Any, CompiledExperiment]:
    """Return the dev chain, its IQM target and the compiled experiment."""
    planned, target = _planned(), _target(**_IQM)
    return (
        planned,
        target,
        compile_iqm_experiment(
            planned.experiment, planned.plan_set, target, seed=7, revisions=_REVISIONS, **changes
        ),
    )


def _fields(compiled: CompiledExperiment, **changes: Any) -> dict[str, Any]:
    """Return the bundle's fields with changes, ready to be frozen again."""
    names = (
        "target_snapshot_digest",
        "provider",
        "device_name",
        "sdk_lock_digest",
        "cells",
        "layout",
        "logical_gates",
        "compiled_gate_counts",
        "substitutions",
        "compiled_depth",
        "compile_log_digest",
        "provider_options",
    )
    return {name: changes.get(name, getattr(compiled.bundle, name)) for name in names}


def test_iqm_cells_compile_to_native_payloads_that_prove_after_reload() -> None:
    """Native prx/cz/measure payloads, named per cell, reload and prove; the bundle names IQM gates only."""
    planned, target, compiled = _compiled()
    verify_iqm_experiment(compiled, planned.experiment, planned.plan_set, target)
    bundle = compiled.bundle
    assert (
        bundle.provider == "iqm" and bundle.layout == (0, 1, 2, 3) and len(compiled.payloads) == 16
    )
    assert {name for name, _ in bundle.compiled_gate_counts} == {"cz", "measure", "prx"}
    assert bundle.substitutions == (
        ("h", ("prx",)),
        ("rxx", ("cz", "prx")),
        ("ry", ("prx",)),
        ("ryy", ("cz", "prx")),
        ("rz", ("prx",)),
        ("sdg", ("prx",)),
    )
    totals: dict[str, int] = {}
    for ordinal, payload in enumerate(compiled.payloads):
        circuit = json.loads(payload)
        for item in circuit["instructions"]:
            totals[item["name"]] = totals.get(item["name"], 0) + 1
        assert circuit["name"] == f"cell_{ordinal:04d}"
        assert {item["name"] for item in circuit["instructions"]} <= {"cz", "measure", "prx"}
    assert tuple(sorted(totals.items())) == bundle.compiled_gate_counts
    lock = json.loads(compiled.sdk_lock)
    assert [name for name, _ in lock["packages"]] == ["iqm-client", "numpy", "qiskit", "rustworkx"]
    again = compile_iqm_experiment(
        planned.experiment, planned.plan_set, target, seed=7, revisions=_REVISIONS
    )
    assert again.bundle == bundle and again.payloads == compiled.payloads
    placed = compile_iqm_experiment(
        planned.experiment,
        planned.plan_set,
        target,
        seed=7,
        revisions=_REVISIONS,
        layout=(4, 5, 6, 7),
    )
    assert placed.bundle.layout == (4, 5, 6, 7)
    verify_iqm_experiment(placed, planned.experiment, planned.plan_set, target)


def test_the_architecture_names_qubits_undirected_pairs_and_a_compile_only_calibration() -> None:
    """QB1..QBn, one cz locus per coupled pair, prx and measure everywhere; refusals for other targets."""
    target = _target(**_IQM)
    architecture = iqm_architecture(target)
    assert architecture.qubits == [f"QB{i + 1}" for i in range(9)]
    cz = architecture.gates["cz"].implementations["tgss"].loci
    assert len(cz) == 8 and ("QB2", "QB9") in cz and ("QB9", "QB2") not in cz
    assert architecture.gates["prx"].implementations["drag_gaussian"].loci == tuple(
        (f"QB{i + 1}",) for i in range(9)
    )
    digest = hashlib.sha256(canonical_bytes(target.to_wire())).hexdigest()
    assert architecture.calibration_set_id == uuid.uuid5(
        uuid.NAMESPACE_URL, f"scpn-llm-qpu-compile-only:{digest}"
    )
    with pytest.raises(ValueError, match="^an IQM compile needs an IQM target snapshot$"):
        iqm_architecture(_target())
    with pytest.raises(ValueError, match="^an IQM target basis is exactly cz, measure and prx$"):
        iqm_architecture(_target(**{**_IQM, "basis_gates": ("cz", "prx")}))
    backend = iqm_backend(target)
    assert backend.num_qubits == 9 and {"r", "cz", "measure"} <= set(
        backend.target.operation_names
    )
    with pytest.raises(RuntimeError, match="^a compile-only backend never submits$"):
        backend.run(None)
    assert backend._default_options() is not None


def test_routing_off_the_chain_is_refused() -> None:
    """A layout whose qubits are not adjacent forces routing, and the moved reads are refused."""
    planned, target = _planned(), _target(**_IQM)
    with pytest.raises(ValueError, match="^a compiled cell"):
        compile_iqm_experiment(
            planned.experiment,
            planned.plan_set,
            target,
            seed=7,
            revisions=_REVISIONS,
            layout=(0, 2, 4, 6),
        )


def test_verification_refuses_edited_records_swapped_payloads_and_another_target() -> None:
    """Records match their digests, cells their payloads, and payloads their own physics after reload."""
    planned, target, compiled = _compiled()
    payloads = compiled.payloads
    swapped = (payloads[1], payloads[0], *payloads[2:])
    rebound = tuple(
        replace(cell, compiled_payload_digest=hashlib.sha256(payload).hexdigest())
        for cell, payload in zip(compiled.bundle.cells, swapped, strict=True)
    )
    forged = CompiledExperiment(
        _bundle(_fields(compiled, cells=rebound), target, _REVISIONS),
        swapped,
        compiled.compile_log,
        compiled.sdk_lock,
    )
    cases: tuple[tuple[CompiledExperiment, str], ...] = (
        (
            replace(compiled, sdk_lock=b"{}"),
            "^the SDK lock or compile log differs from the bundle$",
        ),
        (
            replace(compiled, compile_log=b"{}"),
            "^the SDK lock or compile log differs from the bundle$",
        ),
        (
            replace(compiled, payloads=payloads[:-1]),
            "^every planned cell needs exactly one compiled payload$",
        ),
        (
            replace(compiled, payloads=swapped),
            "^a compiled cell does not follow its planned cell and payload$",
        ),
        (forged, "^a compiled cell differs from the independent statevector$"),
    )
    for edited, reason in cases:
        with pytest.raises(ValueError, match=reason):
            verify_iqm_experiment(edited, planned.experiment, planned.plan_set, target)
    with pytest.raises(
        ValueError, match="^bundle was compiled against a different target snapshot$"
    ):
        verify_iqm_experiment(
            compiled,
            planned.experiment,
            planned.plan_set,
            _target(**{**_IQM, "device_name": "other"}),
        )
