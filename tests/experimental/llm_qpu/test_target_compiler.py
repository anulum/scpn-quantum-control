# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental offline target compiler
"""Offline compilation against a stored target, with every compiled cell proved twice."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _llm_qpu_dev_chain import _BASIS, _REVISIONS, _both_ways, _planned, _target, _worker
from qiskit import QuantumCircuit

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    canonical_bytes,
    cell_probabilities,
)
from scpn_quantum_control.experimental.llm_qpu.data import compiled_cell_proof
from scpn_quantum_control.experimental.llm_qpu.data.compile_records import _bundle, sdk_lock
from scpn_quantum_control.experimental.llm_qpu.data.compiled_cell_proof import check_compiled_cell
from scpn_quantum_control.experimental.llm_qpu.data.target_compiler import (
    CompiledExperiment,
    chain_layout,
    compile_experiment,
    verify_compiled_experiment,
)
from scpn_quantum_control.hardware.hal_qiskit import _reviewed_qpy_load_circuits


def _bundle_fields(compiled: CompiledExperiment, **changes: Any) -> dict[str, Any]:
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


def test_compiling_proves_every_cell_and_freezes_the_declared_bundle() -> None:
    """Every cell compiles on the first coupled chain, passes both proofs and is bound to its payload."""
    planned, target = _planned(), _target()
    compiled = compile_experiment(
        planned.experiment, planned.plan_set, target, seed=7, revisions=_REVISIONS
    )
    bundle = compiled.bundle
    verify_compiled_experiment(compiled, planned.experiment, planned.plan_set, target)
    assert len(bundle.cells) == len(compiled.payloads) == len(planned.experiment.cells) == 16
    assert bundle.layout == (0, 1, 2, 3)
    assert bundle.logical_gates == ("h", "measure", "rxx", "ry", "ryy", "rz", "sdg")
    single, unitary = ("rz", "sx", "x"), ("cz", "rz", "sx", "x")
    assert bundle.substitutions == (
        ("h", single),
        ("rxx", unitary),
        ("ry", single),
        ("ryy", unitary),
        ("sdg", single),
    )
    assert bundle.provider_options == (("optimization_level", 1), ("seed_transpiler", 7))
    assert {name for name, _ in bundle.compiled_gate_counts} <= set(_BASIS)
    assert bundle.sdk_lock_digest == hashlib.sha256(compiled.sdk_lock).hexdigest()
    log = json.loads(compiled.compile_log)
    assert [cell["payload_sha256"] for cell in log["cells"]] == [
        cell.compiled_payload_digest for cell in bundle.cells
    ]
    assert max(cell["depth"] for cell in log["cells"]) == bundle.compiled_depth
    assert log["options"] == [["optimization_level", 1], ["seed_transpiler", 7]]
    totals: dict[str, int] = {}
    for cell in log["cells"]:
        for name, count in cell["gate_counts"]:
            totals[name] = totals.get(name, 0) + count
    assert tuple(sorted(totals.items())) == bundle.compiled_gate_counts
    loaded = [_reviewed_qpy_load_circuits(payload)[0] for payload in compiled.payloads]
    assert all(circuit.layout is None for circuit in loaded)
    assert [circuit.name for circuit in loaded] == [f"cell_{index:04d}" for index in range(16)]
    binding = canonical_bytes(
        {
            "schema": "scpn.experimental.llm_qpu.parameter_binding.v1",
            "binding_index": 0,
            "parameters": [],
        }
    )
    assert {cell.binding_digest for cell in bundle.cells} == {hashlib.sha256(binding).hexdigest()}
    placed = compile_experiment(
        planned.experiment,
        planned.plan_set,
        target,
        seed=7,
        revisions=_REVISIONS,
        layout=(4, 5, 6, 7),
    )
    assert placed.bundle.layout == (4, 5, 6, 7)
    verify_compiled_experiment(placed, planned.experiment, planned.plan_set, target)


def test_the_standalone_worker_accepts_a_compiled_bundle_and_refuses_its_edit(
    tmp_path: Path,
) -> None:
    """The stdlib worker revalidates the target and the real compiled bundle independently."""
    planned, target = _planned(), _target()
    compiled = compile_experiment(
        planned.experiment, planned.plan_set, target, seed=7, revisions=_REVISIONS
    )
    request: dict[str, object] = {
        "op": "roundtrip_compiled_bundle",
        "target": target.to_wire(),
        "bundle": compiled.bundle.to_wire(),
    }
    assert _worker(request, tmp_path).returncode == 0
    edited = compiled.bundle.to_wire()
    edited["compiled_depth"] = compiled.bundle.compiled_depth + 1
    response = _worker({**request, "bundle": edited}, tmp_path)
    assert response.returncode == 2
    assert "payload digest" in json.loads(response.stdout)["reason"]


def test_a_compiled_bundle_is_identical_across_processes_and_hash_seeds(tmp_path: Path) -> None:
    """Payload bytes carry no transpiler layout metadata, so string hash order cannot change them."""
    script = tmp_path / "compile_once.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import test_target_compiler as t\n"
        "from _llm_qpu_dev_chain import _REVISIONS, _planned\n"
        "p = _planned()\n"
        "c = t.compile_experiment(p.experiment, p.plan_set, t._target(), seed=7, revisions=_REVISIONS)\n"
        "print(c.bundle.payload_digest, c.bundle.compile_log_digest)\n"
    )
    root = Path(__file__).resolve().parents[3]
    digests = set()
    for seed in ("1", "2"):
        run = subprocess.run(
            [sys.executable, str(script)],
            env={
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": f"{root / 'src'}:{root}",
                "PATH": "/usr/bin:/bin",
            },
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        digests.add(run.stdout.strip())
    assert len(digests) == 1


def test_chain_layout_backtracks_skips_isolated_qubits_and_refuses_a_short_graph() -> None:
    """The first simple path in ascending order is chosen; a graph without one is refused."""
    assert chain_layout(_target(), 4) == (0, 1, 2, 3)
    star = _both_ways(((1, 2), (1, 3), (3, 4), (4, 5)))
    assert chain_layout(_target(coupling_edges=star), 4) == (1, 3, 4, 5)
    assert chain_layout(_target(coupling_edges=star), 1) == (0,)
    with pytest.raises(ValueError, match="^the target has no coupled chain of the planned width$"):
        chain_layout(_target(coupling_edges=_both_ways(((0, 1), (2, 3)))), 3)


def test_compiling_refuses_adapter_bases_a_basis_without_measure_and_a_routed_layout() -> None:
    """An IQM prx basis needs its adapter; measure must be native; routing must not move a wire."""
    planned = _planned()
    for basis, reason in (
        (("cz", "measure", "prx"), "^the target's basis needs its provider adapter: prx$"),
        (("cz", "rz", "sx", "x"), "^the target basis must include measure$"),
    ):
        with pytest.raises(ValueError, match=reason):
            compile_experiment(
                planned.experiment,
                planned.plan_set,
                _target(basis_gates=basis),
                seed=7,
                revisions=_REVISIONS,
            )
    with pytest.raises(
        ValueError, match="^a compiled cell does not read each logical wire from its layout"
    ):
        compile_experiment(
            planned.experiment,
            planned.plan_set,
            _target(),
            seed=7,
            revisions=_REVISIONS,
            layout=(0, 2, 4, 6),
        )


def test_the_cell_proof_refuses_foreign_gates_rewired_reads_stray_qubits_and_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each proof clause refuses its own defect on an otherwise valid compiled cell."""
    planned, target = _planned(), _target()
    compiled = compile_experiment(
        planned.experiment, planned.plan_set, target, seed=7, revisions=_REVISIONS
    )
    measurement = planned.experiment.measurements[0]
    plan = planned.plan_set.plans[0]
    zero = (0, 1, 2, 3)
    foreign = QuantumCircuit(9, 4)
    foreign.ecr(0, 1)
    with pytest.raises(ValueError, match="^compiled gates outside the target basis: ecr$"):
        check_compiled_cell(foreign, plan.operations, measurement, target, zero, 1e-9)
    rewired = QuantumCircuit(9, 4)
    for wire, clbit in enumerate((1, 0, 2, 3)):
        rewired.measure(wire, clbit)
    with pytest.raises(ValueError, match="^a compiled cell does not read each logical wire"):
        check_compiled_cell(rewired, plan.operations, measurement, target, zero, 1e-9)
    stray = QuantumCircuit(9, 4)
    stray.x(8)
    for wire in range(4):
        stray.measure(wire, wire)
    with pytest.raises(ValueError, match="^a compiled cell acts on a qubit outside its layout$"):
        check_compiled_cell(stray, plan.operations, measurement, target, zero, 1e-9)
    original = cell_probabilities

    def drifted(operations: Any, plan_measurement: Any) -> tuple[float, ...]:
        """Return the contract's probabilities with 1e-7 moved between the first two outcomes."""
        values = original(operations, plan_measurement)
        return (values[0] + 1e-7, values[1] - 1e-7, *values[2:])

    monkeypatch.setattr(compiled_cell_proof, "cell_probabilities", drifted)
    with pytest.raises(
        ValueError, match="^a compiled cell differs from the independent statevector$"
    ):
        verify_compiled_experiment(compiled, planned.experiment, planned.plan_set, target)
    verify_compiled_experiment(
        compiled, planned.experiment, planned.plan_set, target, tolerance=1e-6
    )


def test_verification_refuses_edited_records_swapped_payloads_and_another_target() -> None:
    """Records must match their digests, cells their payloads, and payloads their own physics."""
    planned, target = _planned(), _target()
    compiled = compile_experiment(
        planned.experiment, planned.plan_set, target, seed=7, revisions=_REVISIONS
    )
    payloads = compiled.payloads
    swapped = (payloads[1], payloads[0], *payloads[2:])
    rebound = tuple(
        replace(cell, compiled_payload_digest=hashlib.sha256(payload).hexdigest())
        for cell, payload in zip(compiled.bundle.cells, swapped, strict=True)
    )
    forged = CompiledExperiment(
        _bundle(_bundle_fields(compiled, cells=rebound), target, _REVISIONS),
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
            verify_compiled_experiment(edited, planned.experiment, planned.plan_set, target)
    with pytest.raises(
        ValueError, match="^bundle was compiled against a different target snapshot$"
    ):
        verify_compiled_experiment(
            compiled, planned.experiment, planned.plan_set, _target(device_name="other")
        )


def test_the_sdk_lock_names_the_interpreter_and_the_compile_packages() -> None:
    """The lock records the Python version and the installed numpy, qiskit and rustworkx versions."""
    lock = json.loads(sdk_lock())
    assert lock["schema"] == "scpn.experimental.llm_qpu.sdk_lock.v1"
    assert lock["python"] == platform.python_version()
    assert [name for name, _ in lock["packages"]] == ["numpy", "qiskit", "rustworkx"]
