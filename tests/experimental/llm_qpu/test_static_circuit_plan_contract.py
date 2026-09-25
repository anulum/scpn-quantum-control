# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — LLM-QPU static circuit plan contract
"""Compare the public digital gate plan with an independent dense XY evolution."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    ArrayDescriptor,
    ArtifactHeader,
    CompressedLatentBatch,
    ReservoirSpec,
    StaticCircuitPlan,
    build_static_circuit_plan,
    canonical_bytes,
    decode_contract,
    validate_static_circuit_plan,
)


def _compressed() -> tuple[CompressedLatentBatch, bytes]:
    angles = struct.pack("<4f", 0.1, -0.2, 0.3, -0.4)
    tensor = ArrayDescriptor("<f4", (1, 4), len(angles), hashlib.sha256(angles).hexdigest())
    latent_digest = "a" * 64
    compressor_digest = "b" * 64
    fields: dict[str, object] = {
        "schema": "scpn.experimental.llm_qpu.compressed_latent_batch.v1",
        "object_kind": "compressed_latent_batch",
        "latent_digest": latent_digest,
        "compressor_digest": compressor_digest,
        "split_name": "train",
        "layout": "contextual",
        "sample_ids": ["source-scene-1"],
        "source_ids": ["source-scene-1"],
        "group_ids": ["scene-1"],
        "lengths": [1],
        "mask": [[True]],
        "token_positions": [[3]],
        "answer_start_positions": [[4]],
        "angle_unit": "radian",
        "comparison_tolerance_hex": (1e-6).hex(),
        "tensor": tensor.to_wire(),
    }
    header = ArtifactHeader(
        object_kind="compressed_latent_batch",
        content_digest=hashlib.sha256(canonical_bytes(fields)).hexdigest(),
        parents=tuple(sorted((latent_digest, compressor_digest, tensor.sha256))),
        base_repo_commit="c" * 40,
        implementation_revision="d" * 40,
        execution_origin="offline_design",
        data_origin="synthetic_classical",
        claim_scope="design_only",
    )
    return (
        CompressedLatentBatch(
            latent_digest=latent_digest,
            compressor_digest=compressor_digest,
            split_name="train",
            layout="contextual",
            sample_ids=("source-scene-1",),
            source_ids=("source-scene-1",),
            group_ids=("scene-1",),
            lengths=(1,),
            mask=((True,),),
            token_positions=((3,),),
            answer_start_positions=((4,),),
            angle_unit="radian",
            comparison_tolerance_hex=(1e-6).hex(),
            tensor=tensor,
            header=header,
        ),
        angles,
    )


def _spec() -> ReservoirSpec:
    return ReservoirSpec(
        kernel_id="xy_static_digital_v1",
        n_qubits=4,
        edges=((0, 1), (1, 2), (2, 3)),
        couplings_hex=((0.5).hex(), (0.0).hex(), (0.0).hex()),
        fields_hex=((0.0).hex(),) * 4,
        tau_hex=(0.5).hex(),
        repetitions=1,
        initial_state="zero",
        angle_unit="radian",
        gate_order="encode_then_fields_even_odd_v1",
    )


def _qiskit_state(plan: StaticCircuitPlan) -> np.ndarray:
    circuit = QuantumCircuit(4)
    for gate, wires, parameter in plan.operations:
        angle = float.fromhex(parameter)
        if gate == "ry":
            circuit.ry(angle, wires[0])
        elif gate == "rz":
            circuit.rz(angle, wires[0])
        elif gate == "rxx":
            circuit.rxx(angle, *wires)
        else:
            circuit.ryy(angle, *wires)
    return np.asarray(Statevector.from_instruction(circuit).data)


def test_static_plan_matches_independent_xy_hamiltonian() -> None:
    """Real Qiskit gates must match a separately constructed 4-qubit matrix."""
    compressed, payload = _compressed()
    spec = _spec()
    xy = build_static_circuit_plan(spec, compressed, payload, row_index=0, arm_id="xy")
    no_coupling = build_static_circuit_plan(
        spec, compressed, payload, row_index=0, arm_id="encoding_plus_fields_no_coupling"
    )
    assert isinstance(decode_contract(canonical_bytes(spec.to_wire())), ReservoirSpec)
    assert decode_contract(canonical_bytes(xy.to_wire())) == xy
    validate_static_circuit_plan(spec, compressed, payload, xy)
    validate_static_circuit_plan(spec, compressed, payload, no_coupling)

    angles = struct.unpack("<4f", payload)
    input_state = np.array([1.0 + 0.0j])
    for angle in reversed(angles):
        input_state = np.kron(
            input_state,
            np.array([math.cos(angle / 2), math.sin(angle / 2)], dtype=complex),
        )
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    hamiltonian = -0.5 * np.kron(np.eye(4), np.kron(x, x) + np.kron(y, y))
    reference = expm(-1j * hamiltonian * 0.5) @ input_state
    np.testing.assert_allclose(_qiskit_state(no_coupling), input_state, atol=1e-13)
    np.testing.assert_allclose(_qiskit_state(xy), reference, atol=1e-13)


def test_static_plan_refuses_source_or_gate_drift() -> None:
    """The public validator must reject an edited gate and an unrelated row."""
    compressed, payload = _compressed()
    spec = _spec()
    plan = build_static_circuit_plan(spec, compressed, payload, row_index=0, arm_id="xy")
    changed = replace(plan, operations=plan.operations[:-1])
    with pytest.raises(ValueError, match="differs"):
        validate_static_circuit_plan(spec, compressed, payload, changed)
    with pytest.raises(ValueError, match="digest mismatch"):
        build_static_circuit_plan(
            spec, compressed, payload[:-1] + b"\x00", row_index=0, arm_id="xy"
        )
    with pytest.raises(ValueError, match="open chain"):
        replace(spec, edges=((1, 0), (1, 2), (2, 3)))
    with pytest.raises(ValueError, match="supported n4/n8"):
        replace(spec, n_qubits=True)
