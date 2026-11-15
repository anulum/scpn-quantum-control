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
from typing import Any

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


def _compressed(
    values: tuple[float, float, float, float] = (0.1, -0.2, 0.3, -0.4),
) -> tuple[CompressedLatentBatch, bytes]:
    """Return a one-row contextual batch of four angles and its exact payload."""
    angles = struct.pack("<4f", *values)
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
    """Return the n4 open-chain recipe with one coupled edge and no fields."""
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
    """Run the plan's gates through real Qiskit gates and return the state."""
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


def _eight_qubit_spec() -> ReservoirSpec:
    """Return a valid n8 open-chain recipe."""
    return ReservoirSpec(
        kernel_id="xy_static_digital_v1",
        n_qubits=8,
        edges=tuple((index, index + 1) for index in range(7)),
        couplings_hex=((0.25).hex(),) * 7,
        fields_hex=((0.125).hex(),) * 8,
        tau_hex=(1.0).hex(),
        repetitions=2,
        initial_state="zero",
        angle_unit="radian",
        gate_order="encode_then_fields_even_odd_v1",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"kernel_id": "xy_sequence_digital_v1"}, "static reservoir requires a supported n4/n8"),
        (
            {"couplings_hex": ((0.5).hex(), (0.0).hex())},
            "one coupling is required per ordered edge",
        ),
        ({"couplings_hex": [(0.5).hex()] * 3}, "one coupling is required per ordered edge"),
        ({"fields_hex": ((0.0).hex(),) * 3}, "one local field is required per qubit"),
        ({"couplings_hex": (0.5, "0x0.0p+0", "0x0.0p+0")}, "coupling must be canonical hex float"),
        (
            {"couplings_hex": ("zz", "0x0.0p+0", "0x0.0p+0")},
            "coupling must be canonical hex float",
        ),
        ({"couplings_hex": ((11.0).hex(), "0x0.0p+0", "0x0.0p+0")}, "coupling must be bounded"),
        ({"couplings_hex": ("0x1p-1", "0x0.0p+0", "0x0.0p+0")}, "coupling must be bounded"),
        ({"fields_hex": ("inf", "0x0.0p+0", "0x0.0p+0", "0x0.0p+0")}, "field must be bounded"),
        ({"tau_hex": (0.0).hex()}, "tau must be bounded canonical hex float"),
        ({"tau_hex": (-0.5).hex()}, "tau must be bounded canonical hex float"),
        ({"repetitions": 0}, "repetitions must be a bounded positive integer"),
        ({"repetitions": 33}, "repetitions must be a bounded positive integer"),
        ({"initial_state": "plus"}, "unsupported static preparation or gate convention"),
        ({"angle_unit": "degree"}, "unsupported static preparation or gate convention"),
        (
            {"gate_order": "fields_then_encode"},
            "unsupported static preparation or gate convention",
        ),
    ],
)
def test_reservoir_spec_refuses_unbound_recipes(changes: dict[str, Any], message: str) -> None:
    """A recipe refuses wrong counts, non-canonical or unbounded hex and other conventions."""
    with pytest.raises(ValueError, match=f"^{message}"):
        replace(_spec(), **changes)


def test_reservoir_spec_wire_refusals_through_the_decoder() -> None:
    """The decoder refuses an edited reservoir wire and round-trips the n8 recipe."""
    eight = _eight_qubit_spec()
    assert decode_contract(canonical_bytes(eight.to_wire())) == eight
    encoded = _spec().to_wire()
    cases: list[tuple[dict[str, object], str]] = [
        ({**encoded, "extra": 1}, "reservoir spec fields mismatch"),
        ({**encoded, "object_kind": "static_circuit_plan"}, "unsupported reservoir spec schema"),
        ({**encoded, "edges": {"first": [0, 1]}}, "reservoir edges must be a list"),
        ({**encoded, "fields_hex": "0x0.0p+0"}, "reservoir fields_hex must be a list"),
        ({**encoded, "edges": [[0, 1], [1], [2, 3]]}, "reservoir edges must be index pairs"),
        ({**encoded, "edges": [0, [1, 2], [2, 3]]}, "reservoir edges must be index pairs"),
    ]
    for value, message in cases:
        with pytest.raises(ValueError, match=f"^{message}$"):
            decode_contract(canonical_bytes(value))


def _plan() -> StaticCircuitPlan:
    """Return the coupled plan of the n4 recipe for the default batch row."""
    compressed, payload = _compressed()
    return build_static_circuit_plan(_spec(), compressed, payload, row_index=0, arm_id="xy")


_ANGLE = (0.25).hex()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"reservoir_digest": "Z" * 64}, "reservoir must be a lowercase SHA-256 digest"),
        ({"row_index": -1}, "plan row index out of bounds"),
        ({"row_index": 1_000_000}, "plan row index out of bounds"),
        ({"row_index": True}, "plan row index out of bounds"),
        ({"sample_id": ""}, "plan sample must be bounded nonempty text"),
        ({"arm_id": "zz_only"}, "unknown static arm"),
        ({"operations": ()}, "static operation inventory out of bounds"),
        ({"operations": [("ry", (0,), _ANGLE)]}, "static operation inventory out of bounds"),
        pytest.param(
            {"operations": (("ry", (0,), _ANGLE),) * 1025},
            "static operation inventory out of bounds",
            id="too-many-operations",
        ),
        ({"operations": (("ry", (0,)),)}, "invalid static operation"),
        ({"operations": (["ry", (0,), _ANGLE],)}, "invalid static operation"),
        ({"operations": (("rx", (0,), _ANGLE),)}, "unsupported static gate"),
        ({"operations": (("ry", [0], _ANGLE),)}, "unsupported static gate"),
        ({"operations": (("rxx", (0,), _ANGLE),)}, "static gate arity mismatch"),
        ({"operations": (("rz", (0, 1), _ANGLE),)}, "static gate arity mismatch"),
        ({"operations": (("ry", (8,), _ANGLE),)}, "static gate wire out of bounds"),
        ({"operations": (("ry", (True,), _ANGLE),)}, "static gate wire out of bounds"),
        ({"operations": (("ryy", (2, 2), _ANGLE),)}, "static gate repeats a wire"),
        ({"operations": (("ry", (0,), (641.0).hex()),)}, "gate angle must be bounded"),
    ],
)
def test_static_plan_refuses_malformed_gate_inventories(
    changes: dict[str, Any], message: str
) -> None:
    """A plan refuses bad lineage, rows, arms, inventories, gates, wires and angles."""
    with pytest.raises(ValueError, match=f"^{message}"):
        replace(_plan(), **changes)


def test_static_plan_wire_refusals_through_the_decoder() -> None:
    """The decoder refuses an edited gate-list wire field by field."""
    encoded = _plan().to_wire()
    operations = encoded["operations"]
    assert isinstance(operations, list)
    first = operations[0]
    assert isinstance(first, dict)
    cases: list[tuple[dict[str, object], str]] = [
        ({**encoded, "extra": 1}, "static circuit plan fields mismatch"),
        ({**encoded, "object_kind": "reservoir_spec"}, "unsupported static circuit plan schema"),
        ({**encoded, "operations": {"first": first}}, "static gate list fields mismatch"),
        ({**encoded, "operations": [[]]}, "static gate list fields mismatch"),
        (
            {**encoded, "operations": [{"gate": "ry", "wires": [0]}]},
            "static gate list fields mismatch",
        ),
        ({**encoded, "operations": [{**first, "wires": 0}]}, "static gate list fields mismatch"),
    ]
    for value, message in cases:
        with pytest.raises(ValueError, match=f"^{message}$"):
            decode_contract(canonical_bytes(value))


def test_static_builder_and_validator_refuse_unreviewed_inputs() -> None:
    """The builder needs reviewed contracts, matching width, a real row, a known arm and angles."""
    compressed, payload = _compressed()
    spec = _spec()
    loose_spec: Any = spec.to_wire()
    loose_batch: Any = compressed.to_wire()
    for bad_spec, bad_batch in ((loose_spec, compressed), (spec, loose_batch)):
        with pytest.raises(ValueError, match="^static builder requires reviewed contracts$"):
            build_static_circuit_plan(bad_spec, bad_batch, payload, row_index=0, arm_id="xy")
    with pytest.raises(ValueError, match="^static builder requires matching contextual angles$"):
        build_static_circuit_plan(
            _eight_qubit_spec(), compressed, payload, row_index=0, arm_id="xy"
        )
    flag: Any = True
    for row in (1, -1, flag):
        with pytest.raises(ValueError, match="^static row index out of bounds$"):
            build_static_circuit_plan(spec, compressed, payload, row_index=row, arm_id="xy")
    with pytest.raises(ValueError, match="^unknown static arm$"):
        build_static_circuit_plan(spec, compressed, payload, row_index=0, arm_id="fields_only")
    for edge in (1.6, -1.6):
        wide, wide_payload = _compressed((0.1, -0.2, 0.3, edge))
        with pytest.raises(ValueError, match="^static row has an out-of-range angle$"):
            build_static_circuit_plan(spec, wide, wide_payload, row_index=0, arm_id="xy")
    loose_plan: Any = _plan().to_wire()
    with pytest.raises(ValueError, match="^static plan must be a frozen contract$"):
        validate_static_circuit_plan(spec, compressed, payload, loose_plan)
