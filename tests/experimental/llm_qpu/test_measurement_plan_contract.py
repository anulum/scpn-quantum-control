# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — LLM-QPU measurement plan contract
"""Protect common-axis observables and explicit raw-bit orientation."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    MeasurementPlan,
    ReservoirSpec,
    StaticCircuitPlan,
    build_measurement_plan,
    canonical_bytes,
    decode_contract,
    estimate_sampled_basis,
)


def _reservoir() -> ReservoirSpec:
    """Return the n4 open-chain recipe the circuit is bound to."""
    return ReservoirSpec(
        kernel_id="xy_static_digital_v1",
        n_qubits=4,
        edges=((0, 1), (1, 2), (2, 3)),
        couplings_hex=((0.5).hex(),) * 3,
        fields_hex=((0.0).hex(),) * 4,
        tau_hex=(0.5).hex(),
        repetitions=1,
        initial_state="zero",
        angle_unit="radian",
        gate_order="encode_then_fields_even_odd_v1",
    )


def _circuit() -> StaticCircuitPlan:
    """Return a one-gate static circuit bound to the n4 recipe."""
    reservoir = _reservoir()
    return StaticCircuitPlan(
        reservoir_digest=hashlib.sha256(canonical_bytes(reservoir.to_wire())).hexdigest(),
        compressed_digest="b" * 64,
        row_index=0,
        sample_id="scene-1",
        arm_id="xy",
        operations=(("ry", (0,), (0.25).hex()),),
    )


def _plan(basis: str, *, physical: tuple[int, ...] | None = (5, 8, 6, 9)) -> MeasurementPlan:
    """Return the 256-shot readout plan of one basis with a permuted bit custody."""
    return build_measurement_plan(
        _circuit(),
        _reservoir(),
        basis_id=basis,
        shots=256,
        logical_to_clbit=(2, 0, 3, 1),
        display_order_clbits=(3, 2, 1, 0),
        register_name="readout",
        physical_qubits=physical,
    )


def test_common_axis_plan_and_compiled_bit_orientation() -> None:
    """Three bases provide exactly 21 n4 features with explicit bit custody."""
    plans = tuple(_plan(basis) for basis in ("X", "Y", "Z"))
    assert sum(len(plan.observable_ids) for plan in plans) == 21
    assert plans[0].observable_ids == ("X0", "X1", "X2", "X3", "X0X1", "X1X2", "X2X3")
    assert plans[1].readout_rotations()[:2] == (("sdg", 0), ("h", 0))
    assert plans[2].readout_rotations() == ()
    assert plans[0].logical_bits("1010") == (0, 0, 1, 1)
    assert decode_contract(canonical_bytes(plans[0].to_wire())) == plans[0]
    assert replace(plans[0], shots=512).scientific_digest == plans[0].scientific_digest
    assert replace(plans[0], shots=512).request_digest != plans[0].request_digest
    assert (
        replace(plans[0], display_order_clbits=(0, 1, 2, 3)).request_digest
        != plans[0].request_digest
    )


def test_measurement_plan_refuses_axis_and_mapping_drift() -> None:
    """Unknown layout cannot be treated as compiled or silently reoriented."""
    plan = _plan("Y")
    with pytest.raises(ValueError, match="layout"):
        _plan("Y", physical=None).logical_bits("1010")
    with pytest.raises(ValueError, match="raw key"):
        plan.logical_bits("10 0")
    with pytest.raises(ValueError, match="observable"):
        replace(plan, observable_ids=("X0",) + plan.observable_ids[1:])
    with pytest.raises(ValueError, match="permutation"):
        replace(plan, logical_to_clbit=(0, 0, 2, 3))
    with pytest.raises(ValueError, match="physical"):
        replace(plan, physical_qubits=(5, 5, 6, 9))
    with pytest.raises(ValueError, match="shots"):
        replace(plan, shots=True)
    with pytest.raises(ValueError, match="reservoir digest"):
        build_measurement_plan(
            replace(_circuit(), reservoir_digest="a" * 64),
            _reservoir(),
            basis_id="Y",
            shots=256,
            logical_to_clbit=(2, 0, 3, 1),
            display_order_clbits=(3, 2, 1, 0),
            register_name="readout",
            physical_qubits=(5, 8, 6, 9),
        )
    wrong_wire = plan.to_wire()
    wrong_wire["display_order_clbits"] = [0, 1, 2]
    with pytest.raises(ValueError, match="display bit order"):
        decode_contract(canonical_bytes(wrong_wire))


def test_sparse_joint_counts_preserve_correlations_and_bit_mapping() -> None:
    """Two sparse joint distributions with equal marginals retain distinct ZZ."""
    plan = replace(_plan("Z"), shots=2)
    correlated = estimate_sampled_basis(plan, {"0000": 1, "0101": 1}, origin="hardware_raw")
    anticorrelated = estimate_sampled_basis(plan, {"0001": 1, "0100": 1}, origin="hardware_raw")
    assert correlated.expectations[:2] == anticorrelated.expectations[:2]
    assert correlated.expectations[4] == 1.0
    assert anticorrelated.expectations[4] == -1.0
    assert correlated.origin == "hardware_raw"
    assert correlated.covariance_of_mean is not None
    assert correlated.covariance_of_mean[0][0] == 1.0


def test_sampled_covariance_and_missing_layout_refusal() -> None:
    """A single shot has undefined covariance; wrong shots and layout fail."""
    one = replace(_plan("X"), shots=1)
    result = estimate_sampled_basis(one, {"0000": 1}, origin="digital_sampled")
    assert result.expectations == (1.0,) * 7
    assert result.covariance_of_mean is None
    with pytest.raises(ValueError, match="requested shots"):
        estimate_sampled_basis(one, {"0000": 2}, origin="hardware_raw")
    with pytest.raises(ValueError, match="layout"):
        estimate_sampled_basis(
            replace(one, physical_qubits=None), {"0000": 1}, origin="hardware_raw"
        )
    with pytest.raises(ValueError, match="origin"):
        estimate_sampled_basis(one, {"0000": 1}, origin="rounded_counts")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_qubits": 6}, "measurement plan requires n4/n8"),
        ({"n_qubits": True}, "measurement plan requires n4/n8"),
        ({"basis_id": "W"}, "measurement basis must be X, Y or Z"),
        ({"register_name": "réadout"}, "measurement register must be NFC text"),
        ({"pauli_label_convention": "big_endian"}, "unsupported Pauli label convention"),
    ],
)
def test_measurement_plan_refuses_unsupported_width_basis_register_and_labels(
    changes: dict[str, Any], message: str
) -> None:
    """A plan refuses another width, an unknown basis, a non-NFC register and other labels."""
    with pytest.raises(ValueError, match=f"^{message}$"):
        replace(_plan("Z"), **changes)


def test_measurement_plan_wire_refuses_edited_fields_and_containers() -> None:
    """The decoder refuses an edited field set or object kind and non-list containers."""
    encoded = _plan("Z").to_wire()
    cases: list[tuple[dict[str, object], str]] = [
        ({**encoded, "extra": 1}, "measurement plan fields mismatch"),
        ({**encoded, "object_kind": "static_circuit_plan"}, "unsupported measurement plan schema"),
        ({**encoded, "observable_ids": "Z0"}, "measurement observable_ids must be a list"),
        ({**encoded, "physical_qubits": {"q0": 5}}, "measurement physical qubits must be a list"),
    ]
    for value, message in cases:
        with pytest.raises(ValueError, match=f"^{message}"):
            decode_contract(canonical_bytes(value))
    assert decode_contract(canonical_bytes(_plan("Z", physical=None).to_wire())) == _plan(
        "Z", physical=None
    )


def test_sampled_estimate_refuses_unbounded_or_inconsistent_counts() -> None:
    """Counts must be a bounded nonempty mapping of positive integers that sums to the shots."""
    plan = _plan("Z")
    listed: Any = [("0000", 256)]
    too_many = {f"{index:05b}": 1 for index in range(17)}
    flag: Any = True
    for counts, message in (
        (listed, "sampled estimate requires bounded nonempty counts"),
        ({}, "sampled estimate requires bounded nonempty counts"),
        (too_many, "sampled estimate requires bounded nonempty counts"),
        ({"0000": 0}, "raw count must be a positive integer"),
        ({"0000": flag}, "raw count must be a positive integer"),
        ({"0000": 200}, "raw counts do not match requested shots"),
    ):
        with pytest.raises(ValueError, match=f"^{message}$"):
            estimate_sampled_basis(plan, counts, origin="digital_sampled")


def test_measurement_build_refuses_unreviewed_or_wider_circuits() -> None:
    """The builder needs reviewed contracts and a circuit inside the recipe's width."""
    loose_circuit: Any = _circuit().to_wire()
    loose_reservoir: Any = _reservoir().to_wire()
    for circuit, reservoir in ((loose_circuit, _reservoir()), (_circuit(), loose_reservoir)):
        with pytest.raises(
            ValueError, match="^measurement requires a static circuit and reservoir$"
        ):
            build_measurement_plan(
                circuit,
                reservoir,
                basis_id="Z",
                shots=256,
                logical_to_clbit=(0, 1, 2, 3),
                display_order_clbits=(3, 2, 1, 0),
                register_name="readout",
                physical_qubits=None,
            )
    wider = replace(_circuit(), operations=(("ry", (5,), (0.25).hex()),))
    with pytest.raises(ValueError, match="^circuit operation exceeds measurement width$"):
        build_measurement_plan(
            wider,
            _reservoir(),
            basis_id="Z",
            shots=256,
            logical_to_clbit=(0, 1, 2, 3),
            display_order_clbits=(3, 2, 1, 0),
            register_name="readout",
            physical_qubits=None,
        )
