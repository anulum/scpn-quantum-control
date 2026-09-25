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
