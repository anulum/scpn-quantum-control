# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 promotion autopilot tests
"""Tests for the guarded Paper 0 promotion-slice autopilot."""

from __future__ import annotations

import pytest

from scripts.automate_paper0_promotion_slice import (
    WorkOrder,
    claim_boundary,
    class_name,
    components_from_records,
    ledger_number,
    snake_case,
    validate_work_order,
)


def test_validate_work_order_rejects_non_contiguous_count() -> None:
    payload = {
        "work_orders": [
            {
                "order": 1,
                "source_start": "P0R00010",
                "source_end": "P0R00012",
                "source_record_count": 2,
                "next_source_boundary": "P0R00013",
                "first_header": "Example",
                "section_path": "Example",
                "required_surfaces": [
                    "scripts/build_paper0_example_specs.py",
                    "src/scpn_quantum_control/paper0/example_validation.py",
                    "scripts/run_paper0_example_fixture.py",
                    "tests/test_build_paper0_example_specs.py",
                    "tests/test_paper0_example_validation.py",
                    "tests/test_run_paper0_example_fixture.py",
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="source_record_count does not match contiguous span"):
        validate_work_order(payload, 0)


def test_components_from_records_split_on_headers() -> None:
    order = WorkOrder(
        order=1,
        source_start="P0R00001",
        source_end="P0R00004",
        source_record_count=4,
        next_source_boundary="P0R00005",
        first_header="Root Header",
        section_path="Root",
        required_surfaces=(
            "scripts/build_paper0_root_header_specs.py",
            "src/scpn_quantum_control/paper0/root_header_validation.py",
            "scripts/run_paper0_root_header_fixture.py",
            "tests/test_build_paper0_root_header_specs.py",
            "tests/test_paper0_root_header_validation.py",
            "tests/test_run_paper0_root_header_fixture.py",
        ),
        math_ids=(),
        image_ids=(),
        table_ids=(),
    )
    records = (
        {"ledger_id": "P0R00001", "block_type": "Header", "text": "Alpha Header"},
        {"ledger_id": "P0R00002", "block_type": "Para", "text": "Alpha body"},
        {"ledger_id": "P0R00003", "block_type": "Header", "text": "Beta Header"},
        {"ledger_id": "P0R00004", "block_type": "Para", "text": "Beta body"},
    )
    components = components_from_records(order, records)
    assert [component.component_id for component in components] == ["alpha_header", "beta_header"]
    assert components[0].ledger_ids == ("P0R00001", "P0R00002")
    assert components[1].ledger_ids == ("P0R00003", "P0R00004")


def test_identifier_helpers_are_deterministic_and_ascii() -> None:
    assert ledger_number("P0R01727") == 1727
    assert (
        snake_case("The Intrinsic Dynamics of the Ψ-Field")
        == "the_intrinsic_dynamics_of_the_psi_field"
    )
    assert class_name("source_slice", "Config") == "SourceSliceConfig"
    assert (
        claim_boundary("source_slice")
        == "source-bounded source slice source-accounting bridge; not validation evidence"
    )
