# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable programming benchmark Program AD IR edge tests
"""Program AD IR edge tests for differentiable-programming benchmarks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _differentiable_programming_benchmark_edge_helpers import (
    _fake_whole_program,
    _gradient,
    _program_ir,
    _whole_program_result,
)

from scpn_quantum_control.benchmarks import differentiable_programming as dp


def test_program_ad_case_records_adjoint_error_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic Program AD rows should include adjoint error when replay is supported."""

    gradient = _gradient(2)
    result = _whole_program_result(
        program_ir=_program_ir(),
        gradient=gradient,
        adjoint_supported=True,
    )
    monkeypatch.setattr(dp, "whole_program_value_and_grad", _fake_whole_program(result))
    monkeypatch.setattr(dp, "program_adjoint_gradient", lambda _result: gradient.copy())

    row = dp._program_ad_case(
        "generic",
        "generic",
        lambda values: values[0],
        np.array([1.0, 2.0], dtype=np.float64),
        gradient.copy(),
    )

    assert row.adjoint_supported is True
    assert row.max_abs_adjoint_error == 0.0

    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(gradient=gradient, adjoint_supported=False)),
    )
    forward_only_row = dp._program_ad_case(
        "generic-forward",
        "generic",
        lambda values: values[0],
        np.array([1.0, 2.0], dtype=np.float64),
        gradient.copy(),
    )

    assert forward_only_row.adjoint_supported is False
    assert forward_only_row.max_abs_adjoint_error is None


def test_program_ad_ir_roundtrip_fails_closed_for_missing_and_mismatched_ir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IR round-trip benchmark should reject missing or non-round-tripping IR."""

    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(program_ir=None)),
    )
    with pytest.raises(ValueError, match="requires program IR"):
        dp._program_ad_ir_roundtrip_case()

    emitted_ir = _program_ir()
    parsed_ir = _program_ir(
        effects=(SimpleNamespace(kind="other", ordering=0),),
    )
    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(program_ir=emitted_ir)),
    )
    monkeypatch.setattr(dp, "parse_program_ad_effect_ir", lambda _serialization: parsed_ir)
    with pytest.raises(ValueError, match="did not reconstruct"):
        dp._program_ad_ir_roundtrip_case()


def test_program_ad_control_phi_metadata_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control/phi metadata benchmark should reject missing or incomplete IR."""

    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(program_ir=None)),
    )
    with pytest.raises(ValueError, match="requires program IR"):
        dp._program_ad_control_phi_metadata_case()

    incomplete_ir = _program_ir(control_regions=(), phi_nodes=(), effects=())
    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(
            _whole_program_result(program_ir=incomplete_ir),
            callback_values=np.array([-1.0, 0.0, 0.75], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(dp, "parse_program_ad_effect_ir", lambda _serialization: incomplete_ir)
    with pytest.raises(ValueError, match="metadata provenance"):
        dp._program_ad_control_phi_metadata_case()

    parsed_ir = _program_ir()
    emitted_ir = _program_ir(effects=(SimpleNamespace(kind="different", ordering=0),))
    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(program_ir=emitted_ir)),
    )
    monkeypatch.setattr(dp, "parse_program_ad_effect_ir", lambda _serialization: parsed_ir)
    with pytest.raises(ValueError, match="parser did not reconstruct"):
        dp._program_ad_control_phi_metadata_case()


def test_program_ad_mlir_interchange_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLIR interchange benchmark should reject missing IR or incomplete metadata."""

    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(
            _whole_program_result(program_ir=None),
            callback_values=np.array([0.25, -0.5, 0.75], dtype=np.float64),
        ),
    )
    with pytest.raises(ValueError, match="requires program IR"):
        dp._program_ad_mlir_interchange_case()

    emitted_ir = _program_ir()
    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(program_ir=emitted_ir)),
    )
    monkeypatch.setattr(
        dp,
        "compile_whole_program_ad_trace_to_mlir",
        lambda _result, _config: SimpleNamespace(metadata={}, text="", resource_counts={}),
    )
    with pytest.raises(ValueError, match="metadata is missing"):
        dp._program_ad_mlir_interchange_case()

    monkeypatch.setattr(
        dp,
        "compile_whole_program_ad_trace_to_mlir",
        lambda _result, _config: SimpleNamespace(
            metadata={"program_ad_ir": {"format": "program_ad_effect_ir.v1"}},
            text='scpn.program_ir_format = "program_ad_effect_ir.v1"',
            resource_counts={
                "program_ad_ssa_values": 0,
                "program_ad_effects": 0,
                "program_ad_control_regions": 0,
                "program_ad_phi_nodes": 0,
            },
        ),
    )
    with pytest.raises(ValueError, match="lowering is incomplete"):
        dp._program_ad_mlir_interchange_case()


def test_program_ad_registry_dispatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry-dispatch benchmark should reject incomplete coverage reports."""

    monkeypatch.setattr(
        dp,
        "program_ad_registry_dispatch_coverage_report",
        lambda: SimpleNamespace(supported=False, blocked_identities=("missing",)),
    )
    with pytest.raises(ValueError, match="missing"):
        dp._program_ad_registry_dispatch_coverage_case()

    monkeypatch.setattr(
        dp,
        "program_ad_registry_dispatch_coverage_report",
        lambda: SimpleNamespace(
            supported=True,
            blocked_identities=(),
            covered_primitives=1,
            total_primitives=2,
            family_counts={},
        ),
    )
    with pytest.raises(ValueError, match="count mismatch"):
        dp._program_ad_registry_dispatch_coverage_case()

    monkeypatch.setattr(
        dp,
        "program_ad_registry_dispatch_coverage_report",
        lambda: SimpleNamespace(
            supported=True,
            blocked_identities=(),
            covered_primitives=2,
            total_primitives=2,
            family_counts={"array": 1},
        ),
    )
    with pytest.raises(ValueError, match="family coverage"):
        dp._program_ad_registry_dispatch_coverage_case()


def test_program_ad_adjoint_replay_provenance_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adjoint replay provenance benchmark should reject missing or inconsistent IR."""

    invalid_adjoint = SimpleNamespace(adjoint_steps=(), replay_node_count=0)
    monkeypatch.setattr(dp, "program_adjoint_result", lambda _result: invalid_adjoint)
    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(program_ir=None)),
    )
    with pytest.raises(ValueError, match="requires program IR"):
        dp._program_adjoint_replay_provenance_case()

    step = SimpleNamespace(
        effect_ordering=0,
        operation="branch:runtime",
        supported=False,
        contribution_inputs=("x",),
        contribution_scales=(1.0,),
        contribution_cotangents=(1.0,),
        effect_kind="control_branch",
        effect_version=0,
        control_region=None,
        control_region_kind=None,
        control_region_entered=False,
        phi_node=None,
        phi_selected=None,
        incoming_cotangent=1.0,
        non_executed_phi_inputs=(),
    )
    inconsistent_adjoint = SimpleNamespace(
        replay_node_count=99,
        replay_effect_count=99,
        replay_control_region_count=99,
        replay_phi_node_count=99,
        replay_ir_format="wrong",
        adjoint_step_count=99,
        adjoint_steps=(step,),
    )
    monkeypatch.setattr(dp, "program_adjoint_result", lambda _result: inconsistent_adjoint)
    monkeypatch.setattr(
        dp,
        "whole_program_value_and_grad",
        _fake_whole_program(_whole_program_result(program_ir=_program_ir())),
    )
    with pytest.raises(ValueError, match="provenance does not match"):
        dp._program_adjoint_replay_provenance_case()
