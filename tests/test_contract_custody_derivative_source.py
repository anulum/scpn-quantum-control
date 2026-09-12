# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — derivative source custody tests
"""Own derivative capture inputs, native outputs and local-only evidence limits."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal, cast

import numpy as np
import pytest

from scpn_quantum_control import differentiable as ad
from scpn_quantum_control import stable_core_product as scp
from tools.contract_custody_derivative_source import derivative_evidence_source


@pytest.mark.parametrize("method", ["forward_mode", "reverse_mode"])
def test_canonical_source_replays_full_result_and_observed_calls(
    method: Literal["forward_mode", "reverse_mode"],
) -> None:
    """Recompute the captured ordered/frozen request through the public dispatcher."""
    source = derivative_evidence_source(method)
    request = source["inputs"]
    assert request["objective"] == "x[0]**2 + 5*x[1]"
    result = ad.value_and_grad(
        lambda x: x[0] ** 2 + 5 * x[1],
        request["values"],
        parameters=[ad.Parameter(**item) for item in request["parameters"]],
        method=request["method"],
        rule=request["rule"],
        step=request["step"],
    )
    assert isinstance(result, ad.GradientResult)
    actual = asdict(result)
    actual["gradient"] = result.gradient.tolist()
    assert scp.canonical_json_bytes(source["result"]) == scp.canonical_json_bytes(actual)
    assert source["result_sha256"] == scp.digest_stable_core_payload(actual)
    assert result.value == 19.0
    np.testing.assert_array_equal(result.gradient, [6.0, 0.0])
    assert result.parameter_names == ("z", "a") and result.trainable == (True, False)
    assert (
        result.evaluations
        == source["observed_objective_calls"]
        == (2 if method == "forward_mode" else 1)
    )
    assert result.method == (
        "forward_mode_dual" if method == "forward_mode" else "reverse_mode_tape"
    )
    assert source["gradient_layout"] == {"shape": [2], "dtype": "float64"}
    assert "native_parameter_units" in source["unavailable"]


def test_stochastic_source_replays_complete_uncertainty_and_sample_provenance() -> None:
    """Replay supplied estimates without labelling them acquired hardware samples."""
    source = derivative_evidence_source("parameter_shift")
    request = dict(source["inputs"])
    request["parameters"] = [ad.Parameter(**item) for item in request["parameters"]]
    result = ad.parameter_shift_gradient_with_uncertainty(**request)
    assert source["result"] == result.to_dict()
    assert source["result_sha256"] == scp.digest_stable_core_payload(result.to_dict())
    np.testing.assert_allclose(result.gradient, [0.3, 0.0], rtol=1e-15, atol=0)
    np.testing.assert_allclose(
        result.covariance, np.array([[0.0002, 0.0], [0.0, 0.0]]), rtol=1e-15, atol=0
    )
    assert result.evaluations == 2
    assert source["observed_objective_calls"] is None
    assert result.hardware_execution is False
    assert result.parameter_names == ("z", "a") and result.trainable == (True, False)
    assert result.records[0].source_class == "caller_supplied"
    assert result.records[0].plus_shots == 900 and result.records[0].minus_shots == 400
    assert result.confidence_level == 0.95
    np.testing.assert_allclose(result.confidence_radius, 1.959963984540054 * result.standard_error)


def test_derivative_capture_owns_its_json_snapshot() -> None:
    """Mutating a returned request/result cannot contaminate the next capture."""
    original = derivative_evidence_source("parameter_shift")
    altered = derivative_evidence_source("parameter_shift")
    altered["inputs"]["parameters"][0]["name"] = "tampered"
    altered["result"]["records"][0]["source_class"] = "tampered"
    altered["result"]["gradient"][0] = 99.0
    assert derivative_evidence_source("parameter_shift") == original


def test_unsupported_derivative_capture_is_refused() -> None:
    """Do not silently substitute another producer for an unsupported request."""
    with pytest.raises(ValueError, match="unsupported derivative evidence method"):
        derivative_evidence_source(cast(Literal["forward_mode"], "unknown"))
