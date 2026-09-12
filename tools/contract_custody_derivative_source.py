# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — derivative source evidence capture
"""Capture native derivative requests/results without inventing semantic binding."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

import numpy as np

from scpn_quantum_control import differentiable as ad
from scpn_quantum_control import stable_core_product as scp


def derivative_evidence_source(
    method: Literal["forward_mode", "reverse_mode", "parameter_shift"],
) -> dict[str, Any]:
    """Execute a bounded native derivative producer and capture its evidence.

    Parameters
    ----------
    method
        Forward/reverse AD of a deterministic polynomial, or propagation of
        supplied finite-shot estimates through the parameter-shift rule.

    Returns
    -------
    dict
        JSON-ready inputs, complete native result, its digest, source identities
        and actual gradient shape/dtype. Missing native units, calibration and
        companion bindings stay unavailable. Supplied samples are not hardware
        observations; stochastic evaluations count estimates, not local calls.

    Raises
    ------
    ValueError
        If the requested capture is not one of the three supported producers.

    """
    if method not in {"forward_mode", "reverse_mode", "parameter_shift"}:
        raise ValueError("unsupported derivative evidence method")
    parameters = [ad.Parameter("z"), ad.Parameter("a", trainable=False)]
    metadata = [asdict(parameter) for parameter in parameters]
    calls = 0
    inputs: dict[str, Any]
    result_type: type[ad.StochasticGradientResult] | type[ad.GradientResult]
    call_observation: int | None
    if method == "parameter_shift":
        inputs = {
            "plus_values": [0.8, 0.1],
            "minus_values": [0.2, -0.3],
            "plus_variances": [0.36, 0.25],
            "minus_variances": [0.16, 0.09],
            "plus_shots": [900, 400],
            "minus_shots": [400, 100],
            "value": 0.5,
            "confidence_level": 0.95,
            "confidence_z": 1.959963984540054,
            "rule": None,
            "failure_policy": None,
            "sample_provenance": {
                "sample_seed": "derivative-custody-fixture",
                "shot_batch_id": "supplied-estimate-pair",
                "source_class": "caller_supplied",
            },
        }
        stochastic = ad.parameter_shift_gradient_with_uncertainty(**inputs, parameters=parameters)
        inputs["parameters"] = metadata
        payload = stochastic.to_dict()
        result_type = type(stochastic)
        gradient = stochastic.gradient
        producer = (
            "scpn_quantum_control.differentiable_parameter_shift."
            "parameter_shift_gradient_with_uncertainty"
        )
        call_observation = None
        evaluation_semantics = "materialised shifted estimates; no objective callable invoked"
    else:
        values = np.array([3.0, 2.0])

        def objective(x: Any) -> Any:
            """Count actual calls while accepting the canonical AD scalar types."""
            nonlocal calls
            calls += 1
            return x[0] ** 2 + 5 * x[1]

        result = ad.value_and_grad(objective, values, parameters=parameters, method=method)
        assert isinstance(result, ad.GradientResult)
        inputs = {
            "objective": "x[0]**2 + 5*x[1]",
            "values": values.tolist(),
            "parameters": metadata,
            "method": method,
            "rule": None,
            "step": None,
        }
        payload = asdict(result)
        payload["gradient"] = result.gradient.tolist()
        payload["parameter_names"] = list(result.parameter_names)
        payload["trainable"] = list(result.trainable)
        result_type = type(result)
        gradient = result.gradient
        producer = "scpn_quantum_control.differentiable_canonical_api.value_and_grad"
        call_observation = calls
        evaluation_semantics = "observed calls to the supplied deterministic objective"
    return {
        "producer": producer,
        "parameter_type": f"{ad.Parameter.__module__}.{ad.Parameter.__qualname__}",
        "result_type": f"{result_type.__module__}.{result_type.__qualname__}",
        "inputs": inputs,
        "result": payload,
        "result_sha256": scp.digest_stable_core_payload(payload),
        "observed_objective_calls": call_observation,
        "evaluation_count_semantics": evaluation_semantics,
        "gradient_layout": {"shape": list(gradient.shape), "dtype": str(gradient.dtype)},
        "unavailable": ["native_parameter_units", "calibration_reference", "semantic_binding"],
        "claim_boundary": payload["claim_boundary"],
        "capture_boundary": "bounded local derivative evidence; not hardware or companion conformance",
    }
