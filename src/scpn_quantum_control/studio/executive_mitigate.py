# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive mitigate handler
"""The ``mitigate`` executive action handler — zero-noise extrapolation.

The read-only ``mitigate`` verb applies polynomial zero-noise extrapolation
with delta-method uncertainty propagation
(:mod:`scpn_quantum_control.mitigation.zne_uncertainty`) to measured
expectation values at amplified noise scales: a weighted least-squares fit
when per-scale standard errors are supplied, ordinary least squares
otherwise, and a coverage interval on the zero-noise estimate either way.

The claim boundary is the extrapolation arithmetic only: the handler proves
the polynomial fit and its propagated uncertainty over the *given* measured
values — it does not run circuits, amplify noise itself, model the device
noise physics, or validate that the supplied expectations came from a real
experiment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from ..mitigation.zne_uncertainty import zne_extrapolate_with_uncertainty
from .executive import (
    ActionHandler,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    build_generated_script,
)
from .verbs import MITIGATION_SCHEMA

MITIGATE_VERB: Final[str] = "mitigate"
_DEFAULT_BACKEND: Final[str] = "numpy"
_MAX_POINTS: Final[int] = 32
_REPRODUCTION_TOLERANCE: Final[float] = 1e-9

MITIGATE_CLAIM_BOUNDARY: Final[str] = (
    "polynomial zero-noise extrapolation of the given measured expectation "
    "values with delta-method uncertainty propagation; it does not run "
    "circuits, amplify noise, model the device noise physics, or validate "
    "that the supplied expectations came from a real experiment"
)


def _as_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _as_float_sequence(name: str, value: object) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of real numbers")
    return [_as_float(f"{name} entry", entry) for entry in value]


def _normalise_mitigate(parameters: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"noise_scales", "expectation_values", "standard_errors", "order", "coverage"}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"unknown mitigate parameters: {sorted(unknown)}")

    noise_scales = _as_float_sequence("noise_scales", parameters.get("noise_scales"))
    if not 2 <= len(noise_scales) <= _MAX_POINTS:
        raise ValueError(f"noise_scales must carry between 2 and {_MAX_POINTS} points")
    if any(scale < 1.0 for scale in noise_scales):
        raise ValueError("noise_scales must be >= 1 (physical noise amplification)")
    if len(set(noise_scales)) != len(noise_scales):
        raise ValueError("noise_scales must be distinct")

    expectation_values = _as_float_sequence(
        "expectation_values", parameters.get("expectation_values")
    )
    if len(expectation_values) != len(noise_scales):
        raise ValueError("expectation_values length must match noise_scales")

    raw_errors = parameters.get("standard_errors")
    standard_errors: list[float] | None = None
    if raw_errors is not None:
        standard_errors = _as_float_sequence("standard_errors", raw_errors)
        if len(standard_errors) != len(noise_scales):
            raise ValueError("standard_errors length must match noise_scales")
        if any(error <= 0.0 for error in standard_errors):
            raise ValueError("standard_errors must be positive")

    order = parameters.get("order", 1)
    if order not in (1, 2) or isinstance(order, bool):
        raise ValueError("order must be 1 or 2")

    coverage = _as_float("coverage", parameters.get("coverage", 0.95))
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie strictly between 0 and 1")

    return {
        "noise_scales": noise_scales,
        "expectation_values": expectation_values,
        "standard_errors": standard_errors,
        "order": int(order),
        "coverage": coverage,
    }


class MitigateActionHandler(ActionHandler):
    """Executive handler for the read-only ``mitigate`` verb."""

    @property
    def verb(self) -> str:
        """Return ``"mitigate"``."""
        return MITIGATE_VERB

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Validate the measured points and resolve an extrapolation plan.

        Parameters
        ----------
        request : ExecutiveRequest
            The mitigate request; ``parameters`` must carry ``noise_scales``
            and ``expectation_values``, and may carry ``standard_errors``,
            ``order`` and ``coverage``.
        contract : VerbContract
            The resolved ``mitigate`` contract.

        Returns
        -------
        ExecutionPlan
            The normalised, inspectable plan.
        """
        backend = request.backend or _DEFAULT_BACKEND
        if backend not in contract.backends:
            raise ValueError(f"backend {backend!r} is not declared for the mitigate verb")
        mitigate_spec = _normalise_mitigate(request.parameters)
        weighting = (
            "weighted least squares over the supplied standard errors"
            if mitigate_spec["standard_errors"] is not None
            else "ordinary least squares over the fit residual"
        )
        steps = (
            f"validate the {len(mitigate_spec['noise_scales'])}-point noise-scaled sweep",
            f"fit an order-{mitigate_spec['order']} polynomial to the expectation values",
            f"propagate the uncertainty via {weighting}",
            f"report the {mitigate_spec['coverage']:.2f}-coverage interval",
            "write a standalone reproduction script",
        )
        return ExecutionPlan(
            verb=self.verb,
            action_id=request.action_id,
            backend=backend,
            contract=contract,
            claim_boundary=MITIGATE_CLAIM_BOUNDARY,
            steps=steps,
            parameters=mitigate_spec,
        )

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Extrapolate to zero noise and propagate the uncertainty.

        Parameters
        ----------
        plan : ExecutionPlan
            The planned extrapolation.

        Returns
        -------
        ExecutionResult
            A succeeded result carrying the zero-noise estimate, its standard
            error, and the coverage interval.

        Raises
        ------
        ValueError
            When the fit is under-determined (for example an ordinary
            least-squares request with fewer than ``order + 2`` points) — the
            spine seals this as a failed record.
        """
        mitigate_spec: dict[str, Any] = dict(plan.parameters)
        result = zne_extrapolate_with_uncertainty(
            mitigate_spec["noise_scales"],
            mitigate_spec["expectation_values"],
            standard_errors=mitigate_spec["standard_errors"],
            order=mitigate_spec["order"],
            coverage=mitigate_spec["coverage"],
        )
        outputs = {
            "backend": plan.backend,
            "mitigation_schema": MITIGATION_SCHEMA,
            "method": result.method,
            "zero_noise_estimate": result.zero_noise_estimate,
            "standard_error": result.standard_error,
            "interval_low": result.low,
            "interval_high": result.high,
            "interval_width": result.width,
            "coverage": result.coverage,
            "order": result.order,
            "fit_residual": result.fit_residual,
            "n_points": result.n_points,
        }
        return ExecutionResult(status="succeeded", outputs=outputs)

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Write a standalone script that reproduces the extrapolation.

        Parameters
        ----------
        plan : ExecutionPlan
            The executed plan.
        result : ExecutionResult
            The succeeded mitigate result.

        Returns
        -------
        GeneratedScript
            The reproduction script, digest attached.
        """
        mitigate_spec: dict[str, Any] = dict(plan.parameters)
        source = _render_script(
            action_id=plan.action_id,
            mitigate_spec=mitigate_spec,
            zero_noise_estimate=float(result.outputs["zero_noise_estimate"]),
            standard_error=float(result.outputs["standard_error"]),
            method=str(result.outputs["method"]),
        )
        slug = _safe_slug(plan.action_id)
        return build_generated_script(
            filename=f"mitigate_{slug}.py",
            entrypoint=f"python mitigate_{slug}.py",
            source=source,
        )


def _safe_slug(action_id: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in action_id).strip("_")
    return slug or "action"


def _render_script(
    *,
    action_id: str,
    mitigate_spec: Mapping[str, Any],
    zero_noise_estimate: float,
    standard_error: float,
    method: str,
) -> str:
    return (
        '"""Standalone reproduction of a SCPN-QUANTUM-CONTROL studio mitigate action.\n'
        "\n"
        f"Action id: {action_id}\n"
        "Re-runs the zero-noise extrapolation with uncertainty propagation on\n"
        "the sealed measured points and checks the estimate the studio sealed\n"
        "to a numerical tolerance.\n"
        '"""\n\n'
        "from scpn_quantum_control.mitigation.zne_uncertainty import (\n"
        "    zne_extrapolate_with_uncertainty,\n"
        ")\n\n"
        f"NOISE_SCALES = {mitigate_spec['noise_scales']!r}\n"
        f"EXPECTATION_VALUES = {mitigate_spec['expectation_values']!r}\n"
        f"STANDARD_ERRORS = {mitigate_spec['standard_errors']!r}\n"
        f"ORDER = {mitigate_spec['order']!r}\n"
        f"COVERAGE = {mitigate_spec['coverage']!r}\n"
        f"EXPECTED_ZERO_NOISE_ESTIMATE = {zero_noise_estimate!r}\n"
        f"EXPECTED_STANDARD_ERROR = {standard_error!r}\n"
        f"EXPECTED_METHOD = {method!r}\n"
        f"TOLERANCE = {_REPRODUCTION_TOLERANCE!r}\n\n\n"
        "def main() -> int:\n"
        '    """Re-extrapolate and verify the sealed zero-noise estimate."""\n'
        "    result = zne_extrapolate_with_uncertainty(\n"
        "        NOISE_SCALES,\n"
        "        EXPECTATION_VALUES,\n"
        "        standard_errors=STANDARD_ERRORS,\n"
        "        order=ORDER,\n"
        "        coverage=COVERAGE,\n"
        "    )\n"
        "    assert result.method == EXPECTED_METHOD, result.method\n"
        "    assert abs(result.zero_noise_estimate - EXPECTED_ZERO_NOISE_ESTIMATE) <= TOLERANCE\n"
        "    assert abs(result.standard_error - EXPECTED_STANDARD_ERROR) <= TOLERANCE\n"
        "    print(\n"
        '        f"zero_noise_estimate={result.zero_noise_estimate:.9f} "\n'
        '        f"standard_error={result.standard_error:.3e} verified"\n'
        "    )\n"
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


__all__ = [
    "MITIGATE_CLAIM_BOUNDARY",
    "MITIGATE_VERB",
    "MitigateActionHandler",
]
