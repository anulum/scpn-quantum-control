# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive validate handler
"""The ``validate`` executive action handler — claim-ledger reference validation.

The read-only ``validate`` verb checks the committed differentiable claim
ledger against the committed WS-3 reference-validation registry
(:mod:`scpn_quantum_control.studio.reference_validation`): every certification
must be unique, point at a ledger claim, and certify a promoted claim. It then
measures the coverage frontier (how many ledger claims are reference-validated)
and writes a standalone reproduction script.

The claim boundary is registry-consistency and coverage measurement over the
committed artefacts only: it does not prove any physics claim itself, run a
simulation, or touch hardware.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ..differentiable_claim_ledger import load_differentiable_claim_ledger
from .coverage_frontier import measure_coverage_frontier_from_certifications
from .executive import (
    ActionHandler,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    build_generated_script,
)
from .reference_validation import load_reference_validation_registry
from .verbs import PHYSICS_VALIDATION_SCHEMA

VALIDATE_VERB: Final[str] = "validate"
_DEFAULT_BACKEND: Final[str] = "python"
_MAX_MINIMUM_CLAIMS: Final[int] = 4096

VALIDATE_CLAIM_BOUNDARY: Final[str] = (
    "consistency of the committed WS-3 reference-validation registry against "
    "the committed claim ledger plus the measured coverage frontier; it does "
    "not prove any physics claim itself, run a simulation, or touch hardware"
)


def _as_positive_int(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _normalise_validate(parameters: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(parameters) - {"minimum_total_claims"}
    if unknown:
        raise ValueError(f"unknown validate parameters: {sorted(unknown)}")
    minimum_total_claims = _as_positive_int(
        "minimum_total_claims",
        parameters.get("minimum_total_claims", 1),
        maximum=_MAX_MINIMUM_CLAIMS,
    )
    return {"minimum_total_claims": minimum_total_claims}


class ValidateActionHandler(ActionHandler):
    """Executive handler for the read-only ``validate`` verb."""

    @property
    def verb(self) -> str:
        """Return ``"validate"``."""
        return VALIDATE_VERB

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Resolve a read-only ledger-validation plan.

        Parameters
        ----------
        request : ExecutiveRequest
            The validate request; ``parameters`` may carry an optional
            ``minimum_total_claims`` bound.
        contract : VerbContract
            The resolved ``validate`` contract.

        Returns
        -------
        ExecutionPlan
            The normalised, inspectable plan.
        """
        backend = request.backend or _DEFAULT_BACKEND
        if backend not in contract.backends:
            raise ValueError(f"backend {backend!r} is not declared for the validate verb")
        validate_spec = _normalise_validate(request.parameters)
        steps = (
            "load the committed WS-3 reference-validation registry",
            "load the committed differentiable claim ledger",
            "validate every certification against the ledger",
            "measure the reference-validated coverage frontier",
            "write a standalone reproduction script",
        )
        return ExecutionPlan(
            verb=self.verb,
            action_id=request.action_id,
            backend=backend,
            contract=contract,
            claim_boundary=VALIDATE_CLAIM_BOUNDARY,
            steps=steps,
            parameters=validate_spec,
        )

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Validate the committed registry against the committed ledger.

        Parameters
        ----------
        plan : ExecutionPlan
            The planned validation.

        Returns
        -------
        ExecutionResult
            A succeeded result carrying the validation verdict and frontier.
        """
        validate_spec: dict[str, Any] = dict(plan.parameters)
        registry = load_reference_validation_registry()
        ledger = load_differentiable_claim_ledger()
        validation = registry.validate_against(ledger)
        report = measure_coverage_frontier_from_certifications(ledger, registry=registry)
        minimum = int(validate_spec["minimum_total_claims"])
        outputs = {
            "backend": plan.backend,
            "validation_schema": PHYSICS_VALIDATION_SCHEMA,
            "validation_passed": validation.passed,
            "validation_errors": list(validation.errors),
            "certificate_count": validation.certificate_count,
            "reference_validated_claim_ids": list(validation.reference_validated_claim_ids),
            "total_claims": report.total,
            "minimum_total_claims": minimum,
            "minimum_total_claims_met": report.total >= minimum,
            "answer_rate": report.answer_rate,
            "grade_distribution": dict(report.grade_distribution),
        }
        return ExecutionResult(status="succeeded", outputs=outputs)

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Write a standalone script that reproduces the validation verdict.

        Parameters
        ----------
        plan : ExecutionPlan
            The executed plan.
        result : ExecutionResult
            The succeeded validate result.

        Returns
        -------
        GeneratedScript
            The reproduction script, digest attached.
        """
        source = _render_script(
            action_id=plan.action_id,
            validation_passed=bool(result.outputs["validation_passed"]),
            certificate_count=int(result.outputs["certificate_count"]),
            total_claims=int(result.outputs["total_claims"]),
        )
        slug = _safe_slug(plan.action_id)
        return build_generated_script(
            filename=f"validate_{slug}.py",
            entrypoint=f"python validate_{slug}.py",
            source=source,
        )


def _safe_slug(action_id: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in action_id).strip("_")
    return slug or "action"


def _render_script(
    *,
    action_id: str,
    validation_passed: bool,
    certificate_count: int,
    total_claims: int,
) -> str:
    return (
        '"""Standalone reproduction of a SCPN-QUANTUM-CONTROL studio validate action.\n'
        "\n"
        f"Action id: {action_id}\n"
        "Reloads the committed WS-3 reference-validation registry and claim\n"
        "ledger, revalidates the certifications, and checks the verdict the\n"
        "studio sealed.\n"
        '"""\n\n'
        "from scpn_quantum_control.differentiable_claim_ledger import (\n"
        "    load_differentiable_claim_ledger,\n"
        ")\n"
        "from scpn_quantum_control.studio.coverage_frontier import (\n"
        "    measure_coverage_frontier_from_certifications,\n"
        ")\n"
        "from scpn_quantum_control.studio.reference_validation import (\n"
        "    load_reference_validation_registry,\n"
        ")\n\n"
        f"EXPECTED_VALIDATION_PASSED = {validation_passed!r}\n"
        f"EXPECTED_CERTIFICATE_COUNT = {certificate_count!r}\n"
        f"EXPECTED_TOTAL_CLAIMS = {total_claims!r}\n\n\n"
        "def main() -> int:\n"
        '    """Revalidate the committed registry and verify the sealed verdict."""\n'
        "    registry = load_reference_validation_registry()\n"
        "    ledger = load_differentiable_claim_ledger()\n"
        "    validation = registry.validate_against(ledger)\n"
        "    report = measure_coverage_frontier_from_certifications(ledger, registry=registry)\n"
        "    assert validation.passed == EXPECTED_VALIDATION_PASSED\n"
        "    assert validation.certificate_count == EXPECTED_CERTIFICATE_COUNT\n"
        "    assert report.total == EXPECTED_TOTAL_CLAIMS\n"
        "    print(\n"
        '        f"validation_passed={validation.passed} "\n'
        '        f"certificates={validation.certificate_count} verified"\n'
        "    )\n"
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


__all__ = [
    "VALIDATE_CLAIM_BOUNDARY",
    "VALIDATE_VERB",
    "ValidateActionHandler",
]
