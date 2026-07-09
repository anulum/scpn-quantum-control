# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive execute (QPU deployment) handler
"""The ``execute`` executive action handler — approval-gated QPU deployment.

``execute`` is the studio's only ``live-hardware`` verb. The studio **never**
submits a live QPU job itself: submission needs provider credentials, costs
real money, and is owner/operator territory. This handler is therefore an
approval-gated *deployment planner and script generator*, not a submitter. On an
approved request it:

1. validates a bounded deployment spec (provider, endpoint, backend, the
   bit-exact circuit digest, and shots);
2. builds a **no-submit** deployment dossier that declares, honestly, that no
   live job ran and that the eventual result is attestation-verifiable —
   ``unverifiable`` until the operator submits and attaches a provider
   attestation (the ``studio.qpu-result-pack.v1`` contract);
3. writes a standalone operator submission script — hard-gated behind
   ``--confirm`` — that the operator runs, with their own credentials, to deploy
   the circuit onto the endpoint and produce a signed result pack.

Without an explicit approval on the request the spine never reaches this handler
(the verb is ``live-hardware``/``certified``, so it fails closed). Even with
approval, nothing here contacts a provider or fabricates counts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .executive import (
    ActionHandler,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    build_generated_script,
)
from .verbs import QPU_RESULT_PACK_SCHEMA

EXECUTE_VERB: Final[str] = "execute"
_DEFAULT_BACKEND: Final[str] = "qiskit-runtime"
_MAX_SHOTS: Final[int] = 1_000_000
_DIGEST_PREFIX: Final[str] = "sha256:"

EXECUTE_CLAIM_BOUNDARY: Final[str] = (
    "plans and scripts an approval-gated QPU deployment onto a provider endpoint; "
    "the studio does not submit a live job, use credentials, or produce counts — "
    "the eventual result is attestation-verifiable and stays unverifiable until "
    "the operator submits and attaches a provider attestation"
)

_UNVERIFIABLE: Final[str] = "unverifiable"


def _as_positive_int(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    if value > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return value


def _as_non_empty_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _as_digest(name: str, value: object) -> str:
    text = _as_non_empty_str(name, value)
    if not text.startswith(_DIGEST_PREFIX):
        raise ValueError(f"{name} must be a '{_DIGEST_PREFIX}<hex>' digest")
    return text


def _normalise_deployment(parameters: Mapping[str, Any]) -> dict[str, Any]:
    provider = _as_non_empty_str("provider", parameters.get("provider"))
    endpoint = _as_non_empty_str("endpoint", parameters.get("endpoint"))
    circuit_digest = _as_digest("circuit_digest", parameters.get("circuit_digest"))
    circuit_ref = _as_non_empty_str("circuit_ref", parameters.get("circuit_ref"))
    shots = _as_positive_int("shots", parameters.get("shots"), maximum=_MAX_SHOTS)
    deployment: dict[str, Any] = {
        "provider": provider,
        "endpoint": endpoint,
        "circuit_digest": circuit_digest,
        "circuit_ref": circuit_ref,
        "shots": shots,
    }
    calibration_ref = parameters.get("calibration_ref")
    if calibration_ref is not None:
        deployment["calibration_ref"] = _as_non_empty_str("calibration_ref", calibration_ref)
    return deployment


class ExecuteActionHandler(ActionHandler):
    """Executive handler for the approval-gated live-hardware ``execute`` verb."""

    @property
    def verb(self) -> str:
        """Return ``"execute"``."""
        return EXECUTE_VERB

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Validate the deployment spec and resolve the gated plan.

        Parameters
        ----------
        request : ExecutiveRequest
            The execute request; ``parameters`` must describe a bounded
            deployment (``provider``, ``endpoint``, ``circuit_digest``,
            ``circuit_ref``, ``shots``, optional ``calibration_ref``).
        contract : VerbContract
            The resolved ``execute`` contract (live-hardware, approval-gated).

        Returns
        -------
        ExecutionPlan
            The normalised, inspectable, approval-gated plan.
        """
        backend = request.backend or _DEFAULT_BACKEND
        if backend not in contract.backends:
            raise ValueError(f"backend {backend!r} is not declared for the execute verb")
        deployment = _normalise_deployment(request.parameters)
        steps = (
            f"validate the deployment spec for provider {deployment['provider']!r}",
            "resolve the approval gate (live-hardware fails closed without approval)",
            "build the no-submit deployment dossier (result stays unverifiable)",
            f"write the operator submission script for the {backend} backend",
        )
        return ExecutionPlan(
            verb=self.verb,
            action_id=request.action_id,
            backend=backend,
            contract=contract,
            claim_boundary=EXECUTE_CLAIM_BOUNDARY,
            steps=steps,
            parameters=deployment,
        )

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Build the no-submit deployment dossier (never contacts a provider).

        Parameters
        ----------
        plan : ExecutionPlan
            The approved deployment plan.

        Returns
        -------
        ExecutionResult
            A succeeded result whose outputs are the deployment dossier. The
            dossier declares ``submitted=False`` and ``result_status`` unverifiable:
            no live job ran and no counts were produced.
        """
        deployment: dict[str, Any] = dict(plan.parameters)
        script_name = f"deploy_{_safe_slug(plan.action_id)}.py"
        dossier = {
            "submitted": False,
            "backend": plan.backend,
            "provider": deployment["provider"],
            "endpoint": deployment["endpoint"],
            "circuit_digest": deployment["circuit_digest"],
            "circuit_ref": deployment["circuit_ref"],
            "shots": deployment["shots"],
            "calibration_ref": deployment.get("calibration_ref"),
            "result_schema": QPU_RESULT_PACK_SCHEMA,
            "verifiability_mode": "attestation",
            "result_status": _UNVERIFIABLE,
            "submit_command": f"python {script_name} --confirm",
            "note": (
                "the studio never submits a live job; run the generated script with "
                "your provider credentials to deploy the circuit and produce a "
                "provider-attested result pack"
            ),
        }
        return ExecutionResult(status="succeeded", outputs=dossier)

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Write the operator submission script (hard-gated behind ``--confirm``).

        Parameters
        ----------
        plan : ExecutionPlan
            The executed deployment plan.
        result : ExecutionResult
            The succeeded deployment dossier.

        Returns
        -------
        GeneratedScript
            The operator submission scaffold, digest attached. The studio never
            runs this script; the operator does, with their own credentials.
        """
        deployment: dict[str, Any] = dict(plan.parameters)
        script_name = f"deploy_{_safe_slug(plan.action_id)}.py"
        source = _render_submission_script(
            action_id=plan.action_id,
            backend=plan.backend,
            deployment=deployment,
        )
        return build_generated_script(
            filename=script_name,
            entrypoint=f"python {script_name} --confirm",
            source=source,
        )


def _safe_slug(action_id: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in action_id).strip("_")
    return slug or "action"


def _render_submission_script(
    *, action_id: str, backend: str, deployment: Mapping[str, Any]
) -> str:
    return (
        '"""Operator QPU deployment scaffold from a SCPN-QUANTUM-CONTROL studio execute action.\n'
        "\n"
        f"Action id: {action_id}\n"
        "\n"
        "This script SUBMITS a live QPU job through your provider account and costs\n"
        "real money. The studio never runs it. Fill in `submit_circuit(...)` with your\n"
        "provider client, then run with --confirm. The returned counts are digested and\n"
        "handed to the studio result-pack builder; attach your provider attestation to\n"
        "make the result attestation-verifiable.\n"
        '"""\n\n'
        "import argparse\n"
        "import hashlib\n"
        "import json\n\n"
        f"BACKEND = {backend!r}\n"
        f"PROVIDER = {deployment['provider']!r}\n"
        f"ENDPOINT = {deployment['endpoint']!r}\n"
        f"CIRCUIT_DIGEST = {deployment['circuit_digest']!r}\n"
        f"CIRCUIT_REF = {deployment['circuit_ref']!r}\n"
        f"SHOTS = {deployment['shots']!r}\n"
        f"CALIBRATION_REF = {deployment.get('calibration_ref')!r}\n\n\n"
        "def submit_circuit(client: object) -> dict[str, int]:\n"
        '    """Submit the circuit to the endpoint and return {bitstring: count}.\n'
        "\n"
        "    Wire your provider client here (qiskit-ibm-runtime Sampler, a provider\n"
        "    HAL, etc.). This is the only place that contacts real hardware.\n"
        '    """\n'
        "    raise NotImplementedError(\n"
        '        f"connect your {PROVIDER} client for {ENDPOINT} and return the counts"\n'
        "    )\n\n\n"
        "def raw_results_digest(counts: dict[str, int]) -> str:\n"
        '    """Return the sha256 digest the provider attestation must sign over."""\n'
        '    payload = json.dumps(counts, sort_keys=True, separators=(",", ":"))\n'
        '    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()\n\n\n'
        "def main() -> int:\n"
        '    """Deploy the circuit after an explicit --confirm."""\n'
        "    parser = argparse.ArgumentParser(description=__doc__)\n"
        '    parser.add_argument("--confirm", action="store_true", help="submit the live job")\n'
        "    args = parser.parse_args()\n"
        "    if not args.confirm:\n"
        '        print("refusing to submit a live QPU job without --confirm")\n'
        "        return 2\n"
        "    counts = submit_circuit(client=None)\n"
        "    digest = raw_results_digest(counts)\n"
        "    from scpn_quantum_control.studio import build_qpu_result_pack_unit\n\n"
        "    unit = build_qpu_result_pack_unit(\n"
        '        {"id": ' + repr(f"deploy-{_safe_slug(action_id)}") + "},\n"
        "        raw_results_digest=digest,\n"
        "        circuit_digest=CIRCUIT_DIGEST,\n"
        "        calibration_ref=CALIBRATION_REF,\n"
        "        attestation=None,  # attach your provider attestation to verify\n"
        "    )\n"
        "    print(json.dumps(unit, indent=2, sort_keys=True))\n"
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


__all__ = [
    "EXECUTE_CLAIM_BOUNDARY",
    "EXECUTE_VERB",
    "ExecuteActionHandler",
]
