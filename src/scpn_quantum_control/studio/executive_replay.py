# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive replay handler
"""The ``replay`` executive action handler — hardware-result-pack re-verification.

The read-only ``replay`` verb re-verifies the committed hardware result packs
from their raw artefacts and provenance
(:mod:`scpn_quantum_control.hardware_result_packs`): every declared artefact
must exist with its exact byte size and SHA-256 digest, and every declared
provider job identifier must appear inside the committed raw payloads. The
verification fails closed — a missing artefact, a digest mismatch, or an
absent job identifier seals a ``failed`` record rather than a weakened
summary.

The claim boundary is integrity re-verification of the committed artefacts
only: replay proves the raw evidence on disk is exactly what the manifest
promised, not that any physics claim derived from it is correct, and it never
contacts a provider or produces new counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..differentiable_claim_ledger import REPO_ROOT
from ..hardware_result_packs import MANIFEST_RELATIVE_PATH, verify_manifest
from .executive import (
    ActionHandler,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    build_generated_script,
)
from .verbs import EVIDENCE_REPLAY_SCHEMA

REPLAY_VERB: Final[str] = "replay"
_DEFAULT_BACKEND: Final[str] = "python"
_MAX_PACK_IDS: Final[int] = 32

REPLAY_CLAIM_BOUNDARY: Final[str] = (
    "integrity re-verification of the committed hardware result packs — "
    "artefact existence, byte sizes, SHA-256 digests, and declared provider "
    "job identifiers inside the committed raw payloads; it does not prove any "
    "derived physics claim, contact a provider, or produce new counts"
)


def _as_pack_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError("each pack id must be a non-empty string")
    return value


def _normalise_replay(parameters: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(parameters) - {"pack_ids"}
    if unknown:
        raise ValueError(f"unknown replay parameters: {sorted(unknown)}")
    raw_pack_ids = parameters.get("pack_ids")
    if raw_pack_ids is None:
        return {"pack_ids": None}
    if not isinstance(raw_pack_ids, Sequence) or isinstance(raw_pack_ids, (str, bytes)):
        raise ValueError("pack_ids must be a sequence of pack identifiers")
    pack_ids = [_as_pack_id(value) for value in raw_pack_ids]
    if not 1 <= len(pack_ids) <= _MAX_PACK_IDS:
        raise ValueError(f"pack_ids must carry between 1 and {_MAX_PACK_IDS} identifiers")
    if len(set(pack_ids)) != len(pack_ids):
        raise ValueError("pack_ids must be unique")
    return {"pack_ids": sorted(pack_ids)}


class ReplayActionHandler(ActionHandler):
    """Executive handler for the read-only ``replay`` verb."""

    @property
    def verb(self) -> str:
        """Return ``"replay"``."""
        return REPLAY_VERB

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Resolve a read-only result-pack re-verification plan.

        Parameters
        ----------
        request : ExecutiveRequest
            The replay request; ``parameters`` may carry an optional
            ``pack_ids`` selection (every listed pack must exist — unknown
            identifiers fail closed at execution).
        contract : VerbContract
            The resolved ``replay`` contract.

        Returns
        -------
        ExecutionPlan
            The normalised, inspectable plan.
        """
        backend = request.backend or _DEFAULT_BACKEND
        if backend not in contract.backends:
            raise ValueError(f"backend {backend!r} is not declared for the replay verb")
        replay_spec = _normalise_replay(request.parameters)
        selection = (
            "all committed packs"
            if replay_spec["pack_ids"] is None
            else f"{len(replay_spec['pack_ids'])} selected pack(s)"
        )
        steps = (
            "load the committed hardware result-pack manifest",
            f"select {selection}",
            "re-verify every artefact byte size and SHA-256 digest",
            "re-verify every declared provider job identifier in the raw payloads",
            "write a standalone reproduction script",
        )
        return ExecutionPlan(
            verb=self.verb,
            action_id=request.action_id,
            backend=backend,
            contract=contract,
            claim_boundary=REPLAY_CLAIM_BOUNDARY,
            steps=steps,
            parameters=replay_spec,
        )

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Re-verify the committed result packs, failing closed on any drift.

        Parameters
        ----------
        plan : ExecutionPlan
            The planned re-verification.

        Returns
        -------
        ExecutionResult
            A succeeded result carrying the verified manifest summary.

        Raises
        ------
        ValueError
            On an unknown pack id, a byte-size or digest mismatch, or a
            missing declared job identifier — the spine seals this as a
            failed record.
        FileNotFoundError
            When a declared artefact is missing on disk.
        """
        replay_spec: dict[str, Any] = dict(plan.parameters)
        raw_pack_ids = replay_spec["pack_ids"]
        pack_ids = None if raw_pack_ids is None else set(raw_pack_ids)
        summary = verify_manifest(
            REPO_ROOT / MANIFEST_RELATIVE_PATH, repo_root=REPO_ROOT, pack_ids=pack_ids
        )
        outputs = {
            "backend": plan.backend,
            "replay_schema": EVIDENCE_REPLAY_SCHEMA,
            "manifest": summary["manifest"],
            "manifest_schema_version": summary["schema_version"],
            "replay_passed": True,
            "pack_count": summary["pack_count"],
            "artifact_count": summary["artifact_count"],
            "packs": list(summary["packs"]),
        }
        return ExecutionResult(status="succeeded", outputs=outputs)

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Write a standalone script that reproduces the re-verification.

        Parameters
        ----------
        plan : ExecutionPlan
            The executed plan.
        result : ExecutionResult
            The succeeded replay result.

        Returns
        -------
        GeneratedScript
            The reproduction script, digest attached.
        """
        replay_spec: dict[str, Any] = dict(plan.parameters)
        source = _render_script(
            action_id=plan.action_id,
            pack_ids=replay_spec["pack_ids"],
            pack_count=int(result.outputs["pack_count"]),
            artifact_count=int(result.outputs["artifact_count"]),
        )
        slug = _safe_slug(plan.action_id)
        return build_generated_script(
            filename=f"replay_{slug}.py",
            entrypoint=f"python replay_{slug}.py",
            source=source,
        )


def _safe_slug(action_id: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in action_id).strip("_")
    return slug or "action"


def _render_script(
    *,
    action_id: str,
    pack_ids: list[str] | None,
    pack_count: int,
    artifact_count: int,
) -> str:
    return (
        '"""Standalone reproduction of a SCPN-QUANTUM-CONTROL studio replay action.\n'
        "\n"
        f"Action id: {action_id}\n"
        "Re-verifies the committed hardware result packs from their raw\n"
        "artefacts (byte sizes, SHA-256 digests, declared provider job\n"
        "identifiers) and checks the summary the studio sealed. Any drift in\n"
        "the committed evidence raises before the assertions are reached.\n"
        '"""\n\n'
        "from scpn_quantum_control.differentiable_claim_ledger import REPO_ROOT\n"
        "from scpn_quantum_control.hardware_result_packs import (\n"
        "    MANIFEST_RELATIVE_PATH,\n"
        "    verify_manifest,\n"
        ")\n\n"
        f"PACK_IDS = {pack_ids!r}\n"
        f"EXPECTED_PACK_COUNT = {pack_count!r}\n"
        f"EXPECTED_ARTIFACT_COUNT = {artifact_count!r}\n\n\n"
        "def main() -> int:\n"
        '    """Re-verify the committed result packs and the sealed summary."""\n'
        "    summary = verify_manifest(\n"
        "        REPO_ROOT / MANIFEST_RELATIVE_PATH,\n"
        "        repo_root=REPO_ROOT,\n"
        "        pack_ids=None if PACK_IDS is None else set(PACK_IDS),\n"
        "    )\n"
        '    assert summary["pack_count"] == EXPECTED_PACK_COUNT, summary["pack_count"]\n'
        '    assert summary["artifact_count"] == EXPECTED_ARTIFACT_COUNT, (\n'
        '        summary["artifact_count"]\n'
        "    )\n"
        "    print(\n"
        "        f\"replay_verified packs={summary['pack_count']} \"\n"
        "        f\"artifacts={summary['artifact_count']}\"\n"
        "    )\n"
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


__all__ = [
    "REPLAY_CLAIM_BOUNDARY",
    "REPLAY_VERB",
    "ReplayActionHandler",
]
