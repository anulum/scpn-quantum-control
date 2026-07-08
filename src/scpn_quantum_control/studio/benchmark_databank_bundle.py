# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio schema-B benchmark-databank bundle emission
"""Federate the committed native-speedup benchmark rows as a schema-B bundle.

The ``benchmark`` verb already emits ``studio.native-speedup.v1`` as a
regression gate. This module federates the committed tier-benchmark *databank*
as ``studio.benchmark-databank.v1``: every measured row rides in ``cases[]``
verbatim, carrying its problem size as the case dimension and its
``speedup_vs_python_median`` as the case error.

The mapping is deliberately honest. The bundle claim boundary carries the
artefact's own timing caveat verbatim — the numbers are opportunistic local
timing on a shared workstation, not publication-grade isolated measurements — so
the databank is a ``bounded-model`` claim that never overstates a speedup. A row
whose native path was absent, or whose parity against the Python reference was
not confirmed, keeps an explicit non-``measured`` status rather than a clean one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scpn_studio_platform.evidence import (
    AdmissionDecision,
    CaseResult,
    ClaimBoundary,
    ClaimStatus,
    EvidenceBundle,
    EvidenceKind,
    EvidenceLevel,
    Freshness,
    NumericProvenance,
    ProvEntity,
    Substrate,
    ValidityDomain,
)

from ..differentiable_claim_ledger import REPO_ROOT
from .evidence_bundle import (
    DEFAULT_ACTIVITY_TIMESTAMP,
    _committed_artifact_edge,
    _digest_payload,
    _studio_activity,
    _studio_agent,
    validate_bundle,
)
from .verbs import BENCHMARK, BENCHMARK_DATABANK_SCHEMA, STUDIO_ID

DEFAULT_BENCHMARK_DATABANK_ARTIFACT_PATH = Path(
    "data/rust_vqe_methods/rust_core_benchmark_summary_2026-05-05.json"
)
"""Repository-relative path of the committed native-speedup benchmark databank."""

BENCHMARK_DATABANK_ARTIFACT_ID = "rust-core-benchmark-summary-2026-05-05"

_REGENERATED_BY = "python -m scpn_quantum_control.studio.benchmark_databank_bundle"


def _row_status(row: dict[str, Any]) -> str:
    """Return the honest per-row status, never upgrading an unproven row.

    A missing availability/parity key (``None``) means the benchmark family does
    not assert it — only an explicit ``False`` demotes the row.
    """
    if row.get("rust_engine_build_knm_available") is False:
        return "native-absent"
    if row.get("parity_with_python_reference") is False:
        return "measured-parity-unverified"
    return "measured"


def _row_metric(row: dict[str, Any]) -> float:
    """Return the row's headline numeric, tolerant of the two benchmark schemas.

    Native-speedup rows carry ``speedup_vs_python_median``; the dense-construction
    rows carry ``hermitian_max_abs_error`` (a correctness residual). A row with
    neither, but a median timing, falls back to the median so no row is dropped.
    """
    for key in ("speedup_vs_python_median", "hermitian_max_abs_error", "median_ms"):
        value = row.get(key)
        if value is not None:
            return float(value)
    raise ValueError(f"benchmark row has no headline metric: {row.get('benchmark')}")


def _load_databank(artifact_path: Path) -> dict[str, Any]:
    """Load and shape-check the committed benchmark databank, failing closed."""
    resolved = artifact_path if artifact_path.is_absolute() else REPO_ROOT / artifact_path
    payload: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"benchmark databank has no rows: {resolved.as_posix()}")
    if not isinstance(payload.get("timing_caveat"), str) or not payload["timing_caveat"]:
        raise ValueError("benchmark databank is missing its timing caveat")
    return payload


def build_benchmark_databank_bundle(
    *,
    artifact_path: Path = DEFAULT_BENCHMARK_DATABANK_ARTIFACT_PATH,
    activity_timestamp: str = DEFAULT_ACTIVITY_TIMESTAMP,
) -> EvidenceBundle:
    """Build the schema-B ``studio.benchmark-databank.v1`` bundle.

    Parameters
    ----------
    artifact_path
        Committed benchmark databank artefact; its content digest rides as a
        ``derived_from`` edge (a missing path fails closed).
    activity_timestamp
        Deterministic PROV timestamp for the emission.

    Returns
    -------
    EvidenceBundle
        The ``studio.benchmark-databank.v1`` bundle, benchmark rows verbatim in
        ``cases[]`` with measured speedups as case errors.

    Raises
    ------
    ValueError
        If the databank has no rows or is missing its timing caveat.
    """
    payload = _load_databank(artifact_path)
    rows: list[dict[str, Any]] = payload["rows"]
    return EvidenceBundle(
        schema=BENCHMARK_DATABANK_SCHEMA,
        entity=ProvEntity(
            entity_id=f"{STUDIO_ID}:benchmark-databank:{BENCHMARK_DATABANK_ARTIFACT_ID}",
            digest=_digest_payload(payload),
        ),
        activity=_studio_activity(
            verb=BENCHMARK.name,
            command=_REGENERATED_BY,
            timestamp=activity_timestamp,
        ),
        agent=_studio_agent(),
        evidence_level=EvidenceLevel.SCIENTIFICALLY_CURATED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.BOUNDED_MODEL,
            admission=AdmissionDecision.ADMITTED,
            validity_domain=ValidityDomain(note=payload["timing_caveat"]),
        ),
        substrate=Substrate.NUMERICAL_MODEL,
        freshness=Freshness.TRACEABLE_UNCHECKED,
        numeric_provenance=NumericProvenance(
            active_backend="rust-native-construction",
            reference_backend="python-reference",
        ),
        cases=tuple(
            CaseResult(
                operation_family=f"benchmark:{row['benchmark']}",
                dimension=int(row["n"]),
                status=_row_status(row),
                error=_row_metric(row),
            )
            for row in rows
        ),
        derived_from=(
            _committed_artifact_edge(
                artifact_path if artifact_path.is_absolute() else REPO_ROOT / artifact_path,
                label="benchmark databank artefact",
            ),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: emit the validated benchmark-databank bundle as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_BENCHMARK_DATABANK_ARTIFACT_PATH,
        help="committed benchmark databank artefact for the derivation edge",
    )
    args = parser.parse_args(argv)
    validated = validate_bundle(build_benchmark_databank_bundle(artifact_path=args.artifact_path))
    print(json.dumps(validated.bundle.to_dict(), indent=2, sort_keys=True))
    if not validated.verdict.admitted:
        print(
            "benchmark-databank bundle rejected: " + "; ".join(validated.verdict.rejections),
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = [
    "BENCHMARK_DATABANK_ARTIFACT_ID",
    "DEFAULT_BENCHMARK_DATABANK_ARTIFACT_PATH",
    "build_benchmark_databank_bundle",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
