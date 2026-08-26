# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QIR/CUDA-Q compiler boundary product
"""Fail-closed **external compiler boundary register** product surface.

Productises first-class boundary rows for external compilers (QIR, CUDA-Q,
Catalyst-as-external, in-tree MLIR/Enzyme, future TN) — not aspirational README
lists:

* versioned boundary schema with explicit support statuses;
* catalogue populated from the ambient LLVM/JIT claim gate and Catalyst
  workflow comparison artefacts;
* optional QIR import/export remains validate-only and fail-closed;
* CUDA-Q remains ``permanent_boundary`` without an owner GPU programme;
* refuse invent-green full CUDA-Q runtime or live QIR provider submission.

Does **not** ship full CUDA-Q runtime product, invent Verified-At-Source pins
without terminal evidence, or complete governed-route and watch automation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from .benchmarks.differentiable_catalyst_comparison import (
    catalyst_compiler_workflow_comparison,
)
from .compiler.mlir_llvm_jit_claim_gate import LLVM_JIT_CLAIM_GATE_BOUNDARY

BoundaryStatus = Literal[
    "supported",
    "adapter",
    "implementation_path",
    "permanent_boundary",
]
"""Boundary status enum for external compiler rows."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
    "metadata_only",
]
"""Support posture badges for compiler boundary rows."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

COMPILER_BOUNDARY_PRODUCT_SCHEMA: Final[str] = "compiler_boundary_product.v2"
"""JSON schema identifier for serialised product payloads."""

COMPILER_BOUNDARY_CLAIM_BOUNDARY: Final[str] = (
    "External compiler boundary register product surface only; catalogues QIR, "
    "CUDA-Q, Catalyst-as-external, in-tree MLIR/Enzyme, and future TN compilers "
    "with status enum supported|adapter|implementation_path|permanent_boundary; "
    "composes ambient LLVM/JIT claim gate + Catalyst workflow comparison; refuse "
    "invent-green full CUDA-Q runtime and live QIR provider submission; governed-route "
    "and watch automation plus Rust LLVM/JIT decision citation remain unresolved"
)
"""Shared claim boundary for compiler boundary product payloads."""

_COMPILER_BOUNDARY_POLICY_NOTE: Final[str] = (
    "External compiler boundary register only; not a marketing tick list; CUDA-Q "
    "permanent_boundary without owner GPU programme; QIR validate-only; governed-route "
    "and watch automation plus Rust LLVM/JIT decision citation remain unresolved."
)
_COMPILER_BOUNDARY_REGISTRY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "claim_boundary",
        "compiler_count",
        "blank_entry_count",
        "invent_green_runtime_policy",
        "invent_green_qir_provider_submit_policy",
        "public_surfaces",
        "compilers",
        "policy_note",
    }
)


@dataclass(frozen=True, slots=True)
class CompilerBoundaryRow:
    """One external-compiler boundary register row.

    Attributes
    ----------
    compiler_id
        Stable compiler identifier.
    title
        Human-readable title.
    summary
        Short description of the boundary.
    status
        Boundary status enum.
    ambient_pointer
        Ambient module or constant pointer for evidence.
    route_matrix_pointer
        Governed-route matrix pointer when applicable.
    import_export_allowed
        Whether product allows import/export experiment (validate-only).
    invent_green_runtime
        Must remain False (no invent-green full runtime product).
    support_posture
        Support posture badge.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    compiler_id: str
    title: str
    summary: str
    status: BoundaryStatus
    ambient_pointer: str
    route_matrix_pointer: str
    import_export_allowed: bool
    invent_green_runtime: bool = False
    support_posture: SupportPosture = "metadata_only"
    as_of: str = "2026-07-24"
    claim_boundary: str = COMPILER_BOUNDARY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate compiler boundary row invariants."""
        if not self.compiler_id or not self.compiler_id.strip():
            raise ValueError("compiler_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.status not in {
            "supported",
            "adapter",
            "implementation_path",
            "permanent_boundary",
        }:
            raise ValueError(f"unknown status: {self.status!r}")
        if not self.ambient_pointer or not self.ambient_pointer.strip():
            raise ValueError("ambient_pointer must be non-empty")
        if not self.route_matrix_pointer or not self.route_matrix_pointer.strip():
            raise ValueError("route_matrix_pointer must be non-empty")
        if self.invent_green_runtime:
            raise ValueError("invent_green_runtime must be False")
        if self.status == "permanent_boundary" and self.import_export_allowed:
            raise ValueError("permanent_boundary rows must set import_export_allowed=False")
        if self.support_posture not in {
            "local_research",
            "live_hardware_gated",
            "policy_only",
            "metadata_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "compiler_id": self.compiler_id,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "ambient_pointer": self.ambient_pointer,
            "route_matrix_pointer": self.route_matrix_pointer,
            "import_export_allowed": self.import_export_allowed,
            "invent_green_runtime": self.invent_green_runtime,
            "support_posture": self.support_posture,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for compiler boundary product use.

    Attributes
    ----------
    outcome
        Allowed or refused.
    allowed
        Whether the path may proceed under this product.
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.
    claim_boundary
        Non-promotional claim boundary.

    """

    outcome: PathDecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = COMPILER_BOUNDARY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate path eligibility invariants."""
        if self.outcome not in {"allowed", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.outcome != "allowed":
            raise ValueError("allowed decisions must use outcome=allowed")
        if not self.allowed and self.outcome != "refused":
            raise ValueError("refused decisions must use outcome=refused")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "outcome": self.outcome,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedCompilerBoundaryProbe:
    """Materialised probe over the ambient Catalyst and LLVM claim gates.

    Attributes
    ----------
    catalyst_runner_status
        Ambient catalyst comparison runner status.
    catalyst_promotion_ready
        Ambient promotion_ready flag (must be False for invent-green refuse).
    llvm_claim_gate_boundary
        Ambient LLVM/JIT claim-gate boundary string.
    invent_green_cudaq_runtime
        Always False.
    invent_green_qir_provider_submit
        Always False.
    demo_label
        Demo fixture label.
    claim_boundary
        Non-promotional claim boundary.

    """

    catalyst_runner_status: str
    catalyst_promotion_ready: bool
    llvm_claim_gate_boundary: str
    invent_green_cudaq_runtime: bool
    invent_green_qir_provider_submit: bool
    demo_label: str
    claim_boundary: str = COMPILER_BOUNDARY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate probe invariants."""
        if not self.catalyst_runner_status or not self.catalyst_runner_status.strip():
            raise ValueError("catalyst_runner_status must be non-empty")
        if not self.llvm_claim_gate_boundary or not self.llvm_claim_gate_boundary.strip():
            raise ValueError("llvm_claim_gate_boundary must be non-empty")
        if self.invent_green_cudaq_runtime:
            raise ValueError("invent_green_cudaq_runtime must be False")
        if self.invent_green_qir_provider_submit:
            raise ValueError("invent_green_qir_provider_submit must be False")
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "catalyst_runner_status": self.catalyst_runner_status,
            "catalyst_promotion_ready": self.catalyst_promotion_ready,
            "llvm_claim_gate_boundary": self.llvm_claim_gate_boundary,
            "invent_green_cudaq_runtime": self.invent_green_cudaq_runtime,
            "invent_green_qir_provider_submit": self.invent_green_qir_provider_submit,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _build_compiler_catalogue() -> tuple[CompilerBoundaryRow, ...]:
    """Build the fixed external-compiler boundary catalogue."""
    return (
        CompilerBoundaryRow(
            compiler_id="mlir_enzyme_in_tree",
            title="In-tree MLIR / Enzyme native path",
            summary=(
                "Ambient compiler.mlir_* + LLVM/JIT claim gate; executable "
                "lowering is not promotion without verified evidence."
            ),
            status="adapter",
            ambient_pointer=(
                "scpn_quantum_control.compiler.mlir_llvm_jit_claim_gate."
                "LLVM_JIT_CLAIM_GATE_BOUNDARY"
            ),
            route_matrix_pointer="governed_route:compiler.mlir_enzyme.local",
            import_export_allowed=True,
            support_posture="local_research",
        ),
        CompilerBoundaryRow(
            compiler_id="catalyst_external",
            title="PennyLane Catalyst (external)",
            summary=(
                "Ambient catalyst_compiler_workflow_comparison; compiled "
                "workflows are bounded; provider routes permanent_boundary."
            ),
            status="implementation_path",
            ambient_pointer=(
                "scpn_quantum_control.benchmarks.differentiable_catalyst_comparison."
                "catalyst_compiler_workflow_comparison"
            ),
            route_matrix_pointer="governed_route:compiler.catalyst.external",
            import_export_allowed=True,
            support_posture="local_research",
        ),
        CompilerBoundaryRow(
            compiler_id="qir",
            title="QIR (Quantum Intermediate Representation)",
            summary=(
                "Boundary register for QIR import/export validate-only spikes; "
                "no live provider QIR job submission on this product."
            ),
            status="implementation_path",
            ambient_pointer="scpn_quantum_control.compiler (QIR validate residual)",
            route_matrix_pointer="governed_route:compiler.qir.validate_only",
            import_export_allowed=True,
            support_posture="metadata_only",
        ),
        CompilerBoundaryRow(
            compiler_id="cudaq",
            title="NVIDIA CUDA-Q",
            summary=(
                "Permanent boundary without owner GPU programme; full CUDA-Q "
                "runtime product is out of scope for this surface."
            ),
            status="permanent_boundary",
            ambient_pointer="CUDA-Q runtime policy: unsupported runtime promotion refused",
            route_matrix_pointer="governed_route:compiler.cudaq.permanent_boundary",
            import_export_allowed=False,
            support_posture="policy_only",
        ),
        CompilerBoundaryRow(
            compiler_id="tensor_network_future",
            title="Future tensor-network compilers",
            summary=(
                "Placeholder permanent_boundary for future TN compilers; not a "
                "marketing tick list."
            ),
            status="permanent_boundary",
            ambient_pointer="tensor-network compiler integration remains unsupported",
            route_matrix_pointer="governed_route:compiler.tn.future",
            import_export_allowed=False,
            support_posture="policy_only",
        ),
    )


_CANONICAL: Final[tuple[CompilerBoundaryRow, ...]] = _build_compiler_catalogue()


def _catalogue_map() -> dict[str, CompilerBoundaryRow]:
    """Return compiler_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, CompilerBoundaryRow] = {}
    for row in _CANONICAL:
        key = row.compiler_id.strip()
        if not key:
            raise RuntimeError("compiler boundary catalogue contains blank compiler_id")
        if key in mapping:
            raise RuntimeError(f"duplicate compiler_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("compiler boundary catalogue must be non-empty")
    return mapping


_BY_ID: Final[Mapping[str, CompilerBoundaryRow]] = _catalogue_map()


def list_compiler_ids() -> tuple[str, ...]:
    """Return all compiler identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Stable compiler ids.

    """
    return tuple(row.compiler_id for row in _CANONICAL)


def get_compiler_boundary(compiler_id: str) -> CompilerBoundaryRow:
    """Return one boundary row; fail closed on blank/unknown.

    Parameters
    ----------
    compiler_id
        Compiler identifier.

    Returns
    -------
    CompilerBoundaryRow
        Matching row.

    Raises
    ------
    ValueError
        If blank or unknown.

    """
    if not compiler_id or not str(compiler_id).strip():
        raise ValueError("compiler_id must be non-empty")
    key = str(compiler_id).strip()
    try:
        return _BY_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown compiler_id: {key!r}") from exc


def iter_compiler_boundaries(
    *,
    status: BoundaryStatus | None = None,
    support_posture: SupportPosture | None = None,
) -> tuple[CompilerBoundaryRow, ...]:
    """Return filtered boundary rows in stable order.

    Parameters
    ----------
    status
        Optional status filter.
    support_posture
        Optional posture filter.

    Returns
    -------
    tuple[CompilerBoundaryRow, ...]
        Matching rows.

    """
    rows: Sequence[CompilerBoundaryRow] = _CANONICAL
    if status is not None:
        rows = tuple(row for row in rows if row.status == status)
    if support_posture is not None:
        rows = tuple(row for row in rows if row.support_posture == support_posture)
    return tuple(rows)


def decide_compiler_path(
    compiler_id: str,
    *,
    request_import_export: bool = False,
    invent_green_full_runtime: bool = False,
    invent_green_provider_submit: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a compiler boundary path may proceed.

    Parameters
    ----------
    compiler_id
        Compiler identifier.
    request_import_export
        Whether the caller requests import/export validate-only path.
    invent_green_full_runtime
        If true, refuse (especially CUDA-Q).
    invent_green_provider_submit
        If true, refuse live QIR/provider submission invent-green.

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused with blockers.

    """
    row = get_compiler_boundary(compiler_id)
    blockers: list[str] = []
    if invent_green_full_runtime:
        blockers.append(
            f"invent-green full runtime refused for {row.compiler_id!r} (status={row.status})"
        )
    if invent_green_provider_submit:
        blockers.append(
            f"invent-green provider/QIR submit refused for {row.compiler_id!r} "
            "(validate-only residual)"
        )
    if request_import_export and not row.import_export_allowed:
        blockers.append(f"import/export not allowed for {row.compiler_id!r} (status={row.status})")
    if row.status == "permanent_boundary" and request_import_export:
        blockers.append(f"permanent_boundary compiler {row.compiler_id!r} refuses import/export")
    if blockers:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="compiler path refused under fail-closed boundary product policy",
            blockers=tuple(blockers),
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"compiler path allowed for {row.compiler_id!r} "
            f"(status={row.status}; invent_green_runtime=False)"
        ),
        blockers=(),
    )


def materialise_compiler_boundary_probe(
    *,
    catalyst_runner_status: Literal[
        "dependency_gap", "runtime_gap", "correctness_gap", "success"
    ] = "runtime_gap",
) -> MaterialisedCompilerBoundaryProbe:
    """Materialise probe from ambient Catalyst comparison + LLVM claim gate.

    Parameters
    ----------
    catalyst_runner_status
        Runner status passed to ambient catalyst_compiler_workflow_comparison.

    Returns
    -------
    MaterialisedCompilerBoundaryProbe
        Finite primary observables with invent-green flags False.

    """
    comparison = catalyst_compiler_workflow_comparison(runner_status=catalyst_runner_status)
    return MaterialisedCompilerBoundaryProbe(
        catalyst_runner_status=str(comparison.runner_status),
        catalyst_promotion_ready=bool(comparison.promotion_ready),
        llvm_claim_gate_boundary=str(LLVM_JIT_CLAIM_GATE_BOUNDARY),
        invent_green_cudaq_runtime=False,
        invent_green_qir_provider_submit=False,
        demo_label="ambient_catalyst_and_llvm_claim_gate_probe",
    )


def materialise_demo_compiler_boundary_probe() -> MaterialisedCompilerBoundaryProbe:
    """Materialise the deterministic demo probe (runtime_gap Catalyst path).

    Returns
    -------
    MaterialisedCompilerBoundaryProbe
        Offline ambient composition probe.

    """
    return materialise_compiler_boundary_probe(catalyst_runner_status="runtime_gap")


def map_compiler_boundary_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of compiler boundary product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    return (
        {
            "module_path": "scpn_quantum_control.compiler_boundary_product",
            "role": "compiler_boundary_product_surface",
            "support_posture": "metadata_only",
            "compiler_ids": list(list_compiler_ids()),
            "invent_green_runtime": False,
            "claim_boundary": COMPILER_BOUNDARY_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.compiler.mlir_llvm_jit_claim_gate",
            "role": "ambient_llvm_jit_claim_gate",
            "support_posture": "policy_only",
            "symbol_name": "LLVM_JIT_CLAIM_GATE_BOUNDARY",
            "claim_boundary": COMPILER_BOUNDARY_CLAIM_BOUNDARY,
        },
        {
            "module_path": ("scpn_quantum_control.benchmarks.differentiable_catalyst_comparison"),
            "role": "ambient_catalyst_workflow_comparison",
            "support_posture": "local_research",
            "symbol_name": "catalyst_compiler_workflow_comparison",
            "claim_boundary": COMPILER_BOUNDARY_CLAIM_BOUNDARY,
        },
    )


def build_compiler_boundary_product_registry() -> dict[str, object]:
    """Build the full serialisable compiler boundary product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with rows (no blanks).

    """
    compilers = [row.to_dict() for row in _CANONICAL]
    return {
        "schema": COMPILER_BOUNDARY_PRODUCT_SCHEMA,
        "claim_boundary": COMPILER_BOUNDARY_CLAIM_BOUNDARY,
        "compiler_count": len(compilers),
        "blank_entry_count": 0,
        "invent_green_runtime_policy": False,
        "invent_green_qir_provider_submit_policy": False,
        "public_surfaces": list(map_compiler_boundary_public_surfaces()),
        "compilers": compilers,
        "policy_note": _COMPILER_BOUNDARY_POLICY_NOTE,
    }


def assert_compiler_boundary_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert registry covers compilers without blanks or invent-green runtime.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_compiler_boundary_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green policies appear.

    """
    registry = dict(payload) if payload is not None else build_compiler_boundary_product_registry()
    if registry.get("schema") != COMPILER_BOUNDARY_PRODUCT_SCHEMA:
        raise ValueError("unexpected compiler boundary product schema")
    if set(registry) != _COMPILER_BOUNDARY_REGISTRY_KEYS:
        raise ValueError("compiler boundary product registry keys drift")
    if registry.get("claim_boundary") != COMPILER_BOUNDARY_CLAIM_BOUNDARY:
        raise ValueError("compiler boundary product claim boundary drift")
    if registry.get("public_surfaces") != list(map_compiler_boundary_public_surfaces()):
        raise ValueError("compiler boundary product public surface map drift")
    if registry.get("policy_note") != _COMPILER_BOUNDARY_POLICY_NOTE:
        raise ValueError("compiler boundary product policy note drift")
    compilers = registry.get("compilers")
    if not isinstance(compilers, list) or not compilers:
        raise ValueError(
            "compiler boundary product registry must contain a non-empty compilers list"
        )
    seen: set[str] = set()
    blank = 0
    cudaq_found = False
    qir_found = False
    for index, row in enumerate(compilers):
        if not isinstance(row, Mapping):
            raise ValueError(f"compiler row {index} must be a mapping")
        compiler_id = row.get("compiler_id")
        status = row.get("status")
        invent = row.get("invent_green_runtime")
        import_export = row.get("import_export_allowed")
        ambient = row.get("ambient_pointer")
        if not compiler_id or not str(compiler_id).strip():
            blank += 1
            continue
        cid = str(compiler_id).strip()
        if cid in seen:
            raise ValueError(f"duplicate compiler_id in registry: {cid!r}")
        seen.add(cid)
        if cid == "cudaq":
            cudaq_found = True
        if cid == "qir":
            qir_found = True
        if status not in {
            "supported",
            "adapter",
            "implementation_path",
            "permanent_boundary",
        }:
            raise ValueError(f"compiler {cid!r} has unknown status: {status!r}")
        if invent is not False:
            raise ValueError(f"compiler {cid!r} invent_green_runtime must be False")
        if row.get("claim_boundary") != COMPILER_BOUNDARY_CLAIM_BOUNDARY:
            raise ValueError(f"compiler {cid!r} claim boundary drift")
        if not ambient or not str(ambient).strip():
            raise ValueError(f"compiler {cid!r} must have ambient_pointer")
        if status == "permanent_boundary" and import_export is not False:
            raise ValueError(
                f"compiler {cid!r} permanent_boundary must set import_export_allowed=False"
            )
        canonical = _BY_ID.get(cid)
        if canonical is not None and dict(row) != canonical.to_dict():
            raise ValueError(f"compiler {cid!r} catalogue row drift")
    if blank:
        raise ValueError(
            f"compiler boundary product registry has {blank} blank or invalid entries"
        )
    if not cudaq_found:
        raise ValueError("compiler boundary product registry missing cudaq")
    if not qir_found:
        raise ValueError("compiler boundary product registry missing qir")
    expected = set(list_compiler_ids())
    if seen != expected:
        raise ValueError(
            f"registry compiler set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    compiler_count = registry.get("compiler_count", -1)
    if not isinstance(compiler_count, int) or compiler_count != len(compilers):
        raise ValueError("compiler_count does not match compilers list length")
    invent_policy = registry.get("invent_green_runtime_policy", True)
    if invent_policy is not False:
        raise ValueError("invent_green_runtime_policy must be False")
    submit_policy = registry.get("invent_green_qir_provider_submit_policy", True)
    if submit_policy is not False:
        raise ValueError("invent_green_qir_provider_submit_policy must be False")
    return registry


__all__ = [
    "COMPILER_BOUNDARY_CLAIM_BOUNDARY",
    "COMPILER_BOUNDARY_PRODUCT_SCHEMA",
    "BoundaryStatus",
    "CompilerBoundaryRow",
    "MaterialisedCompilerBoundaryProbe",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SupportPosture",
    "assert_compiler_boundary_product_integrity",
    "build_compiler_boundary_product_registry",
    "decide_compiler_path",
    "get_compiler_boundary",
    "iter_compiler_boundaries",
    "list_compiler_ids",
    "map_compiler_boundary_public_surfaces",
    "materialise_compiler_boundary_probe",
    "materialise_demo_compiler_boundary_probe",
]
