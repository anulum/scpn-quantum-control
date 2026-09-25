# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU lane manifest
"""Inventory and refusal boundary for the opt-in LLM-QPU research lane."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_SCHEMA = "scpn.experimental.llm_qpu.lane_manifest.v1"
_NAMESPACE = "scpn_quantum_control.experimental.llm_qpu"
_KERNELS = (
    "xy_static_digital_v1",
    "xy_sequence_digital_v1",
    "xy_monitored_sequence_v1",
    "conditional_born_policy_v1",
)
_INVENTORY = (
    "__init__.py",
    "llm_qpu/__init__.py",
    "llm_qpu/contracts/__init__.py",
    "llm_qpu/contracts/cells.py",
    "llm_qpu/contracts/compressed.py",
    "llm_qpu/contracts/compressor.py",
    "llm_qpu/contracts/latent.py",
    "llm_qpu/contracts/measurement.py",
    "llm_qpu/contracts/model.py",
    "llm_qpu/contracts/static.py",
    "llm_qpu/contracts/task.py",
    "llm_qpu/contracts/wire.py",
    "llm_qpu/data/artifact_store.py",
    "llm_qpu/data/cache.py",
    "llm_qpu/data/dedup.py",
    "llm_qpu/data/lineage.py",
    "llm_qpu/data/splits.py",
    "llm_qpu/data/tasks.py",
    "llm_qpu/manifest.py",
    "llm_qpu/runtime/__init__.py",
    "llm_qpu/runtime/attempt_journal.py",
)
_WORKER_INVENTORY = ("protocol/worker.py",)
_WRITE_ROOTS = (
    "data/experimental/llm_qpu",
    "results/experimental/llm_qpu",
)


@dataclass(frozen=True, slots=True)
class LaneManifest:
    """Declared experimental scope, with every kernel initially unavailable.

    Parameters
    ----------
    schema
        Versioned manifest schema.
    lane_id
        Exact opt-in lane identifier.
    namespace
        Import namespace, separate from stable product exports.
    kernel_statuses
        Kernel ID and its declared implementation status.
    module_inventory
        Source files relative to the ``experimental`` package directory.
    worker_inventory
        Standalone files relative to ``experimental_workers/llm_qpu``.
    write_roots
        Dedicated repository-relative data and result roots.
    hardware_submission_enabled
        Must remain false until a separately governed execution surface exists.
    claim_promotion_enabled
        Must remain false; this manifest does not certify scientific claims.
    non_claims
        Explicit boundaries on present capability.

    """

    schema: str
    lane_id: str
    namespace: str
    kernel_statuses: tuple[tuple[str, str], ...]
    module_inventory: tuple[str, ...]
    worker_inventory: tuple[str, ...]
    write_roots: tuple[str, ...]
    hardware_submission_enabled: bool
    claim_promotion_enabled: bool
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject undeclared kernels, paths, submissions and promotions."""
        if self.schema != _SCHEMA or self.lane_id != "llm-qpu":
            raise ValueError("unknown experimental lane manifest")
        if self.namespace != _NAMESPACE:
            raise ValueError("experimental namespace mismatch")
        if not isinstance(self.kernel_statuses, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in self.kernel_statuses
        ):
            raise ValueError("kernel statuses must be immutable pairs")
        if tuple(name for name, _ in self.kernel_statuses) != _KERNELS:
            raise ValueError("kernel inventory mismatch")
        if any(status != "not_implemented" for _, status in self.kernel_statuses):
            raise ValueError("kernel readiness requires a new manifest revision")
        if self.module_inventory != _INVENTORY:
            raise ValueError("module inventory requires explicit revision")
        if self.worker_inventory != _WORKER_INVENTORY:
            raise ValueError("worker inventory requires explicit revision")
        if self.write_roots != _WRITE_ROOTS:
            raise ValueError("write roots must remain isolated")
        if (
            type(self.hardware_submission_enabled) is not bool
            or type(self.claim_promotion_enabled) is not bool
        ):
            raise ValueError("execution and promotion flags must be boolean")
        if self.hardware_submission_enabled or self.claim_promotion_enabled:
            raise ValueError("experimental manifest cannot authorize execution or promotion")
        if (
            not isinstance(self.non_claims, tuple)
            or not self.non_claims
            or any(not isinstance(claim, str) or not claim.strip() for claim in self.non_claims)
        ):
            raise ValueError("non-claims must be non-empty")

    def to_wire(self) -> dict[str, object]:
        """Return detached JSON-compatible manifest fields.

        Returns
        -------
        dict[str, object]
            Declared scope and explicit refusal state.

        """
        return {
            "schema": self.schema,
            "lane_id": self.lane_id,
            "status": "experimental",
            "namespace": self.namespace,
            "kernel_statuses": dict(self.kernel_statuses),
            "module_inventory": list(self.module_inventory),
            "worker_inventory": list(self.worker_inventory),
            "write_roots": list(self.write_roots),
            "hardware_submission_enabled": self.hardware_submission_enabled,
            "claim_promotion_enabled": self.claim_promotion_enabled,
            "non_claims": list(self.non_claims),
        }


LANE_MANIFEST = LaneManifest(
    schema=_SCHEMA,
    lane_id="llm-qpu",
    namespace=_NAMESPACE,
    kernel_statuses=tuple((name, "not_implemented") for name in _KERNELS),
    module_inventory=_INVENTORY,
    worker_inventory=_WORKER_INVENTORY,
    write_roots=_WRITE_ROOTS,
    hardware_submission_enabled=False,
    claim_promotion_enabled=False,
    non_claims=(
        "No LLM hidden-state extraction is implemented.",
        "Static gate planning exists; no complete quantum kernel or provider worker is implemented.",
        "No QPU job or scientific advantage follows from this manifest.",
    ),
)


def assert_lane_inventory(package_root: Path | None = None) -> tuple[str, ...]:
    """Refuse unreviewed Python modules in the experimental package.

    Parameters
    ----------
    package_root
        Experimental package directory; defaults to this module's parent
        package. Tests may supply a copied tree to exercise drift refusal.

    Returns
    -------
    tuple[str, ...]
        Sorted exact source paths relative to the experimental package.

    Raises
    ------
    ValueError
        If any declared module is missing or an unlisted module appears.

    """
    root = package_root if package_root is not None else Path(__file__).resolve().parents[1]
    discovered = tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py")))
    if discovered != tuple(sorted(LANE_MANIFEST.module_inventory)):
        raise ValueError("experimental LLM-QPU module inventory drift")
    return discovered


def assert_worker_inventory(worker_root: Path | None = None) -> tuple[str, ...]:
    """Refuse unreviewed standalone worker Python modules.

    Parameters
    ----------
    worker_root
        Worker directory; defaults to this checkout's LLM-QPU workers.

    Returns
    -------
    tuple[str, ...]
        Sorted exact worker source paths relative to their root.

    Raises
    ------
    ValueError
        If declared worker code is missing or unlisted code appears.

    """
    root = (
        worker_root
        if worker_root is not None
        else Path(__file__).resolve().parents[4] / "experimental_workers/llm_qpu"
    )
    discovered = tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py")))
    if discovered != tuple(sorted(LANE_MANIFEST.worker_inventory)):
        raise ValueError("experimental LLM-QPU worker inventory drift")
    return discovered


__all__ = [
    "LANE_MANIFEST",
    "LaneManifest",
    "assert_lane_inventory",
    "assert_worker_inventory",
]
