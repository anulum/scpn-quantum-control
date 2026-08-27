# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable notebook curriculum
"""Fail-closed differentiable notebook curriculum registry and probe.

Productises the core-six differentiable-lane onboarding curriculum:

* versioned catalogue / manifest over ``notebooks/differentiable/``;
* runtime-class badges and mandatory ``hardware_execution: false`` honesty;
* integrity checks refusing blanks, unknown ids, and invent-green QPU notebooks;
* materialised curriculum probe with finite primary observables (counts, ids).

Does **not** convert the full historical notebook archive, invent live QPU
notebooks, or claim a full nbclient CI matrix green.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

RuntimeClass = Literal[
    "cpu_local_fast",
    "cpu_optional_framework",
    "ci_light",
]
"""Runtime class labels for curriculum notebooks."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

DIFFERENTIABLE_NOTEBOOK_CURRICULUM_SCHEMA: Final[str] = "differentiable_notebook_curriculum.v2"
"""JSON schema identifier for serialised curriculum payloads."""

DIFFERENTIABLE_NOTEBOOK_CURRICULUM_CLAIM_BOUNDARY: Final[str] = (
    "Differentiable notebook curriculum registry only; catalogues the "
    "core-six onboarding curriculum under notebooks/differentiable with "
    "hardware_execution=false honesty; materialised manifest/probe only; "
    "refuses invent-green live QPU notebooks and full archive conversion; "
    "does not claim a full nbclient CI matrix green"
)
"""Shared claim boundary for differentiable curriculum payloads."""

_CURRICULUM_DIR_REL: Final[str] = "notebooks/differentiable"
"""Repository-relative curriculum directory."""


@dataclass(frozen=True, slots=True)
class CurriculumNotebookRow:
    """One core curriculum notebook catalogue row.

    Attributes
    ----------
    notebook_id
        Stable curriculum identifier.
    title
        Human-readable title.
    relative_path
        Repo-relative path to the notebook file.
    runtime_class
        Expected runtime class badge.
    summary
        Short description.
    required_packages
        Declared package names for the notebook lane.
    hardware_execution
        Must be False (no invent-green QPU curriculum).
    order
        Stable curriculum order (1-based).
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    notebook_id: str
    title: str
    relative_path: str
    runtime_class: RuntimeClass
    summary: str
    required_packages: tuple[str, ...]
    hardware_execution: bool = False
    order: int = 1
    as_of: str = "2026-07-24"
    claim_boundary: str = DIFFERENTIABLE_NOTEBOOK_CURRICULUM_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate curriculum notebook row invariants."""
        if not self.notebook_id or not self.notebook_id.strip():
            raise ValueError("notebook_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.relative_path or not self.relative_path.strip():
            raise ValueError("relative_path must be non-empty")
        if not self.relative_path.endswith(".ipynb"):
            raise ValueError("relative_path must end with .ipynb")
        if self.runtime_class not in {
            "cpu_local_fast",
            "cpu_optional_framework",
            "ci_light",
        }:
            raise ValueError(f"unknown runtime_class: {self.runtime_class!r}")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.required_packages:
            raise ValueError("required_packages must be non-empty")
        if any(not item or not str(item).strip() for item in self.required_packages):
            raise ValueError("required_packages entries must be non-empty")
        if self.hardware_execution:
            raise ValueError(
                "curriculum notebooks must set hardware_execution=False "
                "(no invent-green QPU notebooks on product surface)"
            )
        if self.order < 1:
            raise ValueError("order must be a positive integer")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "notebook_id": self.notebook_id,
            "title": self.title,
            "relative_path": self.relative_path,
            "runtime_class": self.runtime_class,
            "summary": self.summary,
            "required_packages": list(self.required_packages),
            "hardware_execution": self.hardware_execution,
            "order": self.order,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for differentiable curriculum use.

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

    """

    outcome: PathDecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = DIFFERENTIABLE_NOTEBOOK_CURRICULUM_CLAIM_BOUNDARY

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
class MaterialisedCurriculumProbe:
    """Materialised curriculum catalogue probe with primary observables.

    Attributes
    ----------
    notebook_ids
        Ordered curriculum notebook identifiers.
    notebook_count
        Number of curriculum rows.
    hardware_execution_any
        Whether any row claims hardware execution (must be False).
    default_notebook_id
        Default first-path curriculum id.
    missing_path_count
        Count of rows whose relative paths are missing on disk (0 when present).

    """

    notebook_ids: tuple[str, ...]
    notebook_count: int
    hardware_execution_any: bool
    default_notebook_id: str
    missing_path_count: int
    claim_boundary: str = DIFFERENTIABLE_NOTEBOOK_CURRICULUM_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate materialised curriculum probe invariants."""
        if not self.notebook_ids:
            raise ValueError("notebook_ids must be non-empty")
        if self.notebook_count != len(self.notebook_ids):
            raise ValueError("notebook_count must match notebook_ids length")
        if self.hardware_execution_any:
            raise ValueError("curriculum probe must not report any hardware_execution=True")
        if not self.default_notebook_id or not self.default_notebook_id.strip():
            raise ValueError("default_notebook_id must be non-empty")
        if self.default_notebook_id not in self.notebook_ids:
            raise ValueError("default_notebook_id must be present in notebook_ids")
        if self.missing_path_count < 0:
            raise ValueError("missing_path_count must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "notebook_ids": list(self.notebook_ids),
            "notebook_count": self.notebook_count,
            "hardware_execution_any": self.hardware_execution_any,
            "default_notebook_id": self.default_notebook_id,
            "missing_path_count": self.missing_path_count,
            "claim_boundary": self.claim_boundary,
        }


def _row(
    notebook_id: str,
    *,
    title: str,
    runtime_class: RuntimeClass,
    summary: str,
    required_packages: tuple[str, ...],
    order: int,
) -> CurriculumNotebookRow:
    """Build one curriculum catalogue row."""
    return CurriculumNotebookRow(
        notebook_id=notebook_id,
        title=title,
        relative_path=f"{_CURRICULUM_DIR_REL}/{notebook_id}.ipynb",
        runtime_class=runtime_class,
        summary=summary,
        required_packages=required_packages,
        order=order,
    )


_CANONICAL_CURRICULUM: Final[tuple[CurriculumNotebookRow, ...]] = (
    _row(
        "01_parameter_shift_kuramoto_xy",
        title="Parameter-shift VQE / Kuramoto-XY objective",
        runtime_class="cpu_local_fast",
        summary=(
            "First gradient path: parameter-shift and Kuramoto-XY / Phase-QNode "
            "objectives on CPU-local demos."
        ),
        required_packages=("numpy", "scpn_quantum_control"),
        order=1,
    ),
    _row(
        "02_gradient_tape_simulator",
        title="Gradient tape over simulator objective",
        runtime_class="cpu_local_fast",
        summary=(
            "Reverse / whole-program AD tape on a local simulator objective (no provider submit)."
        ),
        required_packages=("numpy", "scpn_quantum_control"),
        order=2,
    ),
    _row(
        "03_jax_batched_quantum_gradients",
        title="JAX batched quantum gradients",
        runtime_class="cpu_optional_framework",
        summary=(
            "Optional JAX overlay batched quantum gradients; skip cleanly when "
            "JAX is not installed."
        ),
        required_packages=("numpy", "scpn_quantum_control", "jax"),
        order=3,
    ),
    _row(
        "04_pytorch_quantum_layer",
        title="PyTorch quantum layer",
        runtime_class="cpu_optional_framework",
        summary=(
            "Optional PyTorch quantum layer overlay; skip cleanly when torch is not installed."
        ),
        required_packages=("numpy", "scpn_quantum_control", "torch"),
        order=4,
    ),
    _row(
        "05_fail_closed_boundaries",
        title="Fail-closed boundaries / unsupported scenarios",
        runtime_class="cpu_local_fast",
        summary=(
            "Teach unsuitable-scenario safeguards and anti-silent-wrong refusal paths "
            "without invent-green success."
        ),
        required_packages=("numpy", "scpn_quantum_control"),
        order=5,
    ),
    _row(
        "06_witnesses_challenge_fixture",
        title="Sync witnesses + challenge fixture demo",
        runtime_class="cpu_local_fast",
        summary=(
            "Witnesses and challenge fixtures (core families); no exotic science "
            "and no QPU by default."
        ),
        required_packages=("numpy", "scpn_quantum_control"),
        order=6,
    ),
)


def _catalogue_map() -> dict[str, CurriculumNotebookRow]:
    """Return notebook_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, CurriculumNotebookRow] = {}
    for row in _CANONICAL_CURRICULUM:
        key = row.notebook_id.strip()
        if not key:
            raise RuntimeError("notebook curriculum catalogue contains blank notebook_id")
        if key in mapping:
            raise RuntimeError(f"duplicate notebook_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("notebook curriculum catalogue must be non-empty")
    return mapping


_NOTEBOOK_BY_ID: Final[Mapping[str, CurriculumNotebookRow]] = _catalogue_map()


def list_curriculum_notebook_ids() -> tuple[str, ...]:
    """Return all curriculum notebook identifiers in order.

    Returns
    -------
    tuple[str, ...]
        Ordered notebook identifiers.

    """
    return tuple(row.notebook_id for row in _CANONICAL_CURRICULUM)


def get_curriculum_notebook(notebook_id: str) -> CurriculumNotebookRow:
    """Return one curriculum row or raise for blank/unknown identifiers.

    Parameters
    ----------
    notebook_id
        Catalogue notebook key.

    Returns
    -------
    CurriculumNotebookRow
        Matching row.

    Raises
    ------
    ValueError
        If ``notebook_id`` is blank or unknown (fail closed).

    """
    if not notebook_id or not str(notebook_id).strip():
        raise ValueError("notebook_id must be a non-empty string")
    key = str(notebook_id).strip()
    try:
        return _NOTEBOOK_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown notebook_id {key!r}; refuse invent-green notebook "
            f"curriculum claim (known_count={len(_NOTEBOOK_BY_ID)})"
        ) from exc


def iter_curriculum_notebooks(
    *,
    runtime_class: RuntimeClass | None = None,
) -> tuple[CurriculumNotebookRow, ...]:
    """Return filtered curriculum rows in stable order.

    Parameters
    ----------
    runtime_class
        Optional runtime-class filter.

    Returns
    -------
    tuple[CurriculumNotebookRow, ...]
        Matching rows.

    """
    rows: Sequence[CurriculumNotebookRow] = _CANONICAL_CURRICULUM
    if runtime_class is not None:
        rows = tuple(row for row in rows if row.runtime_class == runtime_class)
    return tuple(rows)


def decide_differentiable_curriculum_path(
    *,
    request_hardware_execution: bool = False,
    request_full_archive_conversion: bool = False,
) -> PathEligibilityDecision:
    """Decide whether a differentiable curriculum path may proceed.

    Parameters
    ----------
    request_hardware_execution
        When true, refuse invent-green QPU notebooks.
    request_full_archive_conversion
        When true, refuse invent-green conversion of all historical notebooks.

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused decision with blockers.

    """
    blockers: list[str] = []
    if request_hardware_execution:
        blockers.append(
            "live QPU / hardware_execution notebooks refused on notebook "
            "curriculum registry (hardware_execution must remain false)"
        )
    if request_full_archive_conversion:
        blockers.append(
            "full historical notebook archive conversion refused "
            "(core-six curriculum only; archive remains research lane)"
        )
    if blockers:
        unique = tuple(dict.fromkeys(item for item in blockers if item.strip()))
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="differentiable notebook curriculum refusal: " + "; ".join(unique),
            blockers=unique,
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            "differentiable notebook curriculum path allowed for the core-six CPU set "
            "(hardware_execution=false; no full-archive conversion claim)"
        ),
        blockers=(),
    )


def resolve_curriculum_directory(repo_root: str | Path | None = None) -> Path:
    """Resolve the curriculum directory under a repository root.

    Parameters
    ----------
    repo_root
        Optional repository root (default: package parents to repo root).

    Returns
    -------
    pathlib.Path
        Absolute path to ``notebooks/differentiable``.

    """
    if repo_root is None:
        # src/scpn_quantum_control/this_file -> repo root is parents[2]
        base = Path(__file__).resolve().parents[2]
    else:
        base = Path(repo_root)
    return (base / _CURRICULUM_DIR_REL).resolve()


def materialise_curriculum_probe(
    *,
    repo_root: str | Path | None = None,
) -> MaterialisedCurriculumProbe:
    """Materialise curriculum catalogue primary observables.

    Parameters
    ----------
    repo_root
        Optional repository root for path existence checks.

    Returns
    -------
    MaterialisedCurriculumProbe
        Non-empty ids, counts, and hardware_execution honesty fields.

    Raises
    ------
    ValueError
        If path is refused or catalogue invariants fail.

    """
    decision = decide_differentiable_curriculum_path()
    if not decision.allowed:
        raise ValueError(f"curriculum probe refused: {decision.reason}")

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    missing = 0
    for row in _CANONICAL_CURRICULUM:
        if not (root / row.relative_path).is_file():
            missing += 1
    ids = list_curriculum_notebook_ids()
    return MaterialisedCurriculumProbe(
        notebook_ids=ids,
        notebook_count=len(ids),
        hardware_execution_any=any(row.hardware_execution for row in _CANONICAL_CURRICULUM),
        default_notebook_id="01_parameter_shift_kuramoto_xy",
        missing_path_count=missing,
    )


def map_differentiable_curriculum_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return the public API map for the differentiable curriculum owner.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    return (
        {
            "module_path": "scpn_quantum_control.differentiable_notebook_curriculum",
            "role": "differentiable_notebook_curriculum_registry",
            "support_posture": "curriculum_manifest",
            "notebook_ids": list(list_curriculum_notebook_ids()),
            "curriculum_dir": _CURRICULUM_DIR_REL,
            "hardware_execution": False,
            "claim_boundary": DIFFERENTIABLE_NOTEBOOK_CURRICULUM_CLAIM_BOUNDARY,
        },
    )


def build_differentiable_curriculum_registry() -> dict[str, object]:
    """Build the full serialisable differentiable curriculum registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with curriculum notebooks (no blanks).

    """
    notebooks = [row.to_dict() for row in _CANONICAL_CURRICULUM]
    return {
        "schema": DIFFERENTIABLE_NOTEBOOK_CURRICULUM_SCHEMA,
        "claim_boundary": DIFFERENTIABLE_NOTEBOOK_CURRICULUM_CLAIM_BOUNDARY,
        "notebook_count": len(notebooks),
        "blank_entry_count": 0,
        "default_notebook_id": "01_parameter_shift_kuramoto_xy",
        "curriculum_dir": _CURRICULUM_DIR_REL,
        "hardware_execution_policy": False,
        "public_surfaces": list(map_differentiable_curriculum_public_surfaces()),
        "notebooks": notebooks,
        "policy_note": (
            "Differentiable notebook curriculum catalogue only; core-six set under "
            "notebooks/differentiable; hardware_execution=false; long-form archive "
            "conversion, a full nbclient CI matrix, and framework companion notebooks "
            "remain outside this bounded registry; no invent-green live QPU notebooks."
        ),
    }


def assert_differentiable_curriculum_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers curriculum without blanks or invent-QPU.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_differentiable_curriculum_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green hardware flags appear.

    """
    registry = dict(payload) if payload is not None else build_differentiable_curriculum_registry()
    if registry.get("schema") != DIFFERENTIABLE_NOTEBOOK_CURRICULUM_SCHEMA:
        raise ValueError("unexpected differentiable notebook curriculum schema")
    if registry.get("claim_boundary") != DIFFERENTIABLE_NOTEBOOK_CURRICULUM_CLAIM_BOUNDARY:
        raise ValueError("unexpected differentiable notebook curriculum claim boundary")
    notebooks = registry.get("notebooks")
    if not isinstance(notebooks, list) or not notebooks:
        raise ValueError(
            "differentiable curriculum registry must contain a non-empty notebooks list"
        )
    seen: set[str] = set()
    blank = 0
    default_found = False
    for index, row in enumerate(notebooks):
        if not isinstance(row, Mapping):
            raise ValueError(f"notebook row {index} must be a mapping")
        notebook_id = row.get("notebook_id")
        runtime_class = row.get("runtime_class")
        hardware = row.get("hardware_execution")
        relative_path = row.get("relative_path")
        if not notebook_id or not str(notebook_id).strip():
            blank += 1
            continue
        nid = str(notebook_id).strip()
        if nid in seen:
            raise ValueError(f"duplicate notebook_id in registry: {nid!r}")
        seen.add(nid)
        if nid == "01_parameter_shift_kuramoto_xy":
            default_found = True
        if runtime_class not in {
            "cpu_local_fast",
            "cpu_optional_framework",
            "ci_light",
        }:
            blank += 1
            continue
        if not relative_path or not str(relative_path).strip():
            raise ValueError(f"notebook {nid!r} must have relative_path")
        if hardware is True:
            raise ValueError(
                f"notebook {nid!r} invent-green hardware_execution: curriculum "
                "rows must set hardware_execution=False"
            )
    if blank:
        raise ValueError(
            f"differentiable curriculum registry has {blank} blank or invalid entries"
        )
    if not default_found:
        raise ValueError(
            "differentiable curriculum registry missing 01_parameter_shift_kuramoto_xy"
        )
    expected = set(list_curriculum_notebook_ids())
    if seen != expected:
        raise ValueError(
            f"registry notebook set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    notebook_count = registry.get("notebook_count", -1)
    if not isinstance(notebook_count, int) or notebook_count != len(notebooks):
        raise ValueError("notebook_count does not match notebooks list length")
    hw_policy = registry.get("hardware_execution_policy", True)
    if hw_policy is not False:
        raise ValueError("hardware_execution_policy must be False")
    return registry


__all__ = [
    "DIFFERENTIABLE_NOTEBOOK_CURRICULUM_CLAIM_BOUNDARY",
    "DIFFERENTIABLE_NOTEBOOK_CURRICULUM_SCHEMA",
    "CurriculumNotebookRow",
    "MaterialisedCurriculumProbe",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "RuntimeClass",
    "assert_differentiable_curriculum_integrity",
    "build_differentiable_curriculum_registry",
    "decide_differentiable_curriculum_path",
    "get_curriculum_notebook",
    "iter_curriculum_notebooks",
    "list_curriculum_notebook_ids",
    "map_differentiable_curriculum_public_surfaces",
    "materialise_curriculum_probe",
    "resolve_curriculum_directory",
]
