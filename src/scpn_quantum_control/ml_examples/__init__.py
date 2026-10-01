# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — convergence-example ML convergence examples
"""Public bounded QNN/QGNN/QSNN convergence evidence surface."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .contracts import (
        ML_CONVERGENCE_CLAIM_BOUNDARY,
        ML_CONVERGENCE_SCHEMA,
        ConvergenceCertificate,
        ConvergenceExampleSpec,
        ConvergenceSuiteEvidence,
        FrameworkEvidenceRow,
        FrameworkStatus,
        ModelFamily,
    )
    from .evidence import (
        evidence_payload,
        render_evidence_markdown,
        validate_ml_convergence_evidence,
        write_ml_convergence_evidence,
    )
    from .qgnn_convergence import (
        qgnn_example_spec,
        qgnn_framework_rows,
        run_qgnn_convergence_example,
    )
    from .qnn_convergence import (
        qnn_example_spec,
        run_qnn_convergence_example,
        run_qnn_framework_rows,
    )
    from .qsnn_convergence import (
        qsnn_example_spec,
        qsnn_framework_rows,
        run_qsnn_convergence_example,
    )
    from .suite import run_ml_convergence_suite

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ML_CONVERGENCE_CLAIM_BOUNDARY": (
        "scpn_quantum_control.ml_examples.contracts",
        "ML_CONVERGENCE_CLAIM_BOUNDARY",
    ),
    "ML_CONVERGENCE_SCHEMA": (
        "scpn_quantum_control.ml_examples.contracts",
        "ML_CONVERGENCE_SCHEMA",
    ),
    "ConvergenceCertificate": (
        "scpn_quantum_control.ml_examples.contracts",
        "ConvergenceCertificate",
    ),
    "ConvergenceExampleSpec": (
        "scpn_quantum_control.ml_examples.contracts",
        "ConvergenceExampleSpec",
    ),
    "ConvergenceSuiteEvidence": (
        "scpn_quantum_control.ml_examples.contracts",
        "ConvergenceSuiteEvidence",
    ),
    "FrameworkEvidenceRow": ("scpn_quantum_control.ml_examples.contracts", "FrameworkEvidenceRow"),
    "FrameworkStatus": ("scpn_quantum_control.ml_examples.contracts", "FrameworkStatus"),
    "ModelFamily": ("scpn_quantum_control.ml_examples.contracts", "ModelFamily"),
    "evidence_payload": ("scpn_quantum_control.ml_examples.evidence", "evidence_payload"),
    "render_evidence_markdown": (
        "scpn_quantum_control.ml_examples.evidence",
        "render_evidence_markdown",
    ),
    "validate_ml_convergence_evidence": (
        "scpn_quantum_control.ml_examples.evidence",
        "validate_ml_convergence_evidence",
    ),
    "write_ml_convergence_evidence": (
        "scpn_quantum_control.ml_examples.evidence",
        "write_ml_convergence_evidence",
    ),
    "qgnn_example_spec": (
        "scpn_quantum_control.ml_examples.qgnn_convergence",
        "qgnn_example_spec",
    ),
    "qgnn_framework_rows": (
        "scpn_quantum_control.ml_examples.qgnn_convergence",
        "qgnn_framework_rows",
    ),
    "run_qgnn_convergence_example": (
        "scpn_quantum_control.ml_examples.qgnn_convergence",
        "run_qgnn_convergence_example",
    ),
    "qnn_example_spec": ("scpn_quantum_control.ml_examples.qnn_convergence", "qnn_example_spec"),
    "run_qnn_convergence_example": (
        "scpn_quantum_control.ml_examples.qnn_convergence",
        "run_qnn_convergence_example",
    ),
    "run_qnn_framework_rows": (
        "scpn_quantum_control.ml_examples.qnn_convergence",
        "run_qnn_framework_rows",
    ),
    "qsnn_example_spec": (
        "scpn_quantum_control.ml_examples.qsnn_convergence",
        "qsnn_example_spec",
    ),
    "qsnn_framework_rows": (
        "scpn_quantum_control.ml_examples.qsnn_convergence",
        "qsnn_framework_rows",
    ),
    "run_qsnn_convergence_example": (
        "scpn_quantum_control.ml_examples.qsnn_convergence",
        "run_qsnn_convergence_example",
    ),
    "run_ml_convergence_suite": (
        "scpn_quantum_control.ml_examples.suite",
        "run_ml_convergence_suite",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


__all__ = [
    "ML_CONVERGENCE_CLAIM_BOUNDARY",
    "ML_CONVERGENCE_SCHEMA",
    "ConvergenceCertificate",
    "ConvergenceExampleSpec",
    "ConvergenceSuiteEvidence",
    "FrameworkEvidenceRow",
    "FrameworkStatus",
    "ModelFamily",
    "evidence_payload",
    "qgnn_example_spec",
    "qgnn_framework_rows",
    "qnn_example_spec",
    "qsnn_example_spec",
    "qsnn_framework_rows",
    "render_evidence_markdown",
    "run_ml_convergence_suite",
    "run_qgnn_convergence_example",
    "run_qnn_convergence_example",
    "run_qnn_framework_rows",
    "run_qsnn_convergence_example",
    "validate_ml_convergence_evidence",
    "write_ml_convergence_evidence",
]
