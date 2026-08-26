# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — convergence-example ML convergence examples
"""Public bounded QNN/QGNN/QSNN convergence evidence surface."""

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
