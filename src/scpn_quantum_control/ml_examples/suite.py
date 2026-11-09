# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — convergence-example unified ML convergence suite
"""Compose the three existing ML training surfaces into one evidence suite."""

from __future__ import annotations

from .contracts import ConvergenceSuiteEvidence, ModelFamily
from .qgnn_convergence import qgnn_framework_rows, run_qgnn_convergence_example
from .qnn_convergence import run_qnn_convergence_example, run_qnn_framework_rows
from .qsnn_convergence import qsnn_framework_rows, run_qsnn_convergence_example


def run_ml_convergence_suite(
    *,
    required_qnn_frameworks: tuple[str, ...] = (),
) -> ConvergenceSuiteEvidence:
    """Run deterministic QNN, QGNN, and QSNN examples through public APIs.

    Parameters
    ----------
    required_qnn_frameworks
        Framework adapters whose absence fails their declared evidence gate.

    Returns
    -------
    ConvergenceSuiteEvidence
        Actual training certificates, observed framework outcomes and learning
        pointers. Failed runs remain failed in their native matrix rows.

    Raises
    ------
    ValueError
        If a required QNN framework is not registered.

    """
    certificates = (
        run_qnn_convergence_example(),
        run_qgnn_convergence_example(),
        run_qsnn_convergence_example(),
    )
    qnn, qgnn, qsnn = certificates
    framework_rows = (
        *run_qnn_framework_rows(required_frameworks=required_qnn_frameworks, certificate=qnn),
        *qgnn_framework_rows(certificate=qgnn),
        *qsnn_framework_rows(certificate=qsnn),
    )
    return ConvergenceSuiteEvidence(
        certificates=certificates,
        framework_rows=framework_rows,
        notebook_pointers=(
            (ModelFamily.QNN, "CLI gallery: scripts/run_ml_convergence_examples.py"),
            (ModelFamily.QGNN, "docs/quantum_graph_neural_network.md"),
            (ModelFamily.QSNN, "notebooks/10_qsnn_training.ipynb"),
        ),
    )


__all__ = ["run_ml_convergence_suite"]
