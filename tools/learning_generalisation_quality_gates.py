# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — learning-generalisation quality gates
"""Build strict typing, documentation and coverage gates for training and its evidence consumers."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from os import devnull
from pathlib import Path

import coverage

Gate = tuple[str, list[str]]
LEARNING_GENERALISATION_SOURCES = [
    "src/scpn_quantum_control/qsnn/training.py",
    "src/scpn_quantum_control/phase/gradient_descent.py",
    "src/scpn_quantum_control/ml_examples/contracts.py",
    "src/scpn_quantum_control/ml_examples/qsnn_convergence.py",
    "src/scpn_quantum_control/ml_examples/qnn_convergence.py",
    "src/scpn_quantum_control/ml_examples/qgnn_convergence.py",
    "src/scpn_quantum_control/ml_examples/suite.py",
    "src/scpn_quantum_control/ml_examples/evidence.py",
    "src/scpn_quantum_control/ml_examples/decoding.py",
    "oscillatools/src/oscillatools/neural_operator.py",
    "src/scpn_quantum_control/surrogates/_identity.py",
    "src/scpn_quantum_control/surrogates/models.py",
    "src/scpn_quantum_control/surrogates/train.py",
    "src/scpn_quantum_control/surrogates/fidelity.py",
    "src/scpn_quantum_control/forecasting/multimodal_forecaster.py",
]
LEARNING_GENERALISATION_CONFORMANCE_TEST = "tests/test_learning_rng_and_generalisation.py"
LEARNING_GENERALISATION_COVERAGE_COHORT = [
    LEARNING_GENERALISATION_CONFORMANCE_TEST,
    "tests/test_qsnn_training.py",
    "tests/test_phase_gradient_descent.py",
    "tests/test_phase_optimizer_audit.py",
    "tests/test_qsnn_training_validation.py",
    "tests/test_qsnn_native_parameter_shift.py",
    "tests/test_ml_convergence_evidence.py",
    "tests/test_ml_convergence_contracts.py",
    "tests/test_ml_convergence_decoding.py",
    "tests/test_ml_convergence_qsnn.py",
    "tests/test_ml_convergence_qnn.py",
    "tests/test_ml_convergence_qgnn.py",
    "oscillatools/tests/test_kuramoto_neural_operator.py",
    "tests/neural_operator_rng_custody.py",
    "tests/test_surrogate_identity.py",
    "tests/test_surrogate_models.py",
    "tests/test_surrogate_train.py",
    "tests/test_surrogate_fidelity.py",
    "tests/test_surrogate_hybrid.py",
    "tests/test_surrogate_report.py",
    "tests/test_multimodal_forecaster.py",
    "tests/test_dense_allocation_admission.py",
    "tests/test_learning_generalisation_quality_gate.py",
]
LEARNING_GENERALISATION_TYPING_RATCHET = [
    *LEARNING_GENERALISATION_SOURCES,
    LEARNING_GENERALISATION_CONFORMANCE_TEST,
    "tests/test_phase_gradient_descent.py",
    "tests/test_qsnn_training.py",
    "tests/test_ml_convergence_evidence.py",
    "tests/test_ml_convergence_contracts.py",
    "tests/test_ml_convergence_decoding.py",
    "tests/test_ml_convergence_qsnn.py",
    "tests/test_ml_convergence_qnn.py",
    "tests/ml_convergence_qnn_absence.py",
    "tests/neural_operator_rng_custody.py",
    "tests/test_ml_convergence_qgnn.py",
    "tests/test_surrogate_identity.py",
    "tests/test_surrogate_models.py",
    "tests/test_surrogate_train.py",
    "tests/test_surrogate_fidelity.py",
    "tests/test_multimodal_forecaster.py",
    "tools/learning_generalisation_quality_gates.py",
    "tests/test_learning_generalisation_quality_gate.py",
]
LEARNING_GENERALISATION_COVERAGE_INCLUDE = ",".join(
    "*/" + path for path in LEARNING_GENERALISATION_TYPING_RATCHET
)
LEARNING_GENERALISATION_DOCSTRING_RATCHET = [*LEARNING_GENERALISATION_TYPING_RATCHET]
LEARNING_GENERALISATION_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-learning-generalisation-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates.

    Parameters
    ----------
    python
        Interpreter that runs the gates.

    Returns
    -------
    list[Gate]
        Named commands.

    """
    return [
        (
            "mypy-strict-learning-generalisation-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *LEARNING_GENERALISATION_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D learning-generalisation quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D413,D417,D420",
                "--config",
                "lint.explicit-preview-rules = true",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *LEARNING_GENERALISATION_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build the focused run and the exact coverage threshold.

    Parameters
    ----------
    python
        Interpreter that runs the gates.

    Returns
    -------
    list[Gate]
        Named commands.

    """
    return [
        (
            "learning-generalisation focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={LEARNING_GENERALISATION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *LEARNING_GENERALISATION_COVERAGE_COHORT,
            ],
        ),
        (
            "learning-generalisation SDK-absent coverage",
            [
                python,
                "-c",
                "import sys; from tools.learning_generalisation_quality_gates "
                "import run_sdk_absent_coverage; "
                "raise SystemExit(run_sdk_absent_coverage(sys.argv[1]))",
                LEARNING_GENERALISATION_COVERAGE_DATA_FILE,
            ],
        ),
        (
            "learning-generalisation exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={LEARNING_GENERALISATION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={LEARNING_GENERALISATION_COVERAGE_INCLUDE}",
            ],
        ),
    ]


def run_sdk_absent_coverage(data_file: str) -> int:
    """Append real QNN dependency-absence coverage with the same Python ABI.

    Parameters
    ----------
    data_file
        Coverage database that receives the additional branch observations.

    Returns
    -------
    int
        Actual subprocess exit status; missing libraries or failed tests remain
        failures. The native QNN trainer executes in the isolated process.

    Raises
    ------
    OSError
        If the temporary package view cannot be created or Python cannot run.

    Notes
    -----
    The temporary view references installed base libraries and test tooling
    without exposing optional SDK distributions. No package is installed,
    copied, uninstalled, mocked or changed. Python's ``-S`` keeps the original
    site directory and framework overlays off the child search path.

    """
    root = Path(__file__).resolve().parents[1]
    site = Path(coverage.__file__).parent.parent
    excluded = {"__pycache__", "jax", "jaxlib", "torch", "torchgen", "functorch", "tensorflow"}
    prefixes = ("jax-", "jaxlib-", "torch-", "tensorflow-", "tensorflow_cpu-")
    with tempfile.TemporaryDirectory(prefix="scpn-learning-sdk-absent-") as directory:
        view = Path(directory)
        for source in site.iterdir():
            if source.name in excluded or source.name.startswith(prefixes):
                continue
            (view / source.name).symlink_to(source, target_is_directory=source.is_dir())
        environment = os.environ | {
            "PYTHONPATH": os.pathsep.join(
                str(path) for path in (view, root / "src", root / "oscillatools/src", root)
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        return subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data_file}",
                "--branch",
                "--append",
                "--context=sdk-absent",
                "-m",
                "pytest",
                "-q",
                "tests/test_ml_convergence_qnn.py::test_qnn_framework_rows_execute_installed_adapters_and_expose_absence",
                "tests/ml_convergence_qnn_absence.py",
            ],
            cwd=root,
            env=environment,
            check=False,
        ).returncode


__all__ = [
    "LEARNING_GENERALISATION_CONFORMANCE_TEST",
    "LEARNING_GENERALISATION_COVERAGE_COHORT",
    "LEARNING_GENERALISATION_COVERAGE_DATA_FILE",
    "LEARNING_GENERALISATION_COVERAGE_INCLUDE",
    "LEARNING_GENERALISATION_DOCSTRING_RATCHET",
    "LEARNING_GENERALISATION_SOURCES",
    "LEARNING_GENERALISATION_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
    "run_sdk_absent_coverage",
]
