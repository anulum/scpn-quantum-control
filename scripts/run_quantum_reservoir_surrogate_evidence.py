# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-reservoir deterministic evidence runner
"""Regenerate local exact-statevector QRC and surrogate evidence."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import numpy as np

from scpn_quantum_control.applications.quantum_reservoir_product import (
    ReservoirLinearObjective,
    ReservoirTaskKind,
    certify_reservoir_training,
    generate_synthetic_reservoir_task,
)
from scpn_quantum_control.surrogates import (
    QuantumReservoirSurrogateEvidence,
    SurrogateFidelityThresholds,
    SurrogateFitConfig,
    SurrogateSupportRow,
    certify_surrogate_fidelity,
    certify_surrogate_gradient,
    fit_gaussian_rbf_surrogate,
    propose_and_validate_surrogate_step,
    write_quantum_reservoir_surrogate_evidence,
)

DEFAULT_JSON = Path("data/quantum_reservoir_surrogates/quantum_reservoir_evidence.json")
DEFAULT_MARKDOWN = Path("data/quantum_reservoir_surrogates/quantum_reservoir_evidence.md")


def _stale_evidence_message(path: Path, actual: str, expected: str) -> str:
    """Return a bounded unified diff for one stale evidence artifact."""
    difference = list(
        difflib.unified_diff(
            actual.splitlines(),
            expected.splitlines(),
            fromfile=f"committed/{path.as_posix()}",
            tofile=f"generated/{path.as_posix()}",
            lineterm="",
            n=2,
        )
    )
    excerpt = "\n".join(difference[:80])
    if len(difference) > 80:
        excerpt += f"\n... {len(difference) - 80} additional diff lines omitted"
    return f"stale or missing evidence: {path}\n{excerpt}".rstrip()


def build_evidence() -> QuantumReservoirSurrogateEvidence:
    """Build the complete deterministic quantum-reservoir evidence bundle."""
    coupling = np.array([[0.0, 0.65], [0.65, 0.0]], dtype=np.float64)
    frequencies = np.array([0.15, -0.1], dtype=np.float64)
    reservoir_certificates = tuple(
        certify_reservoir_training(
            generate_synthetic_reservoir_task(
                task_kind,
                n_train=18,
                n_validation=8,
                seed=4139971,
            ),
            coupling,
            omega=frequencies,
            alpha=0.1,
            max_weight=1,
            t=0.8,
            seed=4139971,
        )
        for task_kind in (ReservoirTaskKind.CLASSIFICATION, ReservoirTaskKind.FORECAST)
    )

    objective = ReservoirLinearObjective(
        K=coupling,
        omega=frequencies,
        feature_labels=("IZ", "ZI", "XX"),
        feature_weights=(0.5, -0.35, 0.15),
        t=0.8,
        max_weight=2,
    )
    training_axis = np.linspace(0.1, 0.9, 5, dtype=np.float64)
    training_inputs = np.array(
        [(left, right) for left in training_axis for right in training_axis],
        dtype=np.float64,
    )
    training_targets = np.array(
        [objective(parameters) for parameters in training_inputs],
        dtype=np.float64,
    )
    model = fit_gaussian_rbf_surrogate(
        training_inputs,
        training_targets,
        config=SurrogateFitConfig(regularisation=1.0e-8),
    )

    validation_axis = np.linspace(0.18, 0.82, 4, dtype=np.float64)
    validation_inputs = np.array(
        [(left, right) for left in validation_axis for right in validation_axis],
        dtype=np.float64,
    )
    validation_targets = np.array(
        [objective(parameters) for parameters in validation_inputs],
        dtype=np.float64,
    )
    value_fidelity = certify_surrogate_fidelity(
        model,
        validation_inputs,
        validation_targets,
        thresholds=SurrogateFidelityThresholds(
            max_rmse=0.01,
            max_absolute_error=0.025,
            min_r_squared=0.98,
        ),
    )
    gradient_fidelity = certify_surrogate_gradient(
        model,
        validation_inputs[[0, 5, 10, 15]],
        objective,
        finite_difference_step=1.0e-5,
        max_absolute_error=0.02,
    )
    exact_validated_proposal = propose_and_validate_surrogate_step(
        model,
        np.array([0.34, 0.68], dtype=np.float64),
        objective,
        value_fidelity,
        learning_rate=0.08,
        max_step_norm=0.05,
    )
    support_rows = (
        SurrogateSupportRow(
            surface="qrc_heldout_certificates",
            status="local_exact_supported",
            evidence="Two disjoint synthetic task certificates.",
            boundary="Small-system exact statevector only.",
        ),
        SurrogateSupportRow(
            surface="matched_esn_comparator",
            status="bounded_supported",
            evidence="QRC and ESN use equal readout feature counts.",
            boundary="No winner or advantage assumption.",
        ),
        SurrogateSupportRow(
            surface="gaussian_rbf_value_fidelity",
            status="local_exact_supported",
            evidence="Disjoint held-out values pass frozen thresholds.",
            boundary="One frozen two-parameter simulator objective.",
        ),
        SurrogateSupportRow(
            surface="analytic_rbf_gradient_fidelity",
            status="local_exact_supported",
            evidence="Analytic RBF gradients match exact central differences.",
            boundary="Finite-difference reference, not hardware gradients.",
        ),
        SurrogateSupportRow(
            surface="codesign_exact_validated_proposal",
            status="bounded_supported",
            evidence="Surrogate proposal followed by exact local objective query.",
            boundary="ControllerProposal remains unapplied.",
        ),
        SurrogateSupportRow(
            surface="multimodal_forecasting_adapter",
            status="blocked_dependency",
            evidence="The multimodal forecasting adapter is not implemented.",
            boundary="No invented domain adapter or operational data.",
        ),
        SurrogateSupportRow(
            surface="differentiable_notebook_curriculum",
            status="blocked_dependency",
            evidence=(
                "Differentiable notebook curriculum expansion is outside the "
                "quantum-reservoir surrogate evidence scope."
            ),
            boundary="No notebook is represented as complete.",
        ),
    )
    return QuantumReservoirSurrogateEvidence(
        reservoir_certificates=reservoir_certificates,
        surrogate_model=model,
        value_fidelity=value_fidelity,
        gradient_fidelity=gradient_fidelity,
        exact_validated_proposal=exact_validated_proposal,
        support_rows=support_rows,
    )


def main() -> int:
    """Write evidence files and print their deterministic identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    evidence = build_evidence()
    if args.check:
        expected_json = json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n"
        actual_json = args.json.read_text(encoding="utf-8") if args.json.is_file() else ""
        if actual_json != expected_json:
            raise SystemExit(_stale_evidence_message(args.json, actual_json, expected_json))
        from scpn_quantum_control.surrogates import render_quantum_reservoir_surrogate_markdown

        expected_markdown = render_quantum_reservoir_surrogate_markdown(evidence)
        actual_markdown = (
            args.markdown.read_text(encoding="utf-8") if args.markdown.is_file() else ""
        )
        if actual_markdown != expected_markdown:
            raise SystemExit(
                _stale_evidence_message(args.markdown, actual_markdown, expected_markdown)
            )
        print(
            json.dumps({"check": "passed", "content_digest": evidence.to_dict()["content_digest"]})
        )
        return 0

    json_digest, markdown_digest = write_quantum_reservoir_surrogate_evidence(
        evidence,
        json_path=args.json,
        markdown_path=args.markdown,
    )
    print(
        json.dumps(
            {
                "content_digest": evidence.to_dict()["content_digest"],
                "json_sha256": json_digest,
                "markdown_sha256": markdown_digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
