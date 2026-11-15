# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-owned learning, forecasting and domain-proxy templates
"""Build one bounded template per learning, forecasting and domain-proxy owner.

Every function reads public owner APIs and committed evidence only and returns
JSON-ready source values. Synthetic and curated inputs keep their labels,
domain metrics stay proxies under their honesty kits, and every refusal is the
owner's own message. Nothing here fits, scores or simulates on its own account.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..applications.app_plugins import (
    FEPApplicationPlugin,
    get_application_plugin,
    load_application_dataset,
)
from ..applications.honesty_kits import (
    APPLICATION_HONESTY_CLAIM_BOUNDARY,
    get_domain_application_honesty_kit_for_dataset,
)
from ..forecasting.multimodal_forecaster import fit_multimodal_ridge_forecaster
from ..forecasting.synthetic_multimodal import (
    SyntheticMultimodalConfig,
    SyntheticMultimodalDataset,
    generate_synthetic_multimodal_dataset,
)
from ..jax_nqs_baseline_product import load_jax_nqs_baseline_evidence
from ..qsnn.qlayer import QuantumDenseLayer
from ..qsnn.training import QSNNTrainer, QSNNTrainingCheckpoint

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
FORECASTING_EVIDENCE: Final[Path] = (
    REPOSITORY_ROOT / "data/multimodal_forecasting/multimodal_forecasting_evidence.json"
)
NQS_EVIDENCE: Final[Path] = (
    REPOSITORY_ROOT / "data/jax_nqs_baseline/jax_nqs_baseline_evidence.json"
)
FORECASTING_CONFIG: Final[SyntheticMultimodalConfig] = SyntheticMultimodalConfig(
    train_samples=64,
    calibration_samples=24,
    test_samples=32,
    history_steps=12,
    horizon_steps=4,
    missing_fraction=0.20,
    seed=3701,
)
DOMAIN_PROXY_PLUGINS: Final[tuple[tuple[str, str], ...]] = (
    ("eeg_alpha", "eeg_alpha_plv_8ch"),
    ("power_grid_ieee5", "ieee5bus_power_grid"),
    ("plasma_iter_mhd", "iter_mhd_8mode"),
)
QSNN_LAYER_SEED: Final[int] = 11
QSNN_LEARNING_RATE: Final[float] = 0.2


def _section(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"evidence field {name!r} must be a JSON object")
    return value


def _pick(payload: Mapping[str, object], names: tuple[str, ...]) -> dict[str, object]:
    missing = [name for name in names if name not in payload]
    if missing:
        raise ValueError(f"evidence lacks fields {missing}")
    return {name: payload[name] for name in names}


def _relative(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"evidence must be committed under the repository: {path}")
    return resolved.relative_to(REPOSITORY_ROOT).as_posix()


_SPLIT_FIELDS: Final = ("split", "samples", "content_digest", "missing_fraction", "domain_counts")
_TEST_FIELDS: Final = (
    "samples",
    "wrapped_mse",
    "wrapped_mae",
    "persistence_wrapped_mse",
    "lower_mse_than_persistence",
)
_INTERVAL_FIELDS: Final = (
    "target_coverage",
    "sample_coverage",
    "value_coverage",
    "mean_interval_width",
    "claim_boundary",
)


def _held_out_custody(
    evidence: Mapping[str, object], dataset: SyntheticMultimodalDataset
) -> dict[str, object]:
    committed = _section(evidence, "dataset")
    if dataset.content_digest() != committed.get("content_digest"):
        raise ValueError("committed forecasting evidence no longer matches its generator")
    model = _section(evidence, "model")
    if model.get("training_batch_digest") != _section(committed, "train").get("content_digest"):
        raise ValueError("the forecasting model was not fitted on the training split")
    try:
        fit_multimodal_ridge_forecaster(dataset.test)
    except ValueError as refusal:
        fit_refusal = str(refusal)
    else:
        raise ValueError("the forecasting owner accepted a held-out batch for fitting")
    training = set(dataset.train.sample_ids)
    held_out = set(dataset.calibration.sample_ids) | set(dataset.test.sample_ids)
    return {
        "training_samples": len(training),
        "held_out_samples": len(held_out),
        "overlapping_samples": len(held_out & training),
        "model_training_batch_digest": model["training_batch_digest"],
        "fit_refusal": fit_refusal,
    }


def forecasting_template(evidence_path: Path = FORECASTING_EVIDENCE) -> dict[str, object]:
    """Bind the committed forecasting evidence to its generator and its split custody.

    Parameters
    ----------
    evidence_path
        Committed multimodal forecasting evidence under the repository.

    Returns
    -------
    dict[str, object]
        Dataset provenance and splits, held-out custody, the classical reference
        model, held-out accuracy against persistence and interval coverage.

    Raises
    ------
    ValueError
        If the evidence no longer matches its generator, the model was fitted on
        another split, the owner accepts fitting a held-out batch, a projected
        field is missing, or the evidence lies outside the repository.

    """
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    dataset = generate_synthetic_multimodal_dataset(FORECASTING_CONFIG)
    held_out = _held_out_custody(evidence, dataset)
    committed = _section(evidence, "dataset")
    splits = [
        _pick(_section(committed, name), _SPLIT_FIELDS)
        for name in ("train", "calibration", "test")
    ]
    projected = {
        "model": _pick(_section(evidence, "model"), ("kind", "ridge", "model_digest")),
        "test": _pick(_section(evidence, "test_accuracy"), _TEST_FIELDS),
        "interval": _pick(_section(evidence, "interval_coverage"), _INTERVAL_FIELDS),
    }
    return {
        "owner": "scpn_quantum_control.forecasting",
        **_pick(evidence, ("claim_boundary",)),
        "dataset": {
            **_pick(committed, ("source", "content_digest")),
            "licence": "AGPL-3.0-or-later; generated by the repository's own simulator",
            "seed": FORECASTING_CONFIG.seed,
            "splits": splits,
        },
        "held_out": held_out,
        **projected,
        "evidence_path": _relative(evidence_path),
    }


def _qsnn_data(seed: int, count: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    features = rng.uniform(0.2, 0.8, size=(count, 2))
    return features, (features[:, :1] > 0.5).astype(np.float64)


def _qsnn_trainer() -> QSNNTrainer:
    return QSNNTrainer(QuantumDenseLayer(1, 2, seed=QSNN_LAYER_SEED), lr=QSNN_LEARNING_RATE)


def _checkpoint_round_trip(
    features: NDArray[np.float64], targets: NDArray[np.float64]
) -> dict[str, object]:
    uninterrupted = _qsnn_trainer()
    uninterrupted.train(features, targets, epochs=3)
    first = _qsnn_trainer()
    first.train(features, targets, epochs=2)
    saved = QSNNTrainingCheckpoint.from_json(first.checkpoint().to_json())
    resumed = _qsnn_trainer()
    resumed.restore(saved)
    resumed.train(features, targets, epochs=1)
    return {
        "schema": saved.schema,
        "randomness": saved.randomness,
        "learning_rate": saved.learning_rate,
        "saved_after_epochs": saved.completed_epochs,
        "resumed_matches_uninterrupted": resumed.checkpoint() == uninterrupted.checkpoint(),
    }


def _qsnn_descent(
    features: NDArray[np.float64], targets: NDArray[np.float64]
) -> dict[str, object]:
    descent = _qsnn_trainer().train_with_parameter_shift_descent(features, targets, max_steps=5)
    training = descent.training
    return {
        "optimiser": {
            "method": training.method,
            "learning_rate": QSNN_LEARNING_RATE,
            "shift_terms": training.shift_terms,
            "backend": training.backend_plan.backend,
            "evaluations": training.evaluations,
        },
        "steps": [
            {"index": step.index, "loss": step.value, "gradient_norm": step.gradient_norm}
            for step in training.steps
        ],
    }


def _qsnn_held_out(
    features: NDArray[np.float64], targets: NDArray[np.float64]
) -> dict[str, object]:
    held_features, held_targets = _qsnn_data(2, 4)
    trainer = _qsnn_trainer()
    trainer.train(features, targets, epochs=2)
    evaluation = trainer.evaluate(held_features, held_targets)
    try:
        trainer.evaluate(features[:2], targets[:2])
    except ValueError as refusal:
        overlap_refusal = str(refusal)
    else:
        raise ValueError("the QSNN owner scored samples that were used in training")
    return {
        "samples": evaluation.n_samples,
        "out_of_distribution": evaluation.out_of_distribution_count,
        "in_distribution_loss": evaluation.in_distribution_loss,
        "claim_boundary": evaluation.claim_boundary,
        "overlap_refusal": overlap_refusal,
    }


def qsnn_template() -> dict[str, object]:
    """Train the seeded QSNN layer and record descent, checkpoint and held-out evidence.

    Returns
    -------
    dict[str, object]
        Layer and data provenance, per-step loss and gradient norm, an exact
        checkpoint-resume comparison, the held-out evaluation and the owner's
        refusal to score samples that were used in training.

    Raises
    ------
    ValueError
        If the owner scores held-out samples that overlap the training rows.

    """
    features, targets = _qsnn_data(1, 6)
    return {
        "owner": "scpn_quantum_control.qsnn.training",
        "layer": {"n_neurons": 1, "n_inputs": 2, "seed": QSNN_LAYER_SEED},
        "dataset": {
            "source": "Synthetic features uniform on [0.2, 0.8]^2; target 1 when the first exceeds 0.5",
            "licence": "AGPL-3.0-or-later; generated by this producer",
            "training_seed": 1,
            "training_samples": int(features.shape[0]),
            "held_out_seed": 2,
            "held_out_samples": 4,
        },
        **_qsnn_descent(features, targets),
        "checkpoint": _checkpoint_round_trip(features, targets),
        "held_out": _qsnn_held_out(features, targets),
    }


def nqs_template(evidence_path: Path = NQS_EVIDENCE) -> dict[str, object]:
    """Show the committed JAX NQS baseline, or state plainly that it is absent.

    Parameters
    ----------
    evidence_path
        Committed evidence written by the JAX NQS baseline owner.

    Returns
    -------
    dict[str, object]
        The verified request, environment and exact-reference comparison, or an
        ``unavailable`` status when no evidence is committed.

    """
    location = {"owner": "scpn_quantum_control.jax_nqs_baseline_product"}
    if not evidence_path.is_file():
        return {**location, "status": "unavailable", "evidence_path": None}
    product = load_jax_nqs_baseline_evidence(evidence_path)
    comparison = product.comparison
    return {
        **location,
        "status": "available",
        "evidence_path": _relative(evidence_path),
        "evidence_sha256": product.evidence_sha256,
        "claim_boundary": product.claim_boundary,
        "request": {
            "qubits": product.request.n_qubits,
            "hidden_units": product.request.n_hidden,
            "learning_rate": product.request.learning_rate,
            "iterations": product.request.n_iterations,
            "seed": product.request.seed,
        },
        "environment": _pick(
            product.environment.to_dict(), ("jax_version", "backend", "numeric_posture")
        ),
        "comparison": {
            "exact_ground_energy": comparison.exact_ground_energy,
            "variational_energy": comparison.variational_energy,
            "relative_error": comparison.relative_error,
            "within_declared_tolerance": comparison.within_declared_tolerance,
            "variational_upper_bound_respected": comparison.variational_upper_bound_respected,
            "energy_history": list(comparison.energy_history),
        },
    }


def domain_proxy_templates() -> list[dict[str, object]]:
    """Run each curated domain benchmark and attach its honesty kit.

    Returns
    -------
    list[dict[str, object]]
        One row per EEG, power-grid and plasma benchmark: its proxy metrics,
        provenance, the kit's allowed uses, caveats and forbidden claims, and
        the explicit absence of any measured physical outcome.

    """
    rows: list[dict[str, object]] = []
    for plugin_name, dataset_id in DOMAIN_PROXY_PLUGINS:
        result = get_application_plugin(plugin_name).benchmark_dataset(dataset_id)
        metadata = load_application_dataset(plugin_name, dataset_id).metadata
        kit = get_domain_application_honesty_kit_for_dataset(dataset_id)
        rows.append(
            {
                "dataset_id": dataset_id,
                "domain": result.domain,
                "plugin": plugin_name,
                "licence_note": metadata["licence_note"],
                "public_reference": metadata["public_reference"],
                "proxy_metrics": dict(result.metrics),
                "physical_outcome_in_custody": False,
                "kit": {
                    "kit_id": kit.kit_id,
                    "support_status": kit.support_status.value,
                    "publication_safe": kit.publication_safe,
                    "allowed_uses": list(kit.allowed_uses),
                    "caveats": list(kit.caveats),
                    "claims_forbidden": list(kit.claims_forbidden),
                },
            }
        )
    return rows


def _fep_refusal(
    plugin: FEPApplicationPlugin, case: str, sigma: NDArray[np.float64]
) -> dict[str, str]:
    try:
        message = plugin.covariance_refusal(sigma)
    except ValueError as accepted:
        raise ValueError(f"the free-energy owner accepted the {case}") from accepted
    return {"case": case, "message": message}


def fep_template() -> dict[str, object]:
    """Evaluate the curated free-energy workflow and record the owner's covariance refusals.

    Returns
    -------
    dict[str, object]
        The workflow benchmark's free-energy metrics and provenance, the
        application claim boundary, and the owner's refusals of a negative
        covariance and of a covariance with the wrong shape.

    Raises
    ------
    TypeError
        If the registered ``friston_fep`` plugin is not the packaged workflow.
    ValueError
        If the owner accepts either invalid covariance.

    """
    dataset_id = "friston_fep_6node"
    plugin = get_application_plugin("friston_fep")
    if not isinstance(plugin, FEPApplicationPlugin):
        raise TypeError("friston_fep must be the packaged free-energy workflow plugin")
    result = plugin.benchmark_dataset(dataset_id)
    metadata = load_application_dataset("friston_fep", dataset_id).metadata
    return {
        "owner": f"{plugin.name} application plugin",
        "backend": result.backend,
        "dataset_id": dataset_id,
        "licence_note": metadata["licence_note"],
        "public_reference": metadata["public_reference"],
        "metrics": dict(result.metrics),
        "claim_boundary": APPLICATION_HONESTY_CLAIM_BOUNDARY,
        "refusals": [
            _fep_refusal(plugin, "negative covariance", np.array([[-1.0]])),
            _fep_refusal(plugin, "covariance of the wrong shape", np.eye(2)),
        ],
    }


__all__ = [
    "DOMAIN_PROXY_PLUGINS",
    "FORECASTING_CONFIG",
    "FORECASTING_EVIDENCE",
    "NQS_EVIDENCE",
    "domain_proxy_templates",
    "fep_template",
    "forecasting_template",
    "nqs_template",
    "qsnn_template",
]
