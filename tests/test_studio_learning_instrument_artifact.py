# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — learning instrument custody, refusals and random-state tests
"""Exercise the learning instrument through its owners, its CLI and its refusals."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.applications.app_plugins import (
    FEPApplicationPlugin,
    get_application_plugin,
)
from scpn_quantum_control.applications.honesty_kits import (
    get_domain_application_honesty_kit_for_dataset,
)
from scpn_quantum_control.fep.variational_free_energy import variational_free_energy
from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer
from scpn_quantum_control.qsnn.training import (
    QSNN_CHECKPOINT_SCHEMA,
    QSNN_RANDOMNESS,
    QSNNEvaluation,
    QSNNTrainer,
)
from scpn_quantum_control.studio import learning_instrument_artifact as artifact
from scpn_quantum_control.studio import learning_instrument_templates as templates


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    """Build the instrument once from its owners.

    Returns
    -------
    dict
        The fresh artefact payload.

    """
    return artifact.build_learning_instrument_artifact()


def test_held_out_records_never_enter_fitting(built: dict[str, Any]) -> None:
    """Forecasting and QSNN keep held-out samples out of fitting and scoring.

    Parameters
    ----------
    built
        Fresh artefact.

    """
    forecasting = built["templates"]["forecasting"]
    held_out = forecasting["held_out"]
    assert held_out["overlapping_samples"] == 0
    assert held_out["training_samples"] == 64 and held_out["held_out_samples"] == 56
    train_split = forecasting["dataset"]["splits"][0]
    assert train_split["split"] == "train"
    assert held_out["model_training_batch_digest"] == train_split["content_digest"]
    assert held_out["fit_refusal"] == "forecaster fitting requires a train batch"
    assert forecasting["dataset"]["seed"] == 3701
    qsnn = built["templates"]["qsnn"]["held_out"]
    assert qsnn["overlap_refusal"].startswith("train/test overlap: held-out samples [0, 1]")


def test_every_template_names_its_data_provenance_and_licence(built: dict[str, Any]) -> None:
    """Synthetic, generated and curated inputs each carry their origin and licence.

    Parameters
    ----------
    built
        Fresh artefact.

    """
    templates_ = built["templates"]
    assert templates_["forecasting"]["dataset"]["licence"].startswith("AGPL-3.0-or-later")
    assert (
        templates_["forecasting"]["dataset"]["source"] == "scpn.forecasting.synthetic_kuramoto.v1"
    )
    assert templates_["qsnn"]["dataset"]["licence"].startswith("AGPL-3.0-or-later")
    assert templates_["free_energy"]["licence_note"].endswith("not human-subject data")
    assert templates_["free_energy"]["owner"] == "friston_fep application plugin"
    assert templates_["free_energy"]["backend"] == "numpy+scpn_quantum_control.fep"
    assert all(
        row["licence_note"] and row["public_reference"] for row in templates_["domain_proxies"]
    )
    assert (
        templates_["nqs"]["evidence_path"]
        == "data/jax_nqs_baseline/jax_nqs_baseline_evidence.json"
    )


def test_checkpoint_resume_keeps_seed_and_optimiser_identity(built: dict[str, Any]) -> None:
    """A resumed QSNN run matches the uninterrupted run under the owner's identity.

    Parameters
    ----------
    built
        Fresh artefact.

    """
    qsnn = built["templates"]["qsnn"]
    checkpoint = qsnn["checkpoint"]
    assert checkpoint["resumed_matches_uninterrupted"] is True
    assert checkpoint["schema"] == QSNN_CHECKPOINT_SCHEMA
    assert checkpoint["randomness"] == QSNN_RANDOMNESS
    assert checkpoint["learning_rate"] == qsnn["optimiser"]["learning_rate"] == 0.2
    assert qsnn["layer"]["seed"] == 11
    steps = qsnn["steps"]
    assert [step["index"] for step in steps] == [1, 2, 3, 4, 5]
    assert all(step["gradient_norm"] > 0.0 for step in steps)


def test_invalid_covariance_shows_the_owners_refusal(built: dict[str, Any]) -> None:
    """The free-energy refusals are the owner's own messages for the same inputs.

    Parameters
    ----------
    built
        Fresh artefact.

    """
    refusals = {
        row["case"]: row["message"] for row in built["templates"]["free_energy"]["refusals"]
    }
    for case, sigma in (
        ("negative covariance", np.array([[-1.0]])),
        ("covariance of the wrong shape", np.eye(2)),
    ):
        with pytest.raises(ValueError) as owner:
            variational_free_energy(np.zeros(1), sigma, np.zeros(1), np.eye(1))
        assert refusals[case] == str(owner.value)


def test_domain_proxies_stay_proxies(built: dict[str, Any]) -> None:
    """Each domain row carries its kit's forbidden claims and no physical outcome.

    Parameters
    ----------
    built
        Fresh artefact.

    """
    rows = {row["dataset_id"]: row for row in built["templates"]["domain_proxies"]}
    assert set(rows) == {"eeg_alpha_plv_8ch", "ieee5bus_power_grid", "iter_mhd_8mode"}
    for dataset_id, row in rows.items():
        kit = get_domain_application_honesty_kit_for_dataset(dataset_id)
        assert row["kit"]["claims_forbidden"] == list(kit.claims_forbidden)
        assert row["kit"]["publication_safe"] is False
        assert row["physical_outcome_in_custody"] is False
        assert "topology_similarity_proxy" in row["proxy_metrics"]
    assert "clinical validation" in rows["eeg_alpha_plv_8ch"]["kit"]["claims_forbidden"]
    assert "real plant control" in rows["iter_mhd_8mode"]["kit"]["claims_forbidden"]


def test_build_leaves_the_global_random_state_unchanged(built: dict[str, Any]) -> None:
    """A seeded caller keeps its exact global NumPy state across a build.

    Parameters
    ----------
    built
        Fresh artefact.

    """
    np.random.seed(1234)
    before = np.random.random()
    np.random.seed(1234)
    artifact.build_learning_instrument_artifact()
    assert np.random.random() == before
    assert built["global_random_state_unchanged"] is True


def test_a_template_that_draws_global_randomness_refuses_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when any template changes the caller's random state.

    Parameters
    ----------
    monkeypatch
        Replaces one template with one that draws from the global generator.

    """
    monkeypatch.setattr(artifact, "fep_template", lambda: {"draw": float(np.random.random())})
    with pytest.raises(RuntimeError, match="changed the caller's global random state"):
        artifact.build_learning_instrument_artifact()


def test_nqs_template_reads_committed_evidence_or_says_it_is_absent(tmp_path: Path) -> None:
    """Show the verified committed baseline, and an explicit absence without a file.

    Parameters
    ----------
    tmp_path
        Owned directory without evidence.

    """
    present = templates.nqs_template()
    assert present["status"] == "available"
    request, comparison = present["request"], present["comparison"]
    assert isinstance(request, dict) and isinstance(comparison, dict)
    assert request["seed"] == 7
    assert comparison["within_declared_tolerance"] is True
    assert comparison["energy_history"][-1] == comparison["variational_energy"]
    absent = templates.nqs_template(tmp_path / "missing.json")
    assert absent == {
        "owner": "scpn_quantum_control.jax_nqs_baseline_product",
        "status": "unavailable",
        "evidence_path": None,
    }


def _accepting_covariance_refusal(_plugin: FEPApplicationPlugin, _sigma: object) -> str:
    """Stand in for a free-energy owner that accepts every covariance.

    Parameters
    ----------
    _plugin
        Plugin whose method is replaced.
    _sigma
        Offered covariance, accepted.

    Raises
    ------
    ValueError
        Always, as the plugin reports an accepted covariance.

    """
    raise ValueError("the free-energy owner accepted the covariance")


def test_free_energy_template_requires_the_packaged_workflow_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a registry entry that is not the packaged free-energy workflow.

    Parameters
    ----------
    monkeypatch
        Makes the registry return another application's plugin under the FEP name.

    """
    eeg = get_application_plugin("eeg_alpha")
    monkeypatch.setattr(templates, "get_application_plugin", lambda name: eeg)
    with pytest.raises(TypeError, match="packaged free-energy workflow plugin"):
        templates.fep_template()


def _altered_evidence(tmp_path: Path, path: tuple[str, ...], value: object) -> Path:
    """Copy the committed forecasting evidence with one nested field replaced.

    Parameters
    ----------
    tmp_path
        Owned directory for the copy.
    path
        Field names from the top level down.
    value
        Replacement value.

    Returns
    -------
    Path
        The altered copy.

    """
    payload = json.loads(templates.FORECASTING_EVIDENCE.read_text(encoding="utf-8"))
    target = payload
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    altered = tmp_path / "evidence.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    return altered


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("dataset", "content_digest"), "0" * 64, "no longer matches its generator"),
        (("model", "training_batch_digest"), "0" * 64, "not fitted on the training split"),
        (("dataset",), [], "'dataset' must be a JSON object"),
        (("test_accuracy",), {"samples": 32}, "evidence lacks fields"),
    ],
)
def test_forecasting_template_refuses_inconsistent_evidence(
    tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    """Refuse evidence that no longer binds to its generator or its training split.

    Parameters
    ----------
    tmp_path
        Owned directory for the altered evidence.
    path
        Field to alter.
    value
        Altered value.
    message
        Expected refusal text.

    """
    with pytest.raises(ValueError, match=message):
        templates.forecasting_template(_altered_evidence(tmp_path, path, value))


def test_templates_reference_only_evidence_committed_in_the_repository(tmp_path: Path) -> None:
    """Refuse an unchanged evidence copy that lies outside the repository.

    Parameters
    ----------
    tmp_path
        Owned directory outside the repository.

    """
    forecasting_copy = tmp_path / "forecasting.json"
    forecasting_copy.write_bytes(templates.FORECASTING_EVIDENCE.read_bytes())
    nqs_copy = tmp_path / "nqs.json"
    nqs_copy.write_bytes(templates.NQS_EVIDENCE.read_bytes())
    with pytest.raises(ValueError, match="must be committed under the repository"):
        templates.forecasting_template(forecasting_copy)
    with pytest.raises(ValueError, match="must be committed under the repository"):
        templates.nqs_template(nqs_copy)


def test_owners_that_accept_invalid_requests_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed if an owner fits held-out data, scores overlap or accepts a bad covariance.

    Parameters
    ----------
    monkeypatch
        Replaces each owner entry point with one that accepts the invalid request.

    """
    monkeypatch.setattr(templates, "fit_multimodal_ridge_forecaster", lambda batch: batch)
    with pytest.raises(ValueError, match="accepted a held-out batch"):
        templates.forecasting_template()
    monkeypatch.setattr(FEPApplicationPlugin, "covariance_refusal", _accepting_covariance_refusal)
    with pytest.raises(ValueError, match="accepted the negative covariance"):
        templates.fep_template()

    class AcceptingTrainer(QSNNTrainer):
        accepted: QSNNEvaluation | None = None

        def evaluate(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> QSNNEvaluation:
            if self.accepted is None:
                self.accepted = super().evaluate(X, y)
            return self.accepted

    monkeypatch.setattr(
        templates,
        "_qsnn_trainer",
        lambda: AcceptingTrainer(QuantumDenseLayer(1, 2, seed=11), lr=0.2),
    )
    with pytest.raises(ValueError, match="scored samples that were used in training"):
        templates.qsnn_template()


def test_committed_artefact_and_cli_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write, check and refuse changed data through the public command line.

    Parameters
    ----------
    tmp_path
        Owned scratch output.
    monkeypatch
        Supplies the real command's argument vector.

    """
    assert artifact.main(["--check"]) == 0
    target = tmp_path / "learning.json"
    assert artifact.main(["--write", "--json-path", str(target)]) == 0
    assert artifact.main(["--check", "--json-path", str(target)]) == 0
    target.write_text("{}", encoding="utf-8")
    assert artifact.main(["--check", "--json-path", str(target)]) == 1
    monkeypatch.setattr(sys, "argv", ["learning_instrument_artifact", "--check"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module(artifact.__name__, run_name="__main__")
    assert stopped.value.code == 0
