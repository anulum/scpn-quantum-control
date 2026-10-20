# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — variational qualification owner tests
"""Exercise qualification admission, real observations and corrupted-run refusal."""

from __future__ import annotations

import hashlib
import json
import runpy
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.variational_baseline_qualification as qualification
from scpn_quantum_control.jax_nqs_baseline_product import JAXNQSBaselineSpec
from scpn_quantum_control.phase.jax_nqs import jax_rbm_energy, jax_vmc_ground_state
from scpn_quantum_control.phase.nqs_ansatz import RBMWavefunction
from scpn_quantum_control.variational_baseline_qualification import (
    VariationalBaselineQualification,
    qualify_jax_nqs_baseline,
)

pytestmark = pytest.mark.framework_imports


@pytest.fixture(scope="module")
def request_spec() -> JAXNQSBaselineSpec:
    """Keep the same small interacting, nonzero-field request for fault controls."""
    return JAXNQSBaselineSpec.from_arrays(
        np.array(((0.0, 0.4), (0.4, 0.0))), np.array((0.1, -0.2)), n_iterations=2, seed=7
    )


@pytest.fixture(scope="module")
def replay_payload(request_spec: JAXNQSBaselineSpec) -> dict[str, Any]:
    """Obtain actual native fitted parameters before injecting individual faults."""
    return jax_vmc_ground_state(
        np.asarray(request_spec.coupling),
        np.asarray(request_spec.omega),
        n_hidden=request_spec.n_hidden,
        learning_rate=request_spec.learning_rate,
        n_iterations=request_spec.n_iterations,
        seed=request_spec.seed,
    )


@pytest.fixture(scope="module")
def qualified(request_spec: JAXNQSBaselineSpec) -> VariationalBaselineQualification:
    """Run a real unmodified qualification used by immutable-record checks."""
    return qualify_jax_nqs_baseline(request_spec)


def test_qualification_serialisation_binds_original_record_and_every_observation(
    qualified: VariationalBaselineQualification,
) -> None:
    """Bind the complete companion while leaving the original raw v2 untouched."""
    record = qualified.to_dict()
    digest = record.pop("evidence_sha256")
    expected = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert digest == expected
    assert json.loads(json.dumps(record)) == record
    assert record["baseline"] == qualified.baseline.to_dict()
    assert record["schema"] == "variational_baseline_qualification.v1"
    assert record["sampling_mode"] == "exact_enumeration"
    assert record["cost_classification"] == "functional_non_isolated"
    assert len(qualified.parameters_sha256) == 64
    assert all(dtype.startswith("float") for _, dtype in qualified.parameter_dtypes)
    assert qualified.costs.training_seconds >= 0.0
    assert qualified.costs.replay_seconds >= 0.0
    assert qualified.costs.evaluation_seconds >= 0.0
    assert qualified.costs.training_energy_evaluations == 6
    assert qualified.costs.training_gradient_evaluations == 4
    assert qualified.costs.evaluation_observable_calls == 2
    assert qualified.costs.exact_configurations_per_observable == 4

    def assign(record: object, name: str, value: float) -> None:
        """Attempt an ordinary caller write through a named attribute."""
        setattr(record, name, value)

    with pytest.raises(FrozenInstanceError):
        assign(qualified.costs, "training_seconds", -1.0)
    with pytest.raises(ValueError, match="Out of range"):
        replace(qualified, costs=replace(qualified.costs, training_seconds=float("nan"))).to_dict()


def test_unidentified_components_remain_unknown(
    qualified: VariationalBaselineQualification,
) -> None:
    """Do not attribute a generic variational gap to either unmeasured component."""
    errors = qualified.errors
    assert errors.ansatz_energy_infimum is None
    assert errors.representation_bias is None
    assert errors.optimisation_excess is None
    assert "not_identifiable" in errors.certificate
    assert errors.monte_carlo_standard_error == 0.0
    assert errors.combined_energy_excess == pytest.approx(
        qualified.training.variational_energy - qualified.training.exact_ground_energy
    )


@pytest.mark.parametrize(("visible", "hidden"), ((2, 1), (3, 2), (4, 1), (6, 2)))
def test_real_parameter_and_configuration_scaling(visible: int, hidden: int) -> None:
    """Exercise distinct native shapes against an analytic diagonal ground energy."""
    spec = JAXNQSBaselineSpec.from_arrays(
        np.zeros((visible, visible)), np.ones(visible), n_hidden=hidden, n_iterations=1
    )
    result = qualify_jax_nqs_baseline(spec)
    assert result.training.exact_ground_energy == pytest.approx(-float(visible))
    assert result.training.variational_energy >= -visible - 1e-5
    assert result.baseline.comparison.n_parameters == visible + hidden + visible * hidden
    assert result.costs.exact_configurations_per_observable == 2**visible
    assert dict(result.parameter_shapes)["W"] == (hidden, visible)


@pytest.mark.parametrize("coupling", (0.0, 0.4, -0.4, 1e-18))
def test_analytic_certificate_uses_the_effective_hamiltonian(coupling: float) -> None:
    """Keep ansatz limits valid for zero, either sign and sub-threshold pair terms."""
    spec = JAXNQSBaselineSpec.from_arrays(
        np.array(((9.0, coupling), (coupling, 9.0))), np.zeros(2), n_iterations=1
    )
    result = qualify_jax_nqs_baseline(spec)
    assert result.errors.representation_bias is not None
    assert result.errors.representation_bias >= -1e-12
    assert result.errors.optimisation_excess is not None
    assert result.errors.representation_bias + result.errors.optimisation_excess == pytest.approx(
        result.errors.combined_energy_excess, abs=1e-12
    )
    if coupling == 0.0:
        assert result.training.reference_subspace_dimension == 4
        assert result.training.reference_subspace_overlap == pytest.approx(1.0, abs=2e-6)
        assert result.training.reference_subspace_energy_tolerance == 1e-10
    elif coupling == -0.4:
        assert result.errors.representation_bias == pytest.approx(0.8)
    else:
        assert result.errors.representation_bias == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("case", ("label", "reserved", "shape", "same", "duplicate", "too-many"))
def test_invalid_evaluation_is_refused_before_training(
    request_spec: JAXNQSBaselineSpec,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Reject malformed or misleading held-out data before invoking the optimiser."""
    k = np.zeros((2, 2))
    omega = np.ones(2)
    held_out: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {
        "evaluation": (k, omega)
    }
    if case == "label":
        held_out = {" ": (k, omega)}
    elif case == "reserved":
        held_out = {"training": (k, omega)}
    elif case == "shape":
        held_out = {"evaluation": (np.zeros((3, 3)), np.ones(3))}
    elif case == "same":
        held_out = {
            "copy": (np.asarray(request_spec.coupling) + np.eye(2), np.asarray(request_spec.omega))
        }
    elif case == "duplicate":
        held_out["other-label"] = (k, omega)
    else:
        held_out = {f"case-{i}": (k, omega * (i + 1)) for i in range(9)}

    def forbidden_training(spec: JAXNQSBaselineSpec) -> Any:
        """Detect a training side effect on a request that must be refused first."""
        raise AssertionError("invalid evaluation reached training")

    monkeypatch.setattr(qualification, "run_jax_nqs_baseline", forbidden_training)
    with pytest.raises(ValueError):
        qualify_jax_nqs_baseline(request_spec, held_out=held_out)


def test_parameter_budget_refuses_without_dense_allocation(
    request_spec: JAXNQSBaselineSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse excessive hidden width before requesting a Hamiltonian or native run."""

    def forbidden_matrix(*args: object, **kwargs: object) -> Any:
        """Detect allocation before the qualification parameter budget is admitted."""
        raise AssertionError("excessive width reached matrix construction")

    monkeypatch.setattr(qualification, "knm_to_dense_matrix", forbidden_matrix)
    with pytest.raises(ValueError, match="4096"):
        qualify_jax_nqs_baseline(replace(request_spec, n_hidden=4096))


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        ("history-shape", "energy history"),
        ("history-values", "energy history"),
        ("count", "parameter count"),
        ("energy", "parameter count"),
        ("parameter-keys", "mapping"),
        ("parameter-shape", "parameters"),
        ("parameter-dtype", "parameters"),
        ("parameter-finite", "parameters"),
    ),
)
def test_corrupt_replay_is_rejected(
    request_spec: JAXNQSBaselineSpec,
    replay_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    message: str,
) -> None:
    """Reject each independent corruption of an otherwise actual native result."""
    replay = deepcopy(replay_payload)
    if fault == "history-shape":
        replay["energy_history"] = replay["energy_history"][:-1]
    elif fault == "history-values":
        replay["energy_history"][0] += 1.0
    elif fault == "count":
        replay["n_params"] += 0.5
    elif fault == "energy":
        replay["energy"] += 1.0
    elif fault == "parameter-keys":
        del replay["params"]["a"]
    elif fault == "parameter-shape":
        replay["params"]["a"] = np.zeros(3)
    elif fault == "parameter-dtype":
        replay["params"]["a"] = np.zeros(2, dtype=np.int64)
    else:
        replay["params"]["a"][0] = np.nan
    monkeypatch.setattr(qualification, "jax_vmc_ground_state", lambda *args, **kwargs: replay)
    with pytest.raises(RuntimeError, match=message):
        qualify_jax_nqs_baseline(request_spec)


@pytest.mark.parametrize("observable", (0, 1))
def test_wrong_jax_observable_is_detected_by_independent_state(
    request_spec: JAXNQSBaselineSpec,
    monkeypatch: pytest.MonkeyPatch,
    observable: int,
) -> None:
    """Reject either a wrong energy or a wrong projector expectation after real work."""
    real_energy = jax_rbm_energy
    calls = 0

    def corrupted(params: dict[str, Any], hamiltonian: Any, n: int) -> Any:
        """Perturb exactly one actual JAX result while preserving its native call."""
        nonlocal calls
        value = real_energy(params, hamiltonian, n)
        result = value + 1.0 if calls == observable else value
        calls += 1
        return result

    monkeypatch.setattr(qualification, "jax_rbm_energy", corrupted)
    with pytest.raises(RuntimeError, match="independent NumPy"):
        qualify_jax_nqs_baseline(request_spec)


@pytest.mark.parametrize("finite", (False, True))
def test_invalid_independent_state_is_refused(
    request_spec: JAXNQSBaselineSpec,
    monkeypatch: pytest.MonkeyPatch,
    finite: bool,
) -> None:
    """Reject non-finite and finite unnormalised independent state evaluations."""

    def invalid_state(self: object) -> NDArray[np.complex128]:
        """Inject one invalid NumPy state after the actual JAX optimiser completed."""
        return np.full(4, 0.0 if finite else np.nan, dtype=np.complex128)

    monkeypatch.setattr(RBMWavefunction, "all_amplitudes", invalid_state)
    with pytest.raises(RuntimeError, match="normalised"):
        qualify_jax_nqs_baseline(request_spec)


def test_state_is_bound_to_the_original_energy_history(
    request_spec: JAXNQSBaselineSpec,
    replay_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject finite replay parameters whose energy disagrees with the raw result."""
    replay = deepcopy(replay_payload)
    replay["params"]["a"][:] = (1.0, -1.0)
    monkeypatch.setattr(qualification, "jax_vmc_ground_state", lambda *args, **kwargs: replay)
    with pytest.raises(RuntimeError, match="original final energy"):
        qualify_jax_nqs_baseline(request_spec)


def test_complete_file_example_emits_real_retained_evaluation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the shipped example consumer and inspect its actual JSON output."""
    namespace = runpy.run_path("examples/variational_baseline_qualification.py")
    namespace["main"]()
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "variational_baseline_qualification.v1"
    assert record["held_out"][0]["label"] == "opposite_field"
    assert record["costs"]["training_gradient_evaluations"] == 60
    assert record["observable_absolute_tolerance"] == 1e-5
    assert record["observable_relative_tolerance"] == 1e-4
    assert not record["global_optimum_claimed"]
