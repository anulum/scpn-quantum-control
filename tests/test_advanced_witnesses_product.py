# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for advanced witnesses product
"""Behaviour tests for :mod:`scpn_quantum_control.advanced_witnesses_product`.

One concern per test. No coverage buckets, no unused-assert padding.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping

import numpy as np
import pytest

from scpn_quantum_control.advanced_witnesses_product import (
    ADVANCED_WITNESSES_CLAIM_BOUNDARY,
    ADVANCED_WITNESSES_PRODUCT_SCHEMA,
    MAX_DEMO_SHADOW_SHOTS,
    MAX_WITNESS_QUBITS,
    MIN_SHADOW_SHOTS,
    MaterialisedKrylovProbe,
    MaterialisedOtocProbe,
    MaterialisedShadowProbe,
    PathEligibilityDecision,
    WitnessBoundaryRow,
    WitnessCapabilityRow,
    WitnessEstimate,
    assert_advanced_witnesses_product_integrity,
    build_advanced_witnesses_product_registry,
    decide_witness_path,
    get_witness_boundary,
    get_witness_capability,
    get_witness_glossary_entry,
    iter_witness_boundaries,
    iter_witness_capabilities,
    list_witness_ambient_inventory,
    list_witness_boundary_ids,
    list_witness_capability_ids,
    list_witness_glossary_keys,
    map_advanced_witnesses_public_surfaces,
    materialise_demo_krylov_probe,
    materialise_demo_otoc_probe,
    materialise_demo_shadow_probe,
    materialise_harmonic_order_parameter_compose,
    materialise_krylov_probe,
    materialise_otoc_probe,
    materialise_shadow_probe,
)


def _registry_rows(registry: Mapping[str, object], key: str) -> list[dict[str, object]]:
    """Return mutable copies of registry rows after structural narrowing."""
    raw_rows = registry[key]
    if not isinstance(raw_rows, list):
        raise AssertionError(f"{key} must be a list in the valid registry fixture")
    rows: list[dict[str, object]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise AssertionError(f"{key} rows must be mappings in the valid fixture")
        rows.append(dict(raw_row))
    return rows


# ---------------------------------------------------------------------------
# Registry / catalogue
# ---------------------------------------------------------------------------


def test_registry_integrity_pass() -> None:
    """Built registry validates and freezes invent-green policies to False."""
    reg = assert_advanced_witnesses_product_integrity(build_advanced_witnesses_product_registry())
    assert reg["schema"] == ADVANCED_WITNESSES_PRODUCT_SCHEMA
    assert reg["hardware_submit_allowed_policy"] is False
    assert reg["otoc_advantage_claim_policy"] is False
    assert reg["topology_certification_policy"] is False
    assert reg["live_qpu_witness_policy"] is False
    assert reg["blank_entry_count"] == 0
    assert reg["max_witness_qubits"] == MAX_WITNESS_QUBITS


def test_assert_integrity_none_builds_fresh() -> None:
    """Integrity helper builds a registry when called with None."""
    reg = assert_advanced_witnesses_product_integrity(None)
    assert reg["schema"] == ADVANCED_WITNESSES_PRODUCT_SCHEMA


def test_capability_catalogue_lists_core_kinds() -> None:
    """Catalogue includes Krylov, OTOC, shadows, and harmonic compose rows."""
    ids = list_witness_capability_ids()
    assert "krylov_complexity" in ids
    assert "otoc_probe" in ids
    assert "classical_shadows" in ids
    assert "synchronisation_witness_compose" in ids
    for row in iter_witness_capabilities():
        assert row.hardware_submit_allowed is False
        assert row.to_dict()["capability_id"] == row.capability_id


def test_get_capability_known_and_unknown() -> None:
    """Known capability resolves; unknown raises KeyError."""
    row = get_witness_capability("otoc_probe")
    assert row.kind == "otoc_probe"
    assert "compute_otoc" in row.ambient_symbol
    with pytest.raises(KeyError, match="unknown witness capability"):
        get_witness_capability("not_a_real_capability")


def test_boundary_catalogue_all_fail_closed() -> None:
    """Every hard-gap boundary is fail-closed."""
    assert "otoc_advantage_claim" in list_witness_boundary_ids()
    assert "live_qpu_witness" in list_witness_boundary_ids()
    for row in iter_witness_boundaries():
        assert row.fail_closed is True
    assert get_witness_boundary("live_qpu_witness").kind == "live_qpu_witness"
    with pytest.raises(KeyError, match="unknown witness boundary"):
        get_witness_boundary("nope")


def test_glossary_and_ambient_inventory() -> None:
    """Glossary and ambient inventory expose Krylov/OTOC/shadow entry points."""
    assert "Krylov" in list_witness_glossary_keys()
    assert "OTOC" in list_witness_glossary_keys()
    entry = get_witness_glossary_entry("OTOC")
    assert "correlator" in entry.lower() or "order" in entry.lower()
    with pytest.raises(KeyError):
        get_witness_glossary_entry("not_a_glossary_key")
    modules = {row["module"] for row in list_witness_ambient_inventory()}
    assert any("krylov" in m for m in modules)
    assert any("otoc" in m for m in modules)
    assert any("shadow" in m for m in modules)
    surfaces = map_advanced_witnesses_public_surfaces()
    assert "krylov_probe" in surfaces
    assert "product_registry" in surfaces


# ---------------------------------------------------------------------------
# Path decisions
# ---------------------------------------------------------------------------


def test_decide_path_allows_local_research() -> None:
    """Local research paths are allowed under product caps."""
    decision = decide_witness_path("krylov")
    assert decision.allowed is True
    assert decision.to_dict()["allowed"] is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"invent_green_otoc_advantage": True}, "advantage"),
        ({"invent_green_topology_cert": True}, "topology"),
        ({"invent_green_live_qpu": True}, "live QPU"),
        ({"unrestricted_shadow": True}, "unrestricted"),
    ],
)
def test_decide_path_refuses_invent_green(kwargs: dict[str, bool], match: str) -> None:
    """Invent-green and unrestricted-shadow flags refuse with invent_green_refused."""
    decision = decide_witness_path("otoc", **kwargs)
    assert decision.allowed is False
    assert decision.invent_green_refused is True
    assert match.lower() in decision.reason.lower() or match in decision.reason


def test_decide_path_refuses_over_qubit_cap() -> None:
    """n_qubits above product cap is refused without invent-green flag."""
    decision = decide_witness_path("otoc", n_qubits=MAX_WITNESS_QUBITS + 1)
    assert decision.allowed is False
    assert decision.invent_green_refused is False


def test_decide_path_refuses_unknown_and_empty() -> None:
    """Unknown path_id is refused; empty path_id raises."""
    assert decide_witness_path("not_a_path").allowed is False
    with pytest.raises(ValueError, match="path_id"):
        decide_witness_path("")


# ---------------------------------------------------------------------------
# Real ambient probes
# ---------------------------------------------------------------------------


def test_krylov_demo_probe_real_ambient() -> None:
    """Demo Krylov probe runs ambient krylov_complexity and is digest-stable."""
    p1 = materialise_demo_krylov_probe()
    p2 = materialise_demo_krylov_probe()
    assert p1.digest == p2.digest
    assert p1.n_lanczos >= 0
    assert p1.n_times >= 2
    assert math.isfinite(p1.peak_complexity)
    assert p1.estimate.support_status == "supported"
    assert p1.invent_green_live_qpu is False
    assert len(p1.digest) == 64
    payload = p1.to_dict()
    assert payload["peak_complexity"] == p1.peak_complexity
    assert payload["digest"] == p1.digest


def test_krylov_single_qubit_trivial_lanczos_is_supported() -> None:
    """n_qubits=1 may return n_lanczos=0; product still packages a supported probe."""
    probe = materialise_krylov_probe(n_qubits=1, coupling=0.0, t_max=1.0, n_times=6, max_lanczos=8)
    assert probe.estimate.n_qubits == 1
    assert probe.n_lanczos >= 0
    assert math.isfinite(probe.peak_complexity)
    assert probe.estimate.support_status == "supported"


def test_krylov_refuses_over_cap_and_bad_params() -> None:
    """Krylov probe fails closed on qubit cap and invalid timing params."""
    with pytest.raises(ValueError, match="exceeds product cap"):
        materialise_krylov_probe(n_qubits=MAX_WITNESS_QUBITS + 1)
    with pytest.raises(ValueError, match="n_times"):
        materialise_krylov_probe(n_times=1)
    with pytest.raises(ValueError, match="t_max"):
        materialise_krylov_probe(t_max=0.0)
    with pytest.raises(ValueError, match="max_lanczos"):
        materialise_krylov_probe(max_lanczos=1)


def test_otoc_demo_probe_real_ambient() -> None:
    """Demo OTOC probe runs ambient compute_otoc without invent-green advantage."""
    p1 = materialise_demo_otoc_probe()
    p2 = materialise_demo_otoc_probe()
    assert p1.digest == p2.digest
    assert math.isfinite(p1.final_otoc)
    assert p1.n_times >= 2
    assert p1.invent_green_otoc_advantage is False
    assert p1.invent_green_live_qpu is False
    assert abs(p1.final_otoc) <= 1.5
    payload = p1.to_dict()
    assert payload["final_otoc"] == p1.final_otoc
    assert "lyapunov_estimate" in payload


def test_otoc_refuses_over_cap_and_bad_params() -> None:
    """OTOC probe fails closed on qubit cap and invalid n_times."""
    with pytest.raises(ValueError, match="exceeds product cap"):
        materialise_otoc_probe(n_qubits=MAX_WITNESS_QUBITS + 2)
    with pytest.raises(ValueError, match="n_times"):
        materialise_otoc_probe(n_times=1)
    with pytest.raises(ValueError, match="t_max"):
        materialise_otoc_probe(t_max=-1.0)


def test_shadow_demo_probe_real_ambient() -> None:
    """Demo shadow probe runs ambient classical_shadow_estimation (subprocess)."""
    p1 = materialise_demo_shadow_probe()
    p2 = materialise_demo_shadow_probe()
    assert p1.digest == p2.digest
    assert p1.n_shots == 80
    assert p1.shadow_norm_bound >= 0.0
    assert p1.estimate.support_status == "supported"
    assert p1.invent_green_live_qpu is False
    assert p1.observables
    payload = p1.to_dict()
    assert payload["n_shots"] == 80
    assert payload["observables"] == dict(p1.observables)


def test_shadow_under_sampled_is_honest() -> None:
    """Low-shot shadows report under_sampled rather than silent green."""
    probe = materialise_shadow_probe(n_qubits=2, n_shots=max(1, MIN_SHADOW_SHOTS - 1), seed=3)
    assert probe.estimate.support_status == "under_sampled"
    assert probe.estimate.uncertainty >= 0.0


def test_shadow_refuses_caps_and_bad_labels() -> None:
    """Shadow probe enforces shot/qubit caps and Pauli label length."""
    with pytest.raises(ValueError, match="exceeds product cap"):
        materialise_shadow_probe(n_qubits=MAX_WITNESS_QUBITS + 1)
    with pytest.raises(ValueError, match="MAX_DEMO_SHADOW_SHOTS"):
        materialise_shadow_probe(n_shots=MAX_DEMO_SHADOW_SHOTS + 1)
    with pytest.raises(ValueError, match="n_shots"):
        materialise_shadow_probe(n_shots=0)
    with pytest.raises(ValueError, match="label length"):
        materialise_shadow_probe(n_qubits=2, observables={"bad": "Z"})
    with pytest.raises(ValueError, match="non-empty"):
        materialise_shadow_probe(n_qubits=2, observables={})


def test_shadow_one_qubit_default_observable() -> None:
    """One-qubit default path estimates Z via ambient shadows."""
    probe = materialise_shadow_probe(n_qubits=1, n_shots=40, seed=11)
    assert probe.estimate.n_qubits == 1
    assert probe.observables


def test_shadow_multi_observable_real_ambient() -> None:
    """Custom multi-Pauli labels exercise real ambient shadow estimation."""
    probe = materialise_shadow_probe(
        n_qubits=2,
        n_shots=40,
        seed=5,
        observables={"zi": "ZI", "iz": "IZ"},
    )
    assert set(probe.observables) == {"zi", "iz"}
    assert probe.estimate.support_status == "supported"
    assert probe.shadow_norm_bound >= 0.0


def test_harmonic_order_parameter_compose_sync_cloud() -> None:
    """Harmonic compose returns high R for a tightly synchronised phase cloud."""
    est = materialise_harmonic_order_parameter_compose()
    assert est.mean > 0.9
    assert est.support_status == "supported"
    assert est.invent_green_live_qpu is False
    assert "harmonic_order_parameter" in est.estimator_id


def test_synchronisation_witness_compose_custom_phases_and_validation() -> None:
    """Custom phases yield R≈1 when aligned; empty phases and harmonic=0 refuse."""
    est = materialise_harmonic_order_parameter_compose([0.0, 0.0, 0.0], harmonic=1)
    assert est.mean == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError, match="harmonic"):
        materialise_harmonic_order_parameter_compose(harmonic=0)
    with pytest.raises(ValueError, match="phases"):
        materialise_harmonic_order_parameter_compose([])


# ---------------------------------------------------------------------------
# Invent-green refuse on materialise (public kwargs → real decide_*)
# ---------------------------------------------------------------------------


def test_krylov_refuses_invent_green_live_qpu() -> None:
    """Krylov materialise refuses invent-green live QPU before ambient work."""
    with pytest.raises(ValueError, match="live QPU"):
        materialise_krylov_probe(n_qubits=2, invent_green_live_qpu=True)


def test_krylov_refuses_invent_green_topology_cert() -> None:
    """Krylov materialise refuses invent-green topology certification."""
    with pytest.raises(ValueError, match="topology"):
        materialise_krylov_probe(n_qubits=2, invent_green_topology_cert=True)


def test_otoc_refuses_invent_green_advantage() -> None:
    """OTOC materialise refuses invent-green advantage claims."""
    with pytest.raises(ValueError, match="advantage|OTOC"):
        materialise_otoc_probe(n_qubits=2, invent_green_otoc_advantage=True)


def test_otoc_refuses_invent_green_live_qpu() -> None:
    """OTOC materialise refuses invent-green live QPU."""
    with pytest.raises(ValueError, match="live QPU"):
        materialise_otoc_probe(n_qubits=2, invent_green_live_qpu=True)


def test_shadow_refuses_invent_green_live_qpu() -> None:
    """Shadow materialise refuses invent-green live QPU."""
    with pytest.raises(ValueError, match="live QPU"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, invent_green_live_qpu=True)


def test_shadow_refuses_unrestricted_campaign() -> None:
    """Shadow materialise refuses unrestricted campaigns without support profile."""
    with pytest.raises(ValueError, match="unrestricted"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, unrestricted_shadow=True)


def test_harmonic_compose_refuses_invent_green_topology_cert() -> None:
    """Harmonic compose refuses invent-green topology certification."""
    with pytest.raises(ValueError, match="topology"):
        materialise_harmonic_order_parameter_compose(invent_green_topology_cert=True)


def test_harmonic_compose_refuses_invent_green_live_qpu() -> None:
    """Harmonic compose refuses invent-green live QPU."""
    with pytest.raises(ValueError, match="live QPU"):
        materialise_harmonic_order_parameter_compose(invent_green_live_qpu=True)


# ---------------------------------------------------------------------------
# Ambient / subprocess corruption contracts (injected boundary only)
# ---------------------------------------------------------------------------


def test_krylov_rejects_non_finite_ambient_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrupt ambient peak must not be packaged as a supported probe."""
    import scpn_quantum_control.advanced_witnesses_product as mod

    class _Bad:
        peak_complexity = float("nan")
        n_lanczos = 2
        times = np.linspace(0.0, 1.0, 4)

    monkeypatch.setattr(mod, "krylov_complexity", lambda *_a, **_k: _Bad())
    with pytest.raises(ValueError, match="non-finite peak"):
        materialise_krylov_probe(n_qubits=2)


def test_krylov_rejects_negative_ambient_n_lanczos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative ambient n_lanczos is invalid and fail-closed."""
    import scpn_quantum_control.advanced_witnesses_product as mod

    class _Bad:
        peak_complexity = 0.0
        n_lanczos = -3
        times = np.linspace(0.0, 1.0, 4)

    monkeypatch.setattr(mod, "krylov_complexity", lambda *_a, **_k: _Bad())
    with pytest.raises(ValueError, match="invalid n_lanczos"):
        materialise_krylov_probe(n_qubits=2)


def test_shadow_subprocess_called_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CalledProcessError from ambient shadow subprocess is fail-closed."""

    def boom(*_a: object, **_k: object) -> object:
        raise subprocess.CalledProcessError(1, "x", stderr="shadow boom")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(ValueError, match="shadow subprocess failed"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)


def test_shadow_subprocess_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """TimeoutExpired from ambient shadow subprocess is fail-closed."""

    def timeout(*_a: object, **_k: object) -> object:
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ValueError, match="timed out"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)


def test_shadow_subprocess_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-JSON stdout from ambient shadow subprocess is fail-closed."""

    class _Out:
        stdout = "not-json\n"

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Out())
    with pytest.raises(ValueError, match="non-JSON"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)


def test_shadow_subprocess_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON array (not object) payload from ambient shadow is fail-closed."""

    class _Out:
        stdout = "[1, 2, 3]\n"

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Out())
    with pytest.raises(ValueError, match="must be an object"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)


def test_shadow_rejects_missing_observables_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty estimated_observables map is fail-closed."""
    import scpn_quantum_control.advanced_witnesses_product as mod

    monkeypatch.setattr(
        mod,
        "_run_ambient_shadow_json",
        lambda **_k: {"estimated_observables": {}},
    )
    with pytest.raises(ValueError, match="estimated_observables"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)


def test_shadow_rejects_negative_norm_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative shadow_norm_bound is fail-closed."""
    import scpn_quantum_control.advanced_witnesses_product as mod

    monkeypatch.setattr(
        mod,
        "_run_ambient_shadow_json",
        lambda **_k: {
            "estimated_observables": {"zi": 0.0},
            "shadow_norm_bound": -0.5,
            "n_qubits": 2,
            "n_shots": 20,
        },
    )
    with pytest.raises(ValueError, match="shadow_norm_bound"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)


def test_shadow_rejects_non_numeric_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-numeric shadow_norm_bound is fail-closed."""
    import scpn_quantum_control.advanced_witnesses_product as mod

    monkeypatch.setattr(
        mod,
        "_run_ambient_shadow_json",
        lambda **_k: {
            "estimated_observables": {"z": 0.0},
            "shadow_norm_bound": "nope",
            "n_qubits": 2,
            "n_shots": 20,
        },
    )
    with pytest.raises(ValueError, match="shadow_norm_bound"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)


def test_shadow_rejects_non_numeric_n_qubits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-numeric n_qubits in ambient payload is fail-closed."""
    import scpn_quantum_control.advanced_witnesses_product as mod

    monkeypatch.setattr(
        mod,
        "_run_ambient_shadow_json",
        lambda **_k: {
            "estimated_observables": {"z": 0.0},
            "shadow_norm_bound": 0.1,
            "n_qubits": "two",
            "n_shots": 20,
        },
    )
    with pytest.raises(ValueError, match="n_qubits"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)


def test_shadow_rejects_non_numeric_n_shots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-numeric n_shots in ambient payload is fail-closed."""
    import scpn_quantum_control.advanced_witnesses_product as mod

    monkeypatch.setattr(
        mod,
        "_run_ambient_shadow_json",
        lambda **_k: {
            "estimated_observables": {"z": 0.0},
            "shadow_norm_bound": 0.1,
            "n_qubits": 2,
            "n_shots": "twenty",
        },
    )
    with pytest.raises(ValueError, match="n_shots"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)


# ---------------------------------------------------------------------------
# Dataclass contracts (one concern each)
# ---------------------------------------------------------------------------


def test_witness_estimate_rejects_invent_green_flag() -> None:
    """WitnessEstimate construction rejects invent_green_live_qpu=True."""
    with pytest.raises(ValueError, match="invent_green"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=0.0,
            support_status="supported",
            backend="local",
            n_qubits=1,
            n_shots_or_times=1,
            invent_green_live_qpu=True,
        )


def test_witness_estimate_rejects_non_finite_mean() -> None:
    """WitnessEstimate rejects non-finite mean."""
    with pytest.raises(ValueError, match="mean"):
        WitnessEstimate(
            estimator_id="x",
            mean=float("nan"),
            uncertainty=0.0,
            support_status="supported",
            backend="local",
            n_qubits=1,
            n_shots_or_times=1,
        )


def test_witness_estimate_rejects_negative_uncertainty() -> None:
    """WitnessEstimate rejects negative uncertainty."""
    with pytest.raises(ValueError, match="uncertainty"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=-1.0,
            support_status="supported",
            backend="local",
            n_qubits=1,
            n_shots_or_times=1,
        )


def test_witness_estimate_rejects_blank_id() -> None:
    """WitnessEstimate rejects blank estimator_id."""
    with pytest.raises(ValueError, match="estimator_id"):
        WitnessEstimate(
            estimator_id="",
            mean=0.0,
            uncertainty=0.0,
            support_status="supported",
            backend="b",
            n_qubits=1,
            n_shots_or_times=1,
        )


def test_witness_estimate_rejects_bad_support_status() -> None:
    """WitnessEstimate rejects unknown support_status."""
    with pytest.raises(ValueError, match="support_status"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=0.0,
            support_status="weird",  # type: ignore[arg-type]
            backend="b",
            n_qubits=1,
            n_shots_or_times=1,
        )


def test_witness_estimate_rejects_zero_qubits() -> None:
    """WitnessEstimate rejects n_qubits < 1."""
    with pytest.raises(ValueError, match="n_qubits"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=0.0,
            support_status="supported",
            backend="b",
            n_qubits=0,
            n_shots_or_times=1,
        )


def test_witness_estimate_rejects_negative_shots() -> None:
    """WitnessEstimate rejects negative n_shots_or_times."""
    with pytest.raises(ValueError, match="n_shots_or_times"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=0.0,
            support_status="supported",
            backend="b",
            n_qubits=1,
            n_shots_or_times=-1,
        )


def test_witness_estimate_rejects_blank_backend() -> None:
    """WitnessEstimate rejects blank backend."""
    with pytest.raises(ValueError, match="backend"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=0.0,
            support_status="supported",
            backend="",
            n_qubits=1,
            n_shots_or_times=1,
        )


def test_witness_estimate_to_dict_round_trip_fields() -> None:
    """Valid WitnessEstimate serialises stable fields."""
    est = WitnessEstimate(
        estimator_id="x",
        mean=0.5,
        uncertainty=0.1,
        support_status="supported",
        backend="local",
        n_qubits=2,
        n_shots_or_times=10,
    )
    assert est.to_dict()["estimator_id"] == "x"
    assert est.to_dict()["mean"] == 0.5


def test_capability_row_rejects_hardware_submit() -> None:
    """Capability rows hard-fail if hardware_submit_allowed is True."""
    with pytest.raises(ValueError, match="hardware_submit"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
            hardware_submit_allowed=True,
        )


def test_capability_row_rejects_blank_id() -> None:
    """Capability rows reject blank capability_id."""
    with pytest.raises(ValueError, match="capability_id"):
        WitnessCapabilityRow(
            capability_id="  ",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
        )


def test_capability_row_rejects_unknown_kind() -> None:
    """Capability rows reject unknown kind."""
    with pytest.raises(ValueError, match="unknown capability kind"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="not_a_kind",  # type: ignore[arg-type]
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
        )


def test_capability_row_rejects_blank_title() -> None:
    """Capability rows reject blank title."""
    with pytest.raises(ValueError, match="title"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title=" ",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
        )


def test_capability_row_rejects_blank_summary() -> None:
    """Capability rows reject blank summary."""
    with pytest.raises(ValueError, match="summary"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="",
            ambient_module="m",
            ambient_symbol="s",
        )


def test_capability_row_rejects_blank_ambient_module() -> None:
    """Capability rows reject blank ambient_module."""
    with pytest.raises(ValueError, match="ambient_module"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="",
            ambient_symbol="s",
        )


def test_capability_row_rejects_blank_ambient_symbol() -> None:
    """Capability rows reject blank ambient_symbol."""
    with pytest.raises(ValueError, match="ambient_symbol"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol=" ",
        )


def test_capability_row_rejects_unknown_support_posture() -> None:
    """Capability rows reject unknown support_posture."""
    with pytest.raises(ValueError, match="support_posture"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
            support_posture="nope",  # type: ignore[arg-type]
        )


def test_capability_row_rejects_blank_as_of() -> None:
    """Capability rows reject blank as_of."""
    with pytest.raises(ValueError, match="as_of"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
            as_of="",
        )


def test_boundary_row_rejects_fail_closed_false() -> None:
    """Boundary rows require fail_closed True."""
    with pytest.raises(ValueError, match="fail_closed"):
        WitnessBoundaryRow(
            boundary_id="x",
            kind="otoc_advantage_claim",
            title="t",
            summary="s",
            fail_closed=False,
        )


def test_boundary_row_rejects_blank_id() -> None:
    """Boundary rows reject blank boundary_id."""
    with pytest.raises(ValueError, match="boundary_id"):
        WitnessBoundaryRow(
            boundary_id="",
            kind="otoc_advantage_claim",
            title="t",
            summary="s",
        )


def test_boundary_row_rejects_unknown_kind() -> None:
    """Boundary rows reject unknown kind."""
    with pytest.raises(ValueError, match="unknown boundary kind"):
        WitnessBoundaryRow(
            boundary_id="x",
            kind="nope",  # type: ignore[arg-type]
            title="t",
            summary="s",
        )


def test_boundary_row_rejects_blank_title() -> None:
    """Boundary rows reject blank title."""
    with pytest.raises(ValueError, match="title"):
        WitnessBoundaryRow(
            boundary_id="x",
            kind="otoc_advantage_claim",
            title=" ",
            summary="s",
        )


def test_boundary_row_rejects_blank_summary() -> None:
    """Boundary rows reject blank summary."""
    with pytest.raises(ValueError, match="summary"):
        WitnessBoundaryRow(
            boundary_id="x",
            kind="otoc_advantage_claim",
            title="t",
            summary="",
        )


def test_path_decision_rejects_unknown_outcome() -> None:
    """PathEligibilityDecision rejects unknown outcome."""
    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            path_id="x",
            outcome="maybe",  # type: ignore[arg-type]
            reason="r",
        )


def test_path_decision_rejects_blank_path_id() -> None:
    """PathEligibilityDecision rejects blank path_id."""
    with pytest.raises(ValueError, match="path_id"):
        PathEligibilityDecision(path_id="", outcome="allowed", reason="r")


def test_path_decision_rejects_blank_reason() -> None:
    """PathEligibilityDecision rejects blank reason."""
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(path_id="x", outcome="allowed", reason="  ")


def _valid_estimate() -> WitnessEstimate:
    return WitnessEstimate(
        estimator_id="e",
        mean=0.0,
        uncertainty=0.0,
        support_status="supported",
        backend="b",
        n_qubits=1,
        n_shots_or_times=1,
    )


def test_materialised_krylov_allows_zero_lanczos() -> None:
    """MaterialisedKrylovProbe allows n_lanczos==0 with finite peak."""
    probe = MaterialisedKrylovProbe(
        estimate=_valid_estimate(),
        peak_complexity=0.0,
        n_lanczos=0,
        n_times=4,
        digest="b" * 64,
    )
    assert probe.n_lanczos == 0


def test_materialised_krylov_rejects_negative_lanczos() -> None:
    """MaterialisedKrylovProbe rejects n_lanczos < 0."""
    with pytest.raises(ValueError, match="n_lanczos"):
        MaterialisedKrylovProbe(
            estimate=_valid_estimate(),
            peak_complexity=1.0,
            n_lanczos=-1,
            n_times=4,
            digest="b" * 64,
        )


def test_materialised_krylov_rejects_invent_green() -> None:
    """MaterialisedKrylovProbe rejects invent_green_live_qpu."""
    with pytest.raises(ValueError, match="invent_green"):
        MaterialisedKrylovProbe(
            estimate=_valid_estimate(),
            peak_complexity=1.0,
            n_lanczos=2,
            n_times=4,
            digest="a" * 64,
            invent_green_live_qpu=True,
        )


def test_materialised_krylov_rejects_bad_digest() -> None:
    """MaterialisedKrylovProbe requires 64-char hex digest."""
    with pytest.raises(ValueError, match="digest"):
        MaterialisedKrylovProbe(
            estimate=_valid_estimate(),
            peak_complexity=1.0,
            n_lanczos=2,
            n_times=4,
            digest="nope",
        )


def test_materialised_krylov_rejects_non_finite_peak() -> None:
    """MaterialisedKrylovProbe rejects non-finite peak."""
    with pytest.raises(ValueError, match="peak_complexity"):
        MaterialisedKrylovProbe(
            estimate=_valid_estimate(),
            peak_complexity=float("nan"),
            n_lanczos=2,
            n_times=4,
            digest="c" * 64,
        )


def test_materialised_krylov_rejects_zero_times() -> None:
    """MaterialisedKrylovProbe rejects n_times < 1."""
    with pytest.raises(ValueError, match="n_times"):
        MaterialisedKrylovProbe(
            estimate=_valid_estimate(),
            peak_complexity=1.0,
            n_lanczos=2,
            n_times=0,
            digest="c" * 64,
        )


def test_materialised_otoc_rejects_invent_green_advantage() -> None:
    """MaterialisedOtocProbe rejects invent_green_otoc_advantage."""
    with pytest.raises(ValueError, match="invent_green"):
        MaterialisedOtocProbe(
            estimate=_valid_estimate(),
            final_otoc=0.5,
            lyapunov_estimate=None,
            scrambling_time=None,
            n_times=4,
            digest="a" * 64,
            invent_green_otoc_advantage=True,
        )


def test_materialised_otoc_rejects_invent_green_live_qpu() -> None:
    """MaterialisedOtocProbe rejects invent_green_live_qpu."""
    with pytest.raises(ValueError, match="invent_green"):
        MaterialisedOtocProbe(
            estimate=_valid_estimate(),
            final_otoc=0.1,
            lyapunov_estimate=None,
            scrambling_time=None,
            n_times=2,
            digest="c" * 64,
            invent_green_live_qpu=True,
        )


def test_materialised_otoc_rejects_non_finite_final() -> None:
    """MaterialisedOtocProbe rejects non-finite final_otoc."""
    with pytest.raises(ValueError, match="final_otoc"):
        MaterialisedOtocProbe(
            estimate=_valid_estimate(),
            final_otoc=float("inf"),
            lyapunov_estimate=None,
            scrambling_time=None,
            n_times=2,
            digest="c" * 64,
        )


def test_materialised_otoc_rejects_non_finite_lyapunov() -> None:
    """MaterialisedOtocProbe rejects non-finite lyapunov when present."""
    with pytest.raises(ValueError, match="lyapunov"):
        MaterialisedOtocProbe(
            estimate=_valid_estimate(),
            final_otoc=0.1,
            lyapunov_estimate=float("nan"),
            scrambling_time=None,
            n_times=2,
            digest="c" * 64,
        )


def test_materialised_otoc_rejects_non_finite_scrambling() -> None:
    """MaterialisedOtocProbe rejects non-finite scrambling_time when present."""
    with pytest.raises(ValueError, match="scrambling"):
        MaterialisedOtocProbe(
            estimate=_valid_estimate(),
            final_otoc=0.1,
            lyapunov_estimate=None,
            scrambling_time=float("nan"),
            n_times=2,
            digest="c" * 64,
        )


def test_materialised_otoc_rejects_zero_times() -> None:
    """MaterialisedOtocProbe rejects n_times < 1."""
    with pytest.raises(ValueError, match="n_times"):
        MaterialisedOtocProbe(
            estimate=_valid_estimate(),
            final_otoc=0.1,
            lyapunov_estimate=None,
            scrambling_time=None,
            n_times=0,
            digest="b" * 64,
        )


def test_materialised_otoc_rejects_bad_digest() -> None:
    """MaterialisedOtocProbe requires 64-char digest."""
    with pytest.raises(ValueError, match="digest"):
        MaterialisedOtocProbe(
            estimate=_valid_estimate(),
            final_otoc=0.1,
            lyapunov_estimate=None,
            scrambling_time=None,
            n_times=2,
            digest="x",
        )


def test_materialised_shadow_rejects_short_digest() -> None:
    """MaterialisedShadowProbe requires 64-char digest."""
    with pytest.raises(ValueError, match="digest"):
        MaterialisedShadowProbe(
            estimate=_valid_estimate(),
            observables={"z": 0.0},
            shadow_norm_bound=0.1,
            n_shots=10,
            digest="short",
        )


def test_materialised_shadow_rejects_negative_bound() -> None:
    """MaterialisedShadowProbe rejects negative shadow_norm_bound."""
    with pytest.raises(ValueError, match="shadow_norm_bound"):
        MaterialisedShadowProbe(
            estimate=_valid_estimate(),
            observables={"z": 0.0},
            shadow_norm_bound=-0.1,
            n_shots=10,
            digest="b" * 64,
        )


def test_materialised_shadow_rejects_empty_observables() -> None:
    """MaterialisedShadowProbe rejects empty observables map."""
    with pytest.raises(ValueError, match="observables"):
        MaterialisedShadowProbe(
            estimate=_valid_estimate(),
            observables={},
            shadow_norm_bound=0.1,
            n_shots=10,
            digest="b" * 64,
        )


def test_materialised_shadow_rejects_zero_shots() -> None:
    """MaterialisedShadowProbe rejects n_shots < 1."""
    with pytest.raises(ValueError, match="n_shots"):
        MaterialisedShadowProbe(
            estimate=_valid_estimate(),
            observables={"z": 0.0},
            shadow_norm_bound=0.1,
            n_shots=0,
            digest="b" * 64,
        )


def test_materialised_shadow_rejects_blank_observable_name() -> None:
    """MaterialisedShadowProbe rejects blank observable names."""
    with pytest.raises(ValueError, match="observable"):
        MaterialisedShadowProbe(
            estimate=_valid_estimate(),
            observables={"": 0.0},
            shadow_norm_bound=0.1,
            n_shots=10,
            digest="c" * 64,
        )


def test_materialised_shadow_rejects_non_finite_value() -> None:
    """MaterialisedShadowProbe rejects non-finite observable values."""
    with pytest.raises(ValueError, match="finite"):
        MaterialisedShadowProbe(
            estimate=_valid_estimate(),
            observables={"z": float("nan")},
            shadow_norm_bound=0.1,
            n_shots=10,
            digest="c" * 64,
        )


def test_materialised_shadow_rejects_invent_green() -> None:
    """MaterialisedShadowProbe rejects invent_green_live_qpu."""
    with pytest.raises(ValueError, match="invent_green"):
        MaterialisedShadowProbe(
            estimate=_valid_estimate(),
            observables={"z": 0.0},
            shadow_norm_bound=0.1,
            n_shots=10,
            digest="c" * 64,
            invent_green_live_qpu=True,
        )


def test_require_qubit_cap_rejects_zero() -> None:
    """Internal qubit-cap helper rejects n_qubits < 1."""
    import scpn_quantum_control.advanced_witnesses_product as mod

    with pytest.raises(ValueError, match="n_qubits must be >= 1"):
        mod._require_qubit_cap(0, label="t")


# ---------------------------------------------------------------------------
# Integrity fail-closed (one mutation per test)
# ---------------------------------------------------------------------------


def test_integrity_rejects_wrong_schema() -> None:
    """Integrity fails on schema drift."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["schema"] = "wrong.v0"
    with pytest.raises(ValueError, match="schema"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_otoc_advantage_policy_true() -> None:
    """Integrity fails if otoc_advantage_claim_policy is True."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["otoc_advantage_claim_policy"] = True
    with pytest.raises(ValueError, match="otoc_advantage"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_hardware_submit_on_capability() -> None:
    """Integrity fails if any capability allows hardware submit."""
    base = build_advanced_witnesses_product_registry()
    bad = dict(base)
    caps = _registry_rows(base, "capabilities")
    caps[0]["hardware_submit_allowed"] = True
    bad["capabilities"] = caps
    with pytest.raises(ValueError, match="hardware_submit"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_nonzero_blank_entry_count() -> None:
    """Integrity fails if blank_entry_count is not 0."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["blank_entry_count"] = 3
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_max_qubits_drift() -> None:
    """Integrity fails if max_witness_qubits drifts from constant."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["max_witness_qubits"] = 99
    with pytest.raises(ValueError, match="max_witness_qubits"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_missing_required_capability() -> None:
    """Integrity fails when a required capability row is removed."""
    base = build_advanced_witnesses_product_registry()
    missing = dict(base)
    capabilities = [
        c
        for c in _registry_rows(base, "capabilities")
        if c["capability_id"] != "krylov_complexity"
    ]
    missing["capabilities"] = capabilities
    missing["capability_count"] = len(capabilities)
    with pytest.raises(ValueError, match="missing|drift"):
        assert_advanced_witnesses_product_integrity(missing)


def test_integrity_rejects_bad_claim_boundary() -> None:
    """Integrity fails when claim_boundary is not the product boundary."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["claim_boundary"] = "not the product boundary"
    with pytest.raises(ValueError, match="claim_boundary"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_incomplete_glossary() -> None:
    """Integrity fails when glossary is missing required keys."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["glossary"] = {"only": "one"}
    with pytest.raises(ValueError, match="glossary"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_empty_ambient_inventory() -> None:
    """Integrity fails when ambient_inventory is empty."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["ambient_inventory"] = []
    with pytest.raises(ValueError, match="ambient_inventory"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_capability_count_mismatch() -> None:
    """Integrity fails when capability_count disagrees with list length."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["capability_count"] = 0
    with pytest.raises(ValueError, match="capability_count"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_boundary_count_mismatch() -> None:
    """Integrity fails when boundary_count disagrees with list length."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["boundary_count"] = 0
    with pytest.raises(ValueError, match="boundary_count"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_min_shadow_shots_drift() -> None:
    """Integrity fails when min_shadow_shots drifts."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["min_shadow_shots"] = 999
    with pytest.raises(ValueError, match="min_shadow_shots"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_max_demo_shadow_shots_drift() -> None:
    """Integrity fails when max_demo_shadow_shots drifts."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["max_demo_shadow_shots"] = 1
    with pytest.raises(ValueError, match="max_demo_shadow_shots"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_empty_capabilities() -> None:
    """Integrity fails on empty capabilities list."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["capabilities"] = []
    with pytest.raises(ValueError, match="capabilities"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_empty_boundaries() -> None:
    """Integrity fails on empty boundaries list."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["boundaries"] = []
    with pytest.raises(ValueError, match="boundaries"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_duplicate_capability_id() -> None:
    """Integrity fails on duplicate capability_id."""
    base = build_advanced_witnesses_product_registry()
    dup = dict(base)
    caps = _registry_rows(base, "capabilities")
    caps.append(dict(caps[0]))
    dup["capabilities"] = caps
    dup["capability_count"] = len(caps)
    with pytest.raises(ValueError, match="duplicate capability"):
        assert_advanced_witnesses_product_integrity(dup)


def test_integrity_rejects_boundary_fail_closed_false() -> None:
    """Integrity fails if a boundary is not fail_closed."""
    base = build_advanced_witnesses_product_registry()
    bf = dict(base)
    bounds = _registry_rows(base, "boundaries")
    bounds[0]["fail_closed"] = False
    bf["boundaries"] = bounds
    with pytest.raises(ValueError, match="fail_closed"):
        assert_advanced_witnesses_product_integrity(bf)


def test_integrity_rejects_non_mapping_capability_row() -> None:
    """Integrity fails if a capability row is not a mapping."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["capabilities"] = ["not-a-mapping"]
    bad["capability_count"] = 1
    with pytest.raises(ValueError, match="mapping"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_non_mapping_boundary_row() -> None:
    """Integrity fails if a boundary row is not a mapping."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["boundaries"] = [42]
    bad["boundary_count"] = 1
    with pytest.raises(ValueError, match="mapping"):
        assert_advanced_witnesses_product_integrity(bad)


def test_integrity_rejects_blank_capability_id_in_row() -> None:
    """Integrity fails when a capability_id is blank."""
    base = build_advanced_witnesses_product_registry()
    blank = dict(base)
    caps = _registry_rows(base, "capabilities")
    caps[0]["capability_id"] = "  "
    blank["capabilities"] = caps
    with pytest.raises(ValueError, match="blank|missing|drift"):
        assert_advanced_witnesses_product_integrity(blank)


def test_integrity_rejects_empty_ambient_symbol_in_row() -> None:
    """Integrity fails when ambient_symbol is empty on a capability row."""
    base = build_advanced_witnesses_product_registry()
    empty = dict(base)
    caps = _registry_rows(base, "capabilities")
    caps[0]["ambient_symbol"] = ""
    empty["capabilities"] = caps
    with pytest.raises(ValueError, match="ambient_symbol"):
        assert_advanced_witnesses_product_integrity(empty)


def test_integrity_rejects_blank_boundary_id_in_row() -> None:
    """Integrity fails when boundary_id is blank."""
    base = build_advanced_witnesses_product_registry()
    blank = dict(base)
    bounds = _registry_rows(base, "boundaries")
    bounds[0]["boundary_id"] = ""
    blank["boundaries"] = bounds
    with pytest.raises(ValueError, match="boundary_id"):
        assert_advanced_witnesses_product_integrity(blank)


def test_integrity_rejects_duplicate_boundary_id() -> None:
    """Integrity fails on duplicate boundary_id."""
    base = build_advanced_witnesses_product_registry()
    dup = dict(base)
    bounds = _registry_rows(base, "boundaries")
    bounds.append(dict(bounds[0]))
    dup["boundaries"] = bounds
    dup["boundary_count"] = len(bounds)
    with pytest.raises(ValueError, match="duplicate boundary"):
        assert_advanced_witnesses_product_integrity(dup)


def test_integrity_rejects_extra_capability_not_in_catalogue() -> None:
    """Integrity fails when registry lists a capability outside the catalogue."""
    base = build_advanced_witnesses_product_registry()
    extra = dict(base)
    caps = _registry_rows(base, "capabilities")
    caps.append(
        {
            "capability_id": "extra_not_in_catalogue",
            "kind": "ambient_inventory",
            "title": "x",
            "summary": "s",
            "ambient_module": "m",
            "ambient_symbol": "s",
            "hardware_submit_allowed": False,
            "support_posture": "metadata_only",
            "as_of": "2026-07-24",
            "claim_boundary": ADVANCED_WITNESSES_CLAIM_BOUNDARY,
        }
    )
    extra["capabilities"] = caps
    extra["capability_count"] = len(caps)
    with pytest.raises(ValueError, match="drift|extra"):
        assert_advanced_witnesses_product_integrity(extra)


def test_integrity_rejects_extra_boundary_not_in_catalogue() -> None:
    """Integrity fails when registry lists a boundary outside the catalogue."""
    base = build_advanced_witnesses_product_registry()
    extra = dict(base)
    bounds = _registry_rows(base, "boundaries")
    bounds.append(
        {
            "boundary_id": "extra_boundary",
            "kind": "otoc_advantage_claim",
            "title": "t",
            "summary": "s",
            "fail_closed": True,
            "claim_boundary": ADVANCED_WITNESSES_CLAIM_BOUNDARY,
        }
    )
    extra["boundaries"] = bounds
    extra["boundary_count"] = len(bounds)
    with pytest.raises(ValueError, match="boundary set drift|extra"):
        assert_advanced_witnesses_product_integrity(extra)


def test_integrity_rejects_glossary_not_mapping() -> None:
    """Integrity fails when glossary is not a mapping."""
    bad = dict(build_advanced_witnesses_product_registry())
    bad["glossary"] = ["not", "a", "mapping"]
    with pytest.raises(ValueError, match="glossary"):
        assert_advanced_witnesses_product_integrity(bad)


def test_claim_boundary_mentions_refuse() -> None:
    """Shared claim boundary is non-promotional and mentions refuse paths."""
    assert "refuse" in ADVANCED_WITNESSES_CLAIM_BOUNDARY.lower()
