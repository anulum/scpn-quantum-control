# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental exact feature set contract
"""Exact ideal-digital features from Qiskit, checked by two independent statevectors."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _llm_qpu_dev_chain import (
    _ARMS,
    _REVISIONS,
    _b64,
    _Chain,
    _dev_chain,
    _forge,
    _header,
    _reservoir,
    _sha,
    _worker,
)

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    EXACT_FEATURE_SET_SCHEMA,
    CircuitPlanSet,
    ExactFeatureSet,
    build_circuit_plan_set,
    canonical_bytes,
    decode_contract,
    decode_exact_feature_set,
    exact_expectations,
    validate_exact_feature_set,
)
from scpn_quantum_control.experimental.llm_qpu.data.digital_simulation import simulate_exact


def _plan_set(arms: tuple[str, ...] = _ARMS) -> tuple[_Chain, CircuitPlanSet]:
    """Return the dev chain and its plan set."""
    chain = _dev_chain()
    return chain, build_circuit_plan_set(
        _reservoir(), chain.compressed, chain.angles, arms=arms, revisions=_REVISIONS
    )


def _rebuilt(
    features: ExactFeatureSet, *, origin: str | None = None, **changes: Any
) -> ExactFeatureSet:
    """Return the feature set with changed fields and a header that hashes them."""
    names = ("plan_set_digest", "simulator", "simulator_version", "rows", "tolerance_hex")
    edited = {name: changes.get(name, getattr(features, name)) for name in names}
    content = {
        "schema": EXACT_FEATURE_SET_SCHEMA,
        "object_kind": "exact_feature_set",
        **edited,
        "rows": [row.to_wire() for row in edited["rows"]],
    }
    header = _header(
        "exact_feature_set",
        content,
        (edited["plan_set_digest"],),
        origin or features.header.data_origin,
    )
    return ExactFeatureSet(**edited, header=header)


def _request(chain: _Chain, plan_set: CircuitPlanSet, features: object) -> dict[str, object]:
    """Return one worker exact-feature request carrying the whole chain."""
    fitted = chain.fitted
    return {
        "op": "roundtrip_exact_feature_set",
        "task": chain.task.to_wire(),
        "split": chain.split.to_wire(),
        "model": chain.model.to_wire(),
        "latent": chain.latent.to_wire(),
        "latent_base64": _b64(chain.latent_payload),
        "compressor": fitted.artifact.to_wire(),
        "centering_base64": _b64(fitted.centering_payload),
        "projection_base64": _b64(fitted.projection_payload),
        "scales_base64": _b64(fitted.scales_payload),
        "compressed": chain.compressed.to_wire(),
        "angles_base64": _b64(chain.angles),
        "reservoir": _reservoir().to_wire(),
        "plan_set": plan_set.to_wire(),
        "features": features,
    }


def test_exact_features_agree_across_three_statevectors_and_physics(tmp_path: Path) -> None:
    """Qiskit, the contract's statevector and the worker agree, and no-coupling Z is cos(theta)."""
    chain, plan_set = _plan_set()
    features = simulate_exact(plan_set, 4, bases=("X", "Y", "Z"), revisions=_REVISIONS)
    validate_exact_feature_set(features, plan_set, 4)
    assert len(features.rows) == 4 and features.simulator == "qiskit_statevector"
    assert decode_exact_feature_set(canonical_bytes(features.to_wire())) == features
    angles = np.frombuffer(chain.angles, dtype="<f4").astype(np.float64).reshape(2, 4)
    for row in features.rows:
        values = dict(row.expectations)
        if row.arm_id == "encoding_plus_fields_no_coupling":
            cosines = np.cos(angles[row.row_index])
            assert np.allclose(values["Z"][:4], cosines, atol=1e-12)
            assert np.allclose(values["Z"][4:], cosines[:-1] * cosines[1:], atol=1e-12)
    for plan, row in zip(plan_set.plans, features.rows, strict=True):
        assert dict(row.expectations)["X"] == pytest.approx(
            exact_expectations(plan, 4, "X"), abs=1e-12
        )
    response = _worker(_request(chain, plan_set, features.to_wire()), tmp_path)
    assert response.returncode == 0, response.stdout.decode()
    payload = json.loads(response.stdout)
    assert payload["status"] == "validated_roundtrip_no_compute"
    assert payload["features_sha256"] == _sha(features.to_wire())
    single = simulate_exact(_plan_set(("xy",))[1], 4, bases=("Z",), revisions=_REVISIONS)
    assert [basis for basis, _ in single.rows[0].expectations] == ["Z"]


def test_exact_rows_and_sets_refuse_malformed_records() -> None:
    """Rows carry finite [-1, 1] values per sorted basis; sets name a bounded tolerance."""
    _, plan_set = _plan_set()
    valid = simulate_exact(plan_set, 4, bases=("X", "Z"), revisions=_REVISIONS)
    row = valid.rows[0]
    values = dict(row.expectations)["X"]
    for changes, reason in (
        ({"row_index": -1}, "^row index must be a bounded non-negative integer$"),
        ({"sample_id": ""}, "sample_id must be bounded nonempty text"),
        ({"plan_digest": "E" * 64}, "^plan digest must be a lowercase SHA-256 digest$"),
        ({"expectations": ()}, "^expectations must be \\(basis, values\\) pairs$"),
        ({"expectations": (("X",),)}, "^expectations must be \\(basis, values\\) pairs$"),
        (
            {"expectations": (("Z", values), ("X", values))},
            "^expectation bases must be sorted, unique",
        ),
        (
            {"expectations": (("W", values),)},
            "^expectation bases must be sorted, unique and X, Y or Z$",
        ),
        ({"expectations": (("X", values[:6]),)}, "^an n4 or n8 basis has 7 or 15 observables$"),
        (
            {"expectations": (("X", (2.0, *values[1:])),)},
            "^an ideal Pauli expectation is a finite float",
        ),
        (
            {"expectations": (("X", (1, *values[1:])),)},
            "^an ideal Pauli expectation is a finite float",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(row, **changes)
    stray: Any = (valid.rows[0], "row")
    set_cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"simulator": "aer_statevector"}, "^unknown ideal-digital simulator$"),
        ({"simulator_version": ""}, "simulator version must be bounded nonempty text"),
        ({"rows": ()}, "^rows must be a nonempty tuple of ExactRow records$"),
        ({"rows": stray}, "^rows must be a nonempty tuple of ExactRow records$"),
        ({"rows": (valid.rows[0], valid.rows[0])}, "^rows must name each plan once$"),
        (
            {"tolerance_hex": (0.0).hex()},
            "^tolerance must be a canonical hex float in \\(0, 1e-9\\]$",
        ),
        ({"tolerance_hex": (1e-8).hex()}, "^tolerance must be a canonical hex float in"),
        ({"tolerance_hex": "0x1p-40"}, "^tolerance must be a canonical hex float in"),
        ({"tolerance_hex": "tiny"}, "^tolerance must be a canonical hex float in"),
        ({"plan_set_digest": "E" * 64}, "^plan set digest must be a lowercase SHA-256 digest$"),
    )
    for changes, reason in set_cases:
        with pytest.raises(ValueError, match=reason):
            replace(valid, **changes)
    assert _rebuilt(valid, tolerance_hex=(1e-9).hex()).tolerance_hex == (1e-9).hex()
    with pytest.raises(ValueError, match="^ExactFeatureSet requires its exact artifact header$"):
        replace(valid, header=replace(valid.header, object_kind="planned_experiment"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(valid, rows=valid.rows[:1])


def test_exact_validation_refuses_another_plan_set_origin_rows_or_values() -> None:
    """Each row follows its plan and matches the independent statevector within tolerance."""
    _, plan_set = _plan_set()
    valid = simulate_exact(plan_set, 4, bases=("X", "Y", "Z"), revisions=_REVISIONS)
    other = _plan_set(("xy",))[1]
    rows = valid.rows
    moved = replace(rows[0], sample_id="dev-sample-9")
    x = dict(rows[0].expectations)
    nudged = replace(
        rows[0],
        expectations=(("X", (x["X"][0] + 1e-6, *x["X"][1:])), ("Y", x["Y"]), ("Z", x["Z"])),
    )
    validate_exact_feature_set(valid, plan_set, 4)
    for features, size, reason in (
        (
            _rebuilt(valid, plan_set_digest=_sha(other.to_wire())),
            4,
            "^the features name a different plan set$",
        ),
        (_rebuilt(valid, origin="owner_dataset"), 4, "^feature origin differs from the plan set$"),
        (_rebuilt(valid, rows=rows[:3]), 4, "^every plan needs exactly one exact row$"),
        (_rebuilt(valid, rows=(moved, *rows[1:])), 4, "^an exact row does not follow its plan$"),
        (valid, 8, "^an exact row has the wrong observable count$"),
        (
            _rebuilt(valid, rows=(nudged, *rows[1:])),
            4,
            "^an exact expectation differs from the independent",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            validate_exact_feature_set(features, plan_set, size)
    not_a_set: Any = "features"
    with pytest.raises(ValueError, match="^exact validation needs an ExactFeatureSet$"):
        validate_exact_feature_set(not_a_set, plan_set, 4)


def test_exact_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires are refused by the decoder and the worker; decode_contract stays closed."""
    chain, plan_set = _plan_set()
    features = simulate_exact(plan_set, 4, bases=("X", "Y", "Z"), revisions=_REVISIONS)
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(features.to_wire()))
    decoder_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "^ExactFeatureSet fields mismatch$"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.exact_feature_set.v2",
            "^unknown ExactFeatureSet schema$",
        ),
        (("rows",), {"first": 1}, "^rows must be a list on wire$"),
        (("rows", 0, "unexpected"), True, "^ExactRow fields mismatch$"),
        (
            ("rows", 0, "expectations", 0),
            "X",
            "^expectations must be \\[basis, values\\] pairs on wire$",
        ),
        (("rows", 0, "expectations", 0, 1, 0), "0.5", "^expectations must be f64 tags on wire$"),
    )
    for path, value, reason in decoder_cases:
        with pytest.raises(ValueError, match=reason):
            decode_exact_feature_set(canonical_bytes(_forge(features.to_wire(), path, value)))
    worker_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "ExactFeatureSet fields mismatch"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.exact_feature_set.v2",
            "unknown ExactFeatureSet schema",
        ),
        (("simulator",), "aer_statevector", "unknown ideal-digital simulator"),
        (("simulator_version",), "", "simulator version must be bounded normalized text"),
        (("tolerance_hex",), (1e-8).hex(), "tolerance must be bounded canonical hex float"),
        (("plan_set_digest",), "7" * 64, "the features name a different plan set"),
        (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
    )
    for path, forgery, refusal in worker_cases:
        response = _worker(
            _request(chain, plan_set, _forge(features.to_wire(), path, forgery)), tmp_path
        )
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    rows = features.rows
    x = dict(rows[0].expectations)
    nudged = replace(
        rows[0],
        expectations=(("X", (x["X"][0] + 1e-6, *x["X"][1:])), ("Y", x["Y"]), ("Z", x["Z"])),
    )
    consistent: tuple[tuple[ExactFeatureSet, str], ...] = (
        (_rebuilt(features, rows=rows[:3]), "every plan needs exactly one exact row"),
        (
            _rebuilt(features, rows=(replace(rows[0], sample_id="dev-sample-9"), *rows[1:])),
            "does not follow its plan",
        ),
        (_rebuilt(features, rows=(nudged, *rows[1:])), "differs from the independent statevector"),
    )
    for edited, refusal in consistent:
        response = _worker(_request(chain, plan_set, edited.to_wire()), tmp_path)
        assert response.returncode == 2, refusal
        assert refusal in json.loads(response.stdout)["reason"], refusal
    missing = _request(chain, plan_set, features.to_wire())
    del missing["features"]
    response = _worker(missing, tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "exact request fields mismatch"
