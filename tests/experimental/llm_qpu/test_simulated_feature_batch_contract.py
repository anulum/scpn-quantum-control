# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental simulated feature batch contract
"""Features of seeded simulated counts, recomputed exactly here and in the worker."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _llm_qpu_dev_chain import (
    _REVISIONS,
    _forge,
    _header,
    _planned,
    _request,
    _sha,
    _spec,
    _worker,
)

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    SIMULATED_FEATURE_BATCH_SCHEMA,
    SimulatedCountSet,
    SimulatedFeatureBatch,
    canonical_bytes,
    decode_contract,
    decode_simulated_feature_batch,
    estimate_simulated_features,
    validate_simulated_feature_batch,
)
from scpn_quantum_control.experimental.llm_qpu.data.digital_simulation import simulate_sampled

_SEED = "5eed" * 16


def _counts(planned: Any) -> SimulatedCountSet:
    """Return the seeded counts of a planned experiment."""
    return simulate_sampled(
        planned.experiment, planned.plan_set, seed_hex=_SEED, revisions=_REVISIONS
    )


def _rebuilt(
    batch: SimulatedFeatureBatch, *, origin: str | None = None, **changes: Any
) -> SimulatedFeatureBatch:
    """Return the batch with changed fields and a header that hashes them."""
    names = ("simulated_count_set_digest", "estimator", "basis_id", "observable_ids", "rows")
    edited = {name: changes.get(name, getattr(batch, name)) for name in names}
    content = {
        "schema": SIMULATED_FEATURE_BATCH_SCHEMA,
        "object_kind": "simulated_feature_batch",
        **edited,
        "observable_ids": list(edited["observable_ids"]),
        "rows": [row.to_wire() for row in edited["rows"]],
    }
    header = _header(
        "simulated_feature_batch",
        content,
        (edited["simulated_count_set_digest"],),
        origin or batch.header.data_origin,
    )
    return SimulatedFeatureBatch(**edited, header=header)


def _feature_request(
    planned: Any, counts: SimulatedCountSet, features: object
) -> dict[str, object]:
    """Return one worker simulated-feature request carrying the whole chain."""
    request = _request(planned, planned.spec.to_wire(), planned.experiment.to_wire())
    return {
        **request,
        "op": "roundtrip_simulated_feature_batch",
        "counts": counts.to_wire(),
        "features": features,
    }


def _small() -> Any:
    """Return a one-arm, Z-only experiment so a worker request stays inside its byte bound."""
    return _planned(arms=("xy",), spec=_spec(bases=("Z",)))


def test_features_cover_each_basis_and_sentinels_are_exact() -> None:
    """Every basis gets its cells in order; sentinel features are the exact basis-state values."""
    planned = _planned()
    counts = _counts(planned)
    ordinals = {}
    for basis in "XYZ":
        batch = estimate_simulated_features(
            counts, planned.experiment, basis_id=basis, revisions=_REVISIONS
        )
        validate_simulated_feature_batch(batch, counts, planned.experiment)
        assert decode_simulated_feature_batch(canonical_bytes(batch.to_wire())) == batch
        ordinals[basis] = [row.cell_ordinal for row in batch.rows]
        assert batch.observable_ids[0] == f"{basis}0" and len(batch.observable_ids) == 7
    assert ordinals == {"X": [0, 3, 6, 9], "Y": [1, 4, 7, 10], "Z": [2, 5, 8, 11, 12, 13, 14, 15]}
    z = estimate_simulated_features(counts, planned.experiment, basis_id="Z", revisions=_REVISIONS)
    for row in z.rows[4:]:
        one = row.key.source_sample_id == "sentinel_one"
        assert row.values == ((-1.0,) * 4 if one else (1.0,) * 4) + (1.0,) * 3
        assert row.covariance == tuple((0.0,) * 7 for _ in range(7))
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(z.to_wire()))


def test_the_worker_recomputes_every_row_and_refuses_edits(tmp_path: Path) -> None:
    """The stdlib worker revalidates the count chain and recomputes each row with its own estimator."""
    planned = _small()
    counts = _counts(planned)
    batch = estimate_simulated_features(
        counts, planned.experiment, basis_id="Z", revisions=_REVISIONS
    )
    response = _worker(_feature_request(planned, counts, batch.to_wire()), tmp_path)
    assert response.returncode == 0, response.stdout.decode()
    assert json.loads(response.stdout)["features_sha256"] == _sha(batch.to_wire())
    forgeries: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "SimulatedFeatureBatch fields mismatch"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.simulated_feature_batch.v2",
            "unknown SimulatedFeatureBatch schema",
        ),
        (("estimator",), "quasi_probability_v1", "unknown feature estimator"),
        (("basis_id",), "W", "feature basis must be X, Y or Z"),
        (
            ("simulated_count_set_digest",),
            "7" * 64,
            "features name a different simulated count set",
        ),
        (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
    )
    for path, value, refusal in forgeries:
        response = _worker(
            _feature_request(planned, counts, _forge(batch.to_wire(), path, value)), tmp_path
        )
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    rows = batch.rows
    nudged = replace(rows[0], values=(0.5, *(rows[0].values or ())[1:]))
    swapped = replace(rows[0], key=rows[1].key)
    renamed = replace(rows[0], measurement_request_digest=rows[1].measurement_request_digest)
    gapped = replace(
        rows[0], shots=None, values=None, covariance=None, missing_reason="simulator offline"
    )
    consistent: tuple[tuple[SimulatedFeatureBatch, str], ...] = (
        (
            _rebuilt(batch, rows=rows[1:]),
            "feature rows must cover every cell measured in this basis",
        ),
        (
            _rebuilt(batch, observable_ids=tuple(f"X{i}" for i in range(7))),
            "feature columns differ",
        ),
        (
            _rebuilt(batch, rows=(nudged, *rows[1:])),
            "a feature row differs from its simulated counts",
        ),
        (
            _rebuilt(batch, rows=(swapped, *rows[1:])),
            "feature row key differs from the planned cell",
        ),
        (
            _rebuilt(batch, rows=(renamed, *rows[1:])),
            "feature row names another measurement request",
        ),
    )
    for edited, refusal in consistent:
        response = _worker(_feature_request(planned, counts, edited.to_wire()), tmp_path)
        assert response.returncode == 2, refusal
        assert refusal in json.loads(response.stdout)["reason"], refusal
    forged = batch.to_wire()
    forged_rows: Any = forged["rows"]
    forged_rows[0] = gapped.to_wire()
    content = {name: value for name, value in forged.items() if name != "header"}
    forged_header: Any = forged["header"]
    forged_header["content_digest"] = _sha(content)
    response = _worker(_feature_request(planned, counts, forged), tmp_path)
    assert json.loads(response.stdout)["reason"] == "a simulated cell is never missing"
    missing = _feature_request(planned, counts, batch.to_wire())
    del missing["features"]
    assert (
        json.loads(_worker(missing, tmp_path).stdout)["reason"]
        == "simulated-feature request fields mismatch"
    )


def test_batches_refuse_malformed_records() -> None:
    """Estimator, basis, columns, rows, gaps, order, width and header are each checked."""
    planned = _planned()
    batch = estimate_simulated_features(
        _counts(planned), planned.experiment, basis_id="X", revisions=_REVISIONS
    )
    rows = batch.rows
    gap = replace(
        rows[0], shots=None, values=None, covariance=None, missing_reason="simulator offline"
    )
    narrow = replace(
        rows[0],
        values=(rows[0].values or ())[:6],
        covariance=tuple(r[:6] for r in rows[0].covariance or ())[:6],
    )
    stray: Any = (rows[0], "row")
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        (
            {"simulated_count_set_digest": "E" * 64},
            "^simulated count set digest must be a lowercase",
        ),
        ({"estimator": "quasi_probability_v1"}, "^unknown feature estimator$"),
        ({"basis_id": "W"}, "^feature basis must be X, Y or Z$"),
        ({"observable_ids": ()}, "^observable IDs must be a nonempty tuple of at most 64$"),
        ({"observable_ids": ("X0",) * 7}, "^observable IDs must be unique$"),
        ({"rows": ()}, "^feature rows must be a nonempty tuple of FeatureRow records$"),
        ({"rows": stray}, "^feature rows must be a nonempty tuple of FeatureRow records$"),
        ({"rows": (gap, *rows[1:])}, "^a simulated cell is never missing$"),
        (
            {"rows": (rows[1], rows[0], *rows[2:])},
            "^feature rows must follow the experiment's cell order$",
        ),
        ({"rows": (narrow, *rows[1:])}, "^every row has one value per observable$"),
    )
    for changes, reason in cases:
        with pytest.raises(ValueError, match=reason):
            replace(batch, **changes)
    with pytest.raises(
        ValueError, match="^SimulatedFeatureBatch requires its simulated artifact header$"
    ):
        replace(batch, header=replace(batch.header, object_kind="feature_batch"))
    decoder: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "^SimulatedFeatureBatch fields mismatch$"),
        (("object_kind",), "feature_batch", "^unknown SimulatedFeatureBatch schema$"),
        (("rows",), {"first": 1}, "^observable IDs and rows must be lists on wire$"),
    )
    for path, value, reason in decoder:
        with pytest.raises(ValueError, match=reason):
            decode_simulated_feature_batch(canonical_bytes(_forge(batch.to_wire(), path, value)))


def test_validation_refuses_other_counts_experiments_origins_rows_and_values() -> None:
    """A batch follows its own counts and experiment, every basis cell and the recomputed values."""
    planned, other = _planned(), _planned(arms=("xy",))
    counts = _counts(planned)
    batch = estimate_simulated_features(
        counts, planned.experiment, basis_id="X", revisions=_REVISIONS
    )
    rows = batch.rows
    nudged = replace(rows[0], values=(0.5, *(rows[0].values or ())[1:]))
    cases: tuple[tuple[SimulatedFeatureBatch, Any, Any, str], ...] = (
        (
            batch,
            _counts(other),
            planned.experiment,
            "^features name a different simulated count set$",
        ),
        (batch, counts, other.experiment, "^the counts name a different planned experiment$"),
        (
            _rebuilt(batch, origin="owner_dataset"),
            counts,
            planned.experiment,
            "^feature origin differs from the simulated",
        ),
        (
            _rebuilt(batch, rows=rows[1:]),
            counts,
            planned.experiment,
            "^feature rows must cover every cell measured",
        ),
        (
            _rebuilt(
                batch,
                observable_ids=tuple(f"Y{i}" for i in range(4))
                + tuple(f"Y{i}Y{i + 1}" for i in range(3)),
            ),
            counts,
            planned.experiment,
            "^feature columns differ from the cell's observables$",
        ),
        (
            _rebuilt(batch, rows=(nudged, *rows[1:])),
            counts,
            planned.experiment,
            "^a feature row differs from its simulated",
        ),
    )
    for edited, used_counts, experiment, reason in cases:
        with pytest.raises(ValueError, match=reason):
            validate_simulated_feature_batch(edited, used_counts, experiment)
    not_a_batch: Any = "features"
    with pytest.raises(ValueError, match="^feature validation needs a SimulatedFeatureBatch$"):
        validate_simulated_feature_batch(not_a_batch, counts, planned.experiment)
    with pytest.raises(ValueError, match="^the counts name a different planned experiment$"):
        estimate_simulated_features(counts, other.experiment, basis_id="X", revisions=_REVISIONS)
    z_only = _small()
    with pytest.raises(ValueError, match="^the experiment measures no cell in this basis$"):
        estimate_simulated_features(
            _counts(z_only), z_only.experiment, basis_id="X", revisions=_REVISIONS
        )
