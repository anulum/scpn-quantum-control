# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental simulated count set contract
"""Seeded simulated counts, regenerated exactly by the contract and by the worker."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    PLANNED_EXPERIMENT_SCHEMA,
    SIMULATED_COUNT_SET_SCHEMA,
    ArrayDescriptor,
    ArtifactHeader,
    CircuitPlanSet,
    CompressedLatentBatch,
    LatentBatch,
    MeasurementSpec,
    ModelDescriptor,
    PlannedExperiment,
    ReservoirSpec,
    SimulatedCell,
    SimulatedCountSet,
    SplitManifest,
    TaskSpec,
    build_circuit_plan_set,
    canonical_bytes,
    cell_probabilities,
    decode_contract,
    decode_simulated_count_set,
    draw_counts,
    estimate_sampled_basis,
    exact_expectations,
    plan_measurements,
    validate_simulated_count_set,
)
from scpn_quantum_control.experimental.llm_qpu.contracts.sampled import experiment_circuits
from scpn_quantum_control.experimental.llm_qpu.data import digital_simulation
from scpn_quantum_control.experimental.llm_qpu.data.compression_fit import (
    FittedCompressor,
    compress_latents,
    fit_train_compressor,
)
from scpn_quantum_control.experimental.llm_qpu.data.digital_simulation import simulate_sampled

_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_REVISIONS = (_BASE, _REVISION)
_ARMS = ("encoding_plus_fields_no_coupling", "xy")
_TRAIN = tuple(f"train-{index:02d}" for index in range(10))


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _header(
    kind: str, content: dict[str, object], parents: tuple[str, ...], origin: str
) -> ArtifactHeader:
    """Return a design-only header bound to the content and parents."""
    return ArtifactHeader(
        kind,
        _sha(content),
        tuple(sorted(parents)),
        _BASE,
        _REVISION,
        "offline_design",
        origin,
        "design_only",
    )


def _reservoir(
    fields: tuple[str, ...] = (
        "0x1.999999999999ap-4",
        "-0x1.999999999999ap-4",
        "0x1.999999999999ap-3",
        "-0x1.999999999999ap-3",
    ),
) -> ReservoirSpec:
    """Return the n4 open-chain reservoir of the engineering preview."""
    return ReservoirSpec(
        kernel_id="xy_static_digital_v1",
        n_qubits=4,
        edges=((0, 1), (1, 2), (2, 3)),
        couplings_hex=("0x1.0000000000000p-1",) * 3,
        fields_hex=fields,
        tau_hex="0x1.0000000000000p-1",
        repetitions=1,
        initial_state="zero",
        angle_unit="radian",
        gate_order="encode_then_fields_even_odd_v1",
    )


@dataclass(frozen=True)
class _Chain:
    """Packaged compression of one train batch, with every record and payload."""

    task: TaskSpec
    split: SplitManifest
    model: ModelDescriptor
    latent: LatentBatch
    latent_payload: bytes
    fitted: FittedCompressor
    compressed: CompressedLatentBatch
    angles: bytes


def _chain() -> _Chain:
    """Fit and compress a ten-row synthetic train batch to four angles per row."""
    task_fields: dict[str, Any] = {
        "task_id": "synthetic-plan-task",
        "objective": "Predict a label from a prefix",
        "source_kind": "synthetic_classical",
        "target_origin": "classical_generator",
        "label_schema_digest": "a" * 64,
        "causal_cutoff": 0,
        "primary_metric": "accuracy",
        "group_definition": "One generated scene per source group",
    }
    task_content = {
        "schema": "scpn.experimental.llm_qpu.task_spec.v2",
        "object_kind": "task_spec",
        **task_fields,
    }
    task = TaskSpec(
        **task_fields,
        header=_header("task_spec", task_content, ("a" * 64,), "synthetic_classical"),
    )
    split_fields: dict[str, Any] = {
        "task_digest": _sha(task.to_wire()),
        "dataset_digest": "b" * 64,
        "train_groups": _TRAIN,
        "dev_groups": ("dev-00", "dev-01"),
        "test_groups": ("test-00",),
        "seed": 19,
        "dedup_rule": "exact_source_digest",
        "test_target_custodian": "separate_locked_evaluator",
        "transform_fit_split": "train_only",
    }
    split_content = {
        "schema": "scpn.experimental.llm_qpu.split_manifest.v2",
        "object_kind": "split_manifest",
        **{k: list(v) if k.endswith("_groups") else v for k, v in split_fields.items()},
    }
    split = SplitManifest(
        **split_fields,
        header=_header(
            "split_manifest",
            split_content,
            (split_fields["task_digest"], "b" * 64),
            "synthetic_classical",
        ),
    )
    model_fields: dict[str, Any] = {
        "model_id": "synthetic-hidden-source",
        "checkpoint_digest": "c" * 64,
        "tokenizer_digest": "d" * 64,
        "chat_template_digest": "e" * 64,
        "runtime_build_digest": "f" * 64,
        "loader_id": "fixture-loader",
        "quantization": "none",
        "tensor_dtype": "float32",
        "block_count": 4,
        "hidden_width": 8,
        "tap_block_index": 2,
        "tap_stream": "residual_hidden_state",
        "tap_boundary": "before_norm",
        "probe_evidence_digest": "1" * 64,
    }
    model_content = {
        "schema": "scpn.experimental.llm_qpu.model_descriptor.v1",
        "object_kind": "model_descriptor",
        **model_fields,
    }
    names = (
        "checkpoint_digest",
        "tokenizer_digest",
        "chat_template_digest",
        "runtime_build_digest",
        "probe_evidence_digest",
    )
    model = ModelDescriptor(
        **model_fields,
        header=_header(
            "model_descriptor",
            model_content,
            tuple(model_fields[n] for n in names),
            "owner_checkpoint",
        ),
    )
    values = np.random.default_rng(7).normal(size=(10, 8)).astype(np.float32)
    payload = np.ascontiguousarray(values, dtype="<f4").tobytes()
    tensor = ArrayDescriptor("<f4", (10, 8), len(payload), hashlib.sha256(payload).hexdigest())
    latent_fields: dict[str, Any] = {
        "task_digest": _sha(task.to_wire()),
        "split_digest": _sha(split.to_wire()),
        "model_digest": _sha(model.to_wire()),
        "split_name": "train",
        "layout": "contextual",
        "sample_ids": tuple(f"sample-{i:02d}" for i in range(10)),
        "source_ids": tuple(f"source-{i:02d}" for i in range(10)),
        "group_ids": _TRAIN,
        "lengths": (1,) * 10,
        "mask": ((True,),) * 10,
        "token_positions": ((0,),) * 10,
        "answer_start_positions": ((1,),) * 10,
        "tap_block_index": 2,
        "tap_boundary": "before_norm",
        "tensor": tensor,
    }
    nested = ("mask", "token_positions", "answer_start_positions")
    content = {
        "schema": "scpn.experimental.llm_qpu.latent_batch.v1",
        "object_kind": "latent_batch",
        **{
            k: v.to_wire()
            if k == "tensor"
            else [list(r) for r in v]
            if k in nested
            else list(v)
            if type(v) is tuple
            else v
            for k, v in latent_fields.items()
        },
    }
    parents = (
        latent_fields["task_digest"],
        latent_fields["split_digest"],
        latent_fields["model_digest"],
        tensor.sha256,
    )
    latent = LatentBatch(
        **latent_fields, header=_header("latent_batch", content, parents, "synthetic_classical")
    )
    fitted = fit_train_compressor(
        task, split, model, latent, payload, output_width=4, fit_seed=3, revisions=_REVISIONS
    )
    compressed, angles = compress_latents(
        task, split, model, latent, payload, fitted, revisions=_REVISIONS
    )
    return _Chain(task, split, model, latent, payload, fitted, compressed, angles)


def _dev_chain() -> _Chain:
    """Compress a two-row dev batch with the train-fitted compressor."""
    train = _chain()
    values = np.random.default_rng(13).normal(size=(2, 8)).astype(np.float32)
    payload = np.ascontiguousarray(values, dtype="<f4").tobytes()
    tensor = ArrayDescriptor("<f4", (2, 8), len(payload), hashlib.sha256(payload).hexdigest())
    base = train.latent
    fields: dict[str, Any] = {
        "task_digest": base.task_digest,
        "split_digest": base.split_digest,
        "model_digest": base.model_digest,
        "split_name": "dev",
        "layout": "contextual",
        "sample_ids": ("dev-sample-0", "dev-sample-1"),
        "source_ids": ("dev-source-0", "dev-source-1"),
        "group_ids": ("dev-00", "dev-01"),
        "lengths": (1, 1),
        "mask": ((True,), (True,)),
        "token_positions": ((0,), (0,)),
        "answer_start_positions": ((1,), (1,)),
        "tap_block_index": 2,
        "tap_boundary": "before_norm",
        "tensor": tensor,
    }
    nested = ("mask", "token_positions", "answer_start_positions")
    content = {
        "schema": "scpn.experimental.llm_qpu.latent_batch.v1",
        "object_kind": "latent_batch",
        **{
            k: v.to_wire()
            if k == "tensor"
            else [list(r) for r in v]
            if k in nested
            else list(v)
            if type(v) is tuple
            else v
            for k, v in fields.items()
        },
    }
    parents = (base.task_digest, base.split_digest, base.model_digest, tensor.sha256)
    latent = LatentBatch(
        **fields, header=_header("latent_batch", content, parents, "synthetic_classical")
    )
    compressed, angles = compress_latents(
        train.task, train.split, train.model, latent, payload, train.fitted, revisions=_REVISIONS
    )
    return _Chain(
        train.task, train.split, train.model, latent, payload, train.fitted, compressed, angles
    )


def _spec(**changes: Any) -> MeasurementSpec:
    """Return the engineering-preview readout: three bases, two sentinels in slots A and B."""
    fields: dict[str, Any] = {
        "experiment_id": "llm_qpu_n4_engineering_v1",
        "bases": ("X", "Y", "Z"),
        "data_shots": 256,
        "logical_to_clbit": (0, 1, 2, 3),
        "display_order_clbits": (3, 2, 1, 0),
        "register_name": "readout",
        "sentinel_states": ("one", "zero"),
        "sentinel_slots": ("A", "B"),
        "sentinel_shots": 1024,
        "replicate_id": "r0",
        "planned_epoch_id": "unbound",
        **changes,
    }
    return MeasurementSpec(**fields)


@dataclass(frozen=True)
class _Planned:
    """One dev chain, its plan set and its planned experiment."""

    chain: _Chain
    plan_set: CircuitPlanSet
    spec: MeasurementSpec
    experiment: PlannedExperiment


def _planned(arms: tuple[str, ...] = _ARMS, spec: MeasurementSpec | None = None) -> _Planned:
    """Plan the dev chain's cells under the spec."""
    chain = _dev_chain()
    plan_set = build_circuit_plan_set(
        _reservoir(), chain.compressed, chain.angles, arms=arms, revisions=_REVISIONS
    )
    used = spec or _spec()
    experiment = plan_measurements(plan_set, _reservoir(), used, revisions=_REVISIONS)
    return _Planned(chain, plan_set, used, experiment)


def _rebuilt(
    experiment: PlannedExperiment, *, origin: str | None = None, **changes: Any
) -> PlannedExperiment:
    """Return the experiment with changed fields and a header that hashes them."""
    names = ("plan_set_digest", "measurement_spec_digest", "sentinels", "cells", "measurements")
    edited = {name: changes.get(name, getattr(experiment, name)) for name in names}
    content = {
        "schema": PLANNED_EXPERIMENT_SCHEMA,
        "object_kind": "planned_experiment",
        "plan_set_digest": edited["plan_set_digest"],
        "measurement_spec_digest": edited["measurement_spec_digest"],
        "sentinels": [s.to_wire() for s in edited["sentinels"]],
        "cells": [c.to_wire() for c in edited["cells"]],
        "measurements": [m.to_wire() for m in edited["measurements"]],
    }
    parents = (edited["plan_set_digest"], edited["measurement_spec_digest"])
    header = _header(
        "planned_experiment", content, parents, origin or experiment.header.data_origin
    )
    return PlannedExperiment(**edited, header=header)


def _b64(payload: bytes) -> str:
    """Return a base64 string of the payload."""
    return base64.b64encode(payload).decode("ascii")


def _worker(request: dict[str, object], tmp_path: Path) -> subprocess.CompletedProcess[bytes]:
    """Run the standalone protocol worker on one request with poisoned provider tokens."""
    return subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "IQM_TOKEN": "poison", "IBM_QUANTUM_TOKEN": "poison"},
        capture_output=True,
        timeout=30,
        check=False,
    )


def _request(planned: _Planned, spec: object, experiment: object) -> dict[str, object]:
    """Return one worker experiment request carrying the whole chain."""
    chain, fitted = planned.chain, planned.chain.fitted
    return {
        "op": "roundtrip_planned_experiment",
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
        "plan_set": planned.plan_set.to_wire(),
        "measurement_spec": spec,
        "experiment": experiment,
    }


def _forge(wire: dict[str, Any], path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    """Return the wire with one nested field replaced."""
    holder: Any = wire
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return wire


_SEED = "5eed" * 16


def _counts(planned: _Planned, seed_hex: str = _SEED) -> SimulatedCountSet:
    """Return the seeded counts of the planned experiment."""
    return simulate_sampled(
        planned.experiment, planned.plan_set, seed_hex=seed_hex, revisions=_REVISIONS
    )


def _rebuilt_counts(
    counts: SimulatedCountSet, *, origin: str | None = None, **changes: Any
) -> SimulatedCountSet:
    """Return the count set with changed fields and a header that hashes them."""
    names = (
        "planned_experiment_digest",
        "sampler",
        "seed_hex",
        "simulator",
        "simulator_version",
        "cells",
    )
    edited = {name: changes.get(name, getattr(counts, name)) for name in names}
    content = {
        "schema": SIMULATED_COUNT_SET_SCHEMA,
        "object_kind": "simulated_count_set",
        **edited,
        "cells": [cell.to_wire() for cell in edited["cells"]],
    }
    parents = (edited["planned_experiment_digest"],)
    header = _header("simulated_count_set", content, parents, origin or counts.header.data_origin)
    return SimulatedCountSet(**edited, header=header)


def _count_request(planned: _Planned, counts: object) -> dict[str, object]:
    """Return one worker simulated-count request carrying the whole chain."""
    request = _request(planned, planned.spec.to_wire(), planned.experiment.to_wire())
    return {**request, "op": "roundtrip_simulated_count_set", "counts": counts}


def _moved(cell: SimulatedCell) -> SimulatedCell:
    """Return the cell with one shot moved to another outcome, keeping the total."""
    counts = dict(cell.counts)
    first = next(key for key, count in cell.counts if count > 1)
    other = next(key for key in ("0000", "1111", "0101") if key != first)
    counts[first] -= 1
    counts[other] = counts.get(other, 0) + 1
    return replace(cell, counts=tuple(sorted(counts.items())))


def _sigma_gaps(planned: _Planned, seed_hex: str) -> list[float]:
    """Return |estimate - exact| / sigma for every data-cell observable of the experiment."""
    plans = {_sha(plan.to_wire()): plan for plan in planned.plan_set.plans}
    circuits = experiment_circuits(planned.experiment, planned.plan_set)
    gaps = []
    for ordinal, (cell, measurement) in enumerate(
        zip(planned.experiment.cells, planned.experiment.measurements, strict=True)
    ):
        if cell.role != "data":
            continue
        probabilities = cell_probabilities(circuits[measurement.circuit_plan_digest], measurement)
        drawn = draw_counts(probabilities, measurement, seed_hex=seed_hex, ordinal=ordinal)
        virtual = replace(measurement, physical_qubits=(0, 1, 2, 3))
        estimate = estimate_sampled_basis(virtual, dict(drawn), origin="digital_sampled")
        exact = exact_expectations(plans[measurement.circuit_plan_digest], 4, measurement.basis_id)
        covariance = estimate.covariance_of_mean
        assert covariance is not None
        for index, (value, ideal) in enumerate(zip(estimate.expectations, exact, strict=True)):
            sigma = covariance[index][index] ** 0.5
            gaps.append(abs(value - ideal) / sigma if sigma > 0 else abs(value - ideal) * 1e12)
    return gaps


def test_seeded_counts_reproduce_here_and_in_the_worker(tmp_path: Path) -> None:
    """The sampler is deterministic, seed-dependent, exact on sentinels and replayed by the worker."""
    planned = _planned()
    counts = _counts(planned)
    validate_simulated_count_set(counts, planned.experiment, planned.plan_set)
    assert len(counts.cells) == len(planned.experiment.cells) == 16
    assert decode_simulated_count_set(canonical_bytes(counts.to_wire())) == counts
    assert _counts(planned) == counts
    assert _counts(planned, "0" * 64).cells != counts.cells
    for cell, planned_cell in zip(counts.cells, planned.experiment.cells, strict=True):
        if planned_cell.role == "sentinel":
            state = "1111" if planned_cell.key.source_sample_id == "sentinel_one" else "0000"
            assert cell.counts == ((state, planned_cell.shots),)
    response = _worker(_count_request(planned, counts.to_wire()), tmp_path)
    assert response.returncode == 0, response.stdout.decode()
    payload = json.loads(response.stdout)
    assert payload["status"] == "validated_roundtrip_no_compute"
    assert payload["counts_sha256"] == _sha(counts.to_wire())
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(counts.to_wire()))


def test_seeded_counts_estimate_the_exact_expectations_under_any_bit_map(tmp_path: Path) -> None:
    """At 20,000 shots every estimate lies within five sigma of the exact value, for two bit maps."""
    maps = (((0, 1, 2, 3), (3, 2, 1, 0)), ((1, 2, 3, 0), (0, 2, 1, 3)))
    for logical, display in maps:
        spec = _spec(data_shots=20_000, logical_to_clbit=logical, display_order_clbits=display)
        gaps = _sigma_gaps(_planned(spec=spec), _SEED)
        assert len(gaps) == 4 * 3 * 7
        assert max(gaps) < 5.0, (logical, max(gaps))
    shuffled = _planned(spec=_spec(logical_to_clbit=maps[1][0], display_order_clbits=maps[1][1]))
    response = _worker(_count_request(shuffled, _counts(shuffled).to_wire()), tmp_path)
    assert response.returncode == 0, response.stdout.decode()


def test_sampling_refuses_when_qiskit_and_the_independent_statevector_disagree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cell whose Qiskit probabilities drift past the tolerance is never sampled."""
    planned = _planned()
    original = digital_simulation._qiskit_probabilities

    def drifted(operations: Any, measurement: Any) -> list[float]:
        """Return Qiskit's probabilities with 1e-9 moved between the first two outcomes."""
        values = original(operations, measurement)
        return [values[0] + 1e-9, values[1] - 1e-9, *values[2:]]

    monkeypatch.setattr(digital_simulation, "_qiskit_probabilities", drifted)
    with pytest.raises(
        ValueError, match="^Qiskit and the independent statevector disagree on a cell$"
    ):
        _counts(planned)
    tolerant = simulate_sampled(
        planned.experiment, planned.plan_set, seed_hex=_SEED, revisions=_REVISIONS, tolerance=1e-8
    )
    assert len(tolerant.cells) == len(planned.experiment.cells)


def test_sampler_draws_only_supported_outcomes_even_below_unit_mass() -> None:
    """Uniforms past a short cumulative sum fall back to the last outcome of nonzero probability."""
    measurement = _planned().experiment.measurements[-1]
    probabilities = (0.25, 0.0, 0.0, 0.25, 0.0, 0.25) + (0.0,) * 10
    drawn = dict(draw_counts(probabilities, measurement, seed_hex=_SEED, ordinal=0))
    assert sum(drawn.values()) == measurement.shots
    assert measurement.display_order_clbits == (3, 2, 1, 0)
    assert set(drawn) == {"0000", "0011", "0101"}
    assert drawn["0101"] > drawn["0000"] + 128 and drawn["0101"] > drawn["0011"] + 128


def test_simulated_cells_and_sets_refuse_malformed_records() -> None:
    """Cells carry sorted bitstring counts summing to their shots; sets name sampler, seed and order."""
    counts = _counts(_planned())
    cell = counts.cells[0]
    cell_cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"cell_ordinal": -1}, "^cell ordinal must be a bounded non-negative integer$"),
        ({"cell_ordinal": 4096}, "^cell ordinal must be a bounded non-negative integer$"),
        ({"planned_cell_digest": "E" * 64}, "^planned cell digest must be a lowercase SHA-256"),
        ({"measurement_request_digest": "x"}, "^measurement request digest must be a lowercase"),
        ({"shots": 0}, "^shots must be a positive bounded integer$"),
        ({"shots": 1_000_001}, "^shots must be a positive bounded integer$"),
        ({"counts": ()}, "^counts must be \\(key, count\\) pairs$"),
        ({"counts": (("0000",),)}, "^counts must be \\(key, count\\) pairs$"),
        ({"counts": (("000", 256),)}, "^count keys must be n4 or n8 bitstrings of one width$"),
        ({"counts": (("0002", 256),)}, "^count keys must be n4 or n8 bitstrings of one width$"),
        ({"counts": (("0000", 128), ("00000000", 128))}, "of one width$"),
        ({"counts": ((0, 256),)}, "^count keys must be n4 or n8 bitstrings of one width$"),
        ({"counts": (("1111", 128), ("0000", 128))}, "^count keys must be sorted and unique$"),
        ({"counts": (("0000", 128), ("0000", 128))}, "^count keys must be sorted and unique$"),
        ({"counts": (("0000", 0), ("1111", 256))}, "^every count must be a positive integer$"),
        ({"counts": (("0000", True), ("1111", 255))}, "^every count must be a positive integer$"),
        ({"counts": (("0000", 255),)}, "^counts must sum to the cell's shots$"),
    )
    for changes, reason in cell_cases:
        with pytest.raises(ValueError, match=reason):
            replace(cell, **changes)
    assert replace(cell, shots=1_000_000, counts=(("0000", 1_000_000),)).shots == 1_000_000
    stray: Any = (cell, "cell")
    set_cases: tuple[tuple[dict[str, Any], str], ...] = (
        (
            {"planned_experiment_digest": "E" * 64},
            "^planned experiment digest must be a lowercase",
        ),
        ({"sampler": "numpy_pcg64_multinomial"}, "^unknown simulated-count sampler$"),
        ({"seed_hex": "5EED" * 16}, "^seed must be 64 lowercase hex characters$"),
        ({"seed_hex": "5eed" * 15}, "^seed must be 64 lowercase hex characters$"),
        ({"seed_hex": 7}, "^seed must be 64 lowercase hex characters$"),
        ({"simulator": "aer_statevector"}, "^unknown ideal-digital simulator$"),
        ({"simulator_version": ""}, "simulator version must be bounded nonempty text"),
        ({"cells": ()}, "^cells must be a nonempty tuple of SimulatedCell records$"),
        ({"cells": stray}, "^cells must be a nonempty tuple of SimulatedCell records$"),
        ({"cells": counts.cells[1:]}, "^cells must follow the experiment's cell order$"),
        ({"cells": (counts.cells[1], counts.cells[0])}, "^cells must follow the experiment's"),
    )
    for changes, reason in set_cases:
        with pytest.raises(ValueError, match=reason):
            replace(counts, **changes)
    with pytest.raises(ValueError, match="^SimulatedCountSet requires its simulated artifact"):
        replace(counts, header=replace(counts.header, object_kind="exact_feature_set"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(counts, cells=counts.cells[:1])


def test_simulated_validation_refuses_drift_from_experiment_plan_set_or_seed() -> None:
    """Counts follow their experiment, its plan set, its origin and exactly the seeded sampler."""
    planned = _planned()
    counts = _counts(planned)
    cells = counts.cells
    other = _planned(arms=("xy",))
    no_sentinels = _rebuilt(planned.experiment, sentinels=())
    cases: tuple[tuple[SimulatedCountSet, PlannedExperiment, CircuitPlanSet, str], ...] = (
        (_counts(other), planned.experiment, planned.plan_set, "different planned experiment$"),
        (
            counts,
            planned.experiment,
            other.plan_set,
            "^the experiment names a different plan set$",
        ),
        (
            _rebuilt_counts(counts, origin="owner_dataset"),
            planned.experiment,
            planned.plan_set,
            "^count origin differs from the planned experiment$",
        ),
        (
            _rebuilt_counts(counts, cells=cells[:-1]),
            planned.experiment,
            planned.plan_set,
            "^every planned cell needs exactly one simulated cell$",
        ),
        (
            _rebuilt_counts(
                counts, cells=(replace(cells[0], shots=1, counts=(("0000", 1),)),) + cells[1:]
            ),
            planned.experiment,
            planned.plan_set,
            "^a simulated cell does not follow its planned cell$",
        ),
        (
            _rebuilt_counts(
                counts,
                cells=(replace(cells[0], planned_cell_digest=cells[1].planned_cell_digest),)
                + cells[1:],
            ),
            planned.experiment,
            planned.plan_set,
            "^a simulated cell does not follow its planned cell$",
        ),
        (
            _rebuilt_counts(counts, planned_experiment_digest=_sha(no_sentinels.to_wire())),
            no_sentinels,
            planned.plan_set,
            "^a planned cell names no plan or sentinel of the experiment$",
        ),
        (
            _rebuilt_counts(counts, cells=(_moved(cells[0]),) + cells[1:]),
            planned.experiment,
            planned.plan_set,
            "^simulated counts differ from the seeded sampler$",
        ),
        (
            _rebuilt_counts(counts, seed_hex="0" * 64),
            planned.experiment,
            planned.plan_set,
            "^simulated counts differ from the seeded sampler$",
        ),
    )
    for edited, experiment, plan_set, reason in cases:
        with pytest.raises(ValueError, match=reason):
            validate_simulated_count_set(edited, experiment, plan_set)
    not_counts: Any = "counts"
    with pytest.raises(ValueError, match="^simulated-count validation needs a SimulatedCountSet$"):
        validate_simulated_count_set(not_counts, planned.experiment, planned.plan_set)


def test_simulated_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires are refused by the decoder and the worker; consistent edits are replayed and refused."""
    planned = _planned()
    counts = _counts(planned)
    decoder_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "^SimulatedCountSet fields mismatch$"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.simulated_count_set.v2",
            "^unknown SimulatedCountSet",
        ),
        (("cells",), {"first": 1}, "^cells must be a list on wire$"),
        (("cells", 0, "unexpected"), True, "^SimulatedCell fields mismatch$"),
        (("cells", 0, "counts"), "0000", "^SimulatedCell fields mismatch$"),
        (("cells", 0, "counts", 0), "0000", "^counts must be \\[key, count\\] pairs on wire$"),
    )
    for path, value, reason in decoder_cases:
        with pytest.raises(ValueError, match=reason):
            decode_simulated_count_set(canonical_bytes(_forge(counts.to_wire(), path, value)))
    worker_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "SimulatedCountSet fields mismatch"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.simulated_count_set.v2",
            "unknown SimulatedCountSet",
        ),
        (("sampler",), "numpy_pcg64_multinomial", "unknown simulated-count sampler"),
        (("seed_hex",), "5EED" * 16, "seed must be 64 lowercase hex characters"),
        (("simulator",), "aer_statevector", "unknown ideal-digital simulator"),
        (("simulator_version",), "", "simulator version must be bounded normalized text"),
        (
            ("planned_experiment_digest",),
            "7" * 64,
            "the counts name a different planned experiment",
        ),
        (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
    )
    for path, forgery, refusal in worker_cases:
        response = _worker(
            _count_request(planned, _forge(counts.to_wire(), path, forgery)), tmp_path
        )
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    cells = counts.cells
    consistent: tuple[tuple[SimulatedCountSet, str], ...] = (
        (
            _rebuilt_counts(counts, cells=cells[:-1]),
            "every planned cell needs exactly one simulated cell",
        ),
        (
            _rebuilt_counts(
                counts, cells=(replace(cells[0], shots=1, counts=(("0000", 1),)),) + cells[1:]
            ),
            "a simulated cell does not follow its planned cell",
        ),
        (
            _rebuilt_counts(counts, cells=(_moved(cells[0]),) + cells[1:]),
            "differ from the seeded sampler",
        ),
        (_rebuilt_counts(counts, seed_hex="0" * 64), "differ from the seeded sampler"),
    )
    for edited, refusal in consistent:
        response = _worker(_count_request(planned, edited.to_wire()), tmp_path)
        assert response.returncode == 2, refusal
        assert refusal in json.loads(response.stdout)["reason"], refusal
    rehashed = _forge(counts.to_wire(), ("cells", 0, "unexpected"), True)
    content = {name: value for name, value in rehashed.items() if name != "header"}
    rehashed["header"]["content_digest"] = _sha(content)
    response = _worker(_count_request(planned, rehashed), tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "SimulatedCell fields mismatch"
    missing = _count_request(planned, counts.to_wire())
    del missing["counts"]
    response = _worker(missing, tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "simulated-count request fields mismatch"
