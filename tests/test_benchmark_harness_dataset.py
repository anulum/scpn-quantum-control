# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — benchmark_harness.dataset tests
"""Tests for `scpn_quantum_control.benchmark_harness.dataset`.

Cover:

* Real load of the published Phase 1 dataset (342 circuits, single
  backend, aggregate counts).
* Integrity check — published digests verify.
* Integrity check — tampered file raises :class:`DatasetIntegrityError`.
* Missing directory raises :class:`FileNotFoundError`.
* Missing sub-phase file raises :class:`FileNotFoundError`.
* Top-level JSON that is not an object raises :class:`ValueError`.
* Missing required top-level key raises :class:`ValueError` with the key path.
* ``n_circuits`` disagreeing with list length raises :class:`ValueError`.
* Missing circuit-level ``meta`` or ``counts`` key raises :class:`ValueError`.
* Invalid ``sector`` raises :class:`ValueError`.
* Malformed ``counts`` entry raises :class:`ValueError`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scpn_quantum_control.benchmark_harness.dataset import (
    DEFAULT_DATA_DIR,
    PUBLISHED_SHA256,
    SUBPHASE_FILES,
    DatasetIntegrityError,
    load_benchmark_dataset,
)

REAL_DATA_DIR = DEFAULT_DATA_DIR


def _has_real_data() -> bool:
    return REAL_DATA_DIR.is_dir() and all((REAL_DATA_DIR / f).is_file() for f in SUBPHASE_FILES)


needs_real_data = pytest.mark.skipif(
    not _has_real_data(),
    reason="Benchmark dataset not present — skipping real-data tests",
)


@needs_real_data
class TestRealDataset:
    def test_load_full(self) -> None:
        ds = load_benchmark_dataset()
        assert len(ds.subphases) == 4
        assert ds.n_circuits_total == 342
        assert ds.backends == frozenset({"ibm_kingston"})
        assert {run.experiment for run in ds.subphases} == {
            "phase1_dla_parity_mini_bench",
            "phase1_5_reinforce",
            "phase2_exhaust_cycle",
            "phase2_5_final_burn",
        }

    def test_load_order_is_publication_order(self) -> None:
        ds = load_benchmark_dataset()
        assert tuple(run.experiment for run in ds.subphases) == (
            "phase1_dla_parity_mini_bench",
            "phase1_5_reinforce",
            "phase2_exhaust_cycle",
            "phase2_5_final_burn",
        )

    def test_meta_fields_populated(self) -> None:
        ds = load_benchmark_dataset()
        c = ds.circuits[0]
        assert c.meta.n_qubits == 4
        assert c.meta.t_step == 0.3
        assert c.meta.shots == 2048
        assert c.meta.sector in ("even", "odd")

    def test_integrity_verifies(self) -> None:
        ds = load_benchmark_dataset(verify_integrity=True)
        assert ds.n_circuits_total == 342

    def test_integrity_digests_are_present(self) -> None:
        assert set(PUBLISHED_SHA256.keys()) == set(SUBPHASE_FILES)
        assert all(len(v) == 64 for v in PUBLISHED_SHA256.values())


@needs_real_data
class TestIntegrityTamper:
    def test_tampered_file_raises(self, tmp_path: Path) -> None:
        for name in SUBPHASE_FILES:
            shutil.copy(REAL_DATA_DIR / name, tmp_path / name)
        tampered = tmp_path / SUBPHASE_FILES[0]
        with tampered.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        raw["backend"] = "tampered_backend"
        with tampered.open("w", encoding="utf-8") as fh:
            json.dump(raw, fh)
        with pytest.raises(DatasetIntegrityError, match="SHA-256 mismatch"):
            load_benchmark_dataset(data_dir=tmp_path, verify_integrity=True)

    def test_tampered_file_without_verify_loads(self, tmp_path: Path) -> None:
        for name in SUBPHASE_FILES:
            shutil.copy(REAL_DATA_DIR / name, tmp_path / name)
        tampered = tmp_path / SUBPHASE_FILES[0]
        with tampered.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        raw["backend"] = "tampered_backend"
        with tampered.open("w", encoding="utf-8") as fh:
            json.dump(raw, fh)
        ds = load_benchmark_dataset(data_dir=tmp_path, verify_integrity=False)
        assert "tampered_backend" in ds.backends


def _minimal_subphase(
    *,
    experiment: str = "phase1_dla_parity_mini_bench",
    n_circuits: int = 1,
    circuits: list | None = None,
    **overrides: object,
) -> dict[str, object]:
    if circuits is None:
        circuits = [
            {
                "meta": {
                    "experiment": "A_dla_parity_n4",
                    "n_qubits": 4,
                    "depth": 2,
                    "sector": "even",
                    "initial": "0011",
                    "rep": 0,
                    "shots": 2048,
                    "t_step": 0.3,
                },
                "counts": {"1100": 1500, "1010": 500},
            }
        ]
    doc: dict[str, object] = {
        "experiment": experiment,
        "timestamp_utc": "2026-04-10T000000Z",
        "backend": "ibm_kingston",
        "job_ids": ["job0"],
        "wall_time_s": 1.0,
        "n_circuits": n_circuits,
        "t_step": 0.3,
        "circuits": circuits,
        "aggregated": {"note": "synthetic"},
    }
    doc.update(overrides)
    return doc


def _write_fake_dataset(root: Path, subphase_docs: dict[str, dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, doc in subphase_docs.items():
        with (root / name).open("w", encoding="utf-8") as fh:
            json.dump(doc, fh)


class TestFileDiscovery:
    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="data directory not found"):
            load_benchmark_dataset(data_dir=missing)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        docs = {name: _minimal_subphase() for name in SUBPHASE_FILES[:3]}
        _write_fake_dataset(tmp_path, docs)
        with pytest.raises(FileNotFoundError, match="Missing benchmark run"):
            load_benchmark_dataset(data_dir=tmp_path)


class TestSchemaValidation:
    def _build_all_valid(self, tmp_path: Path) -> None:
        docs = {name: _minimal_subphase() for name in SUBPHASE_FILES}
        _write_fake_dataset(tmp_path, docs)

    def test_minimal_valid_dataset_loads(self, tmp_path: Path) -> None:
        self._build_all_valid(tmp_path)
        ds = load_benchmark_dataset(data_dir=tmp_path)
        assert ds.n_circuits_total == 4

    def test_extra_top_level_keys_land_in_extra(self, tmp_path: Path) -> None:
        self._build_all_valid(tmp_path)
        ds = load_benchmark_dataset(data_dir=tmp_path)
        assert all("aggregated" in sp.extra for sp in ds.subphases)

    def test_top_level_not_object(self, tmp_path: Path) -> None:
        tmp_path.mkdir(exist_ok=True)
        path = tmp_path / SUBPHASE_FILES[0]
        with path.open("w", encoding="utf-8") as fh:
            json.dump(["not", "an", "object"], fh)
        for name in SUBPHASE_FILES[1:]:
            with (tmp_path / name).open("w", encoding="utf-8") as fh:
                json.dump(_minimal_subphase(), fh)
        with pytest.raises(ValueError, match="expected top-level JSON object"):
            load_benchmark_dataset(data_dir=tmp_path)

    def test_missing_top_level_key(self, tmp_path: Path) -> None:
        docs = {name: _minimal_subphase() for name in SUBPHASE_FILES}
        del docs[SUBPHASE_FILES[0]]["backend"]
        _write_fake_dataset(tmp_path, docs)
        with pytest.raises(ValueError, match="missing required keys.*'backend'"):
            load_benchmark_dataset(data_dir=tmp_path)

    def test_n_circuits_mismatch(self, tmp_path: Path) -> None:
        docs = {name: _minimal_subphase() for name in SUBPHASE_FILES}
        docs[SUBPHASE_FILES[0]]["n_circuits"] = 9
        _write_fake_dataset(tmp_path, docs)
        with pytest.raises(ValueError, match="n_circuits=9"):
            load_benchmark_dataset(data_dir=tmp_path)

    def test_circuit_missing_meta(self, tmp_path: Path) -> None:
        bad_circuit = [{"counts": {"0000": 10}}]
        docs = {name: _minimal_subphase() for name in SUBPHASE_FILES}
        docs[SUBPHASE_FILES[0]] = _minimal_subphase(circuits=bad_circuit)
        _write_fake_dataset(tmp_path, docs)
        with pytest.raises(ValueError, match=r"circuits\[0\].*missing required keys.*'meta'"):
            load_benchmark_dataset(data_dir=tmp_path)

    def test_circuit_meta_not_object(self, tmp_path: Path) -> None:
        bad_circuit = [{"meta": "not-an-object", "counts": {"0000": 10}}]
        docs = {name: _minimal_subphase() for name in SUBPHASE_FILES}
        docs[SUBPHASE_FILES[0]] = _minimal_subphase(circuits=bad_circuit)
        _write_fake_dataset(tmp_path, docs)
        with pytest.raises(ValueError, match=r"circuits\[0\].meta: expected object"):
            load_benchmark_dataset(data_dir=tmp_path)

    def test_invalid_sector(self, tmp_path: Path) -> None:
        bad_circuit = [
            {
                "meta": {
                    "experiment": "A_dla_parity_n4",
                    "n_qubits": 4,
                    "depth": 2,
                    "sector": "skew",
                    "initial": "0011",
                    "rep": 0,
                    "shots": 2048,
                    "t_step": 0.3,
                },
                "counts": {"1100": 10},
            }
        ]
        docs = {name: _minimal_subphase() for name in SUBPHASE_FILES}
        docs[SUBPHASE_FILES[0]] = _minimal_subphase(circuits=bad_circuit)
        _write_fake_dataset(tmp_path, docs)
        with pytest.raises(ValueError, match="expected 'even'.*'baseline'"):
            load_benchmark_dataset(data_dir=tmp_path)

    def test_counts_not_object(self, tmp_path: Path) -> None:
        bad_circuit = [
            {
                "meta": {
                    "experiment": "A_dla_parity_n4",
                    "n_qubits": 4,
                    "depth": 2,
                    "sector": "even",
                    "initial": "0011",
                    "rep": 0,
                    "shots": 2048,
                    "t_step": 0.3,
                },
                "counts": [["1100", 10]],
            }
        ]
        docs = {name: _minimal_subphase() for name in SUBPHASE_FILES}
        docs[SUBPHASE_FILES[0]] = _minimal_subphase(circuits=bad_circuit)
        _write_fake_dataset(tmp_path, docs)
        with pytest.raises(ValueError, match=r"circuits\[0\].counts: expected object"):
            load_benchmark_dataset(data_dir=tmp_path)

    def test_counts_bad_value_type(self, tmp_path: Path) -> None:
        bad_circuit = [
            {
                "meta": {
                    "experiment": "A_dla_parity_n4",
                    "n_qubits": 4,
                    "depth": 2,
                    "sector": "even",
                    "initial": "0011",
                    "rep": 0,
                    "shots": 2048,
                    "t_step": 0.3,
                },
                "counts": {"1100": "many"},
            }
        ]
        docs = {name: _minimal_subphase() for name in SUBPHASE_FILES}
        docs[SUBPHASE_FILES[0]] = _minimal_subphase(circuits=bad_circuit)
        _write_fake_dataset(tmp_path, docs)
        with pytest.raises(ValueError, match="expected str.*int mapping"):
            load_benchmark_dataset(data_dir=tmp_path)

    def test_circuits_not_list(self, tmp_path: Path) -> None:
        docs = {name: _minimal_subphase() for name in SUBPHASE_FILES}
        docs[SUBPHASE_FILES[0]]["circuits"] = {"not": "a-list"}
        _write_fake_dataset(tmp_path, docs)
        with pytest.raises(ValueError, match="circuits: expected list"):
            load_benchmark_dataset(data_dir=tmp_path)

    def test_job_ids_not_list_of_str(self, tmp_path: Path) -> None:
        docs = {name: _minimal_subphase() for name in SUBPHASE_FILES}
        docs[SUBPHASE_FILES[0]]["job_ids"] = [1, 2, 3]
        _write_fake_dataset(tmp_path, docs)
        with pytest.raises(ValueError, match=r"job_ids: expected list\[str\]"):
            load_benchmark_dataset(data_dir=tmp_path)
