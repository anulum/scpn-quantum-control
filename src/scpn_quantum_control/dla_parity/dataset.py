# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA parity — dataset loader
"""DLA-parity dataset loader.

Loads the four published JSON run files from
``data/phase1_dla_parity/`` (data-directory name is part of the
published URLs and preserved for citation stability) into the
strongly-typed
:class:`~scpn_quantum_control.dla_parity.schema.DlaParityDataset`.

Single public entry point: :func:`load_dla_parity_dataset`. Optional
SHA-256 integrity check gates the loader against
:data:`PUBLISHED_SHA256`, which lists the digests of the dataset as
released alongside the short paper. Mismatch raises
:class:`DatasetIntegrityError`; shape violations raise
:class:`ValueError` with the offending key path.

No statistical computation — that lives in :mod:`.reproduce`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import (
    DlaParityCircuit,
    DlaParityCircuitMeta,
    DlaParityDataset,
    DlaParityRun,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "phase1_dla_parity"

PUBLISHED_SHA256: dict[str, str] = {
    "phase1_bench_2026-04-10T183728Z.json": (
        "700273064367b0fbe1bd245c448eee0611d62f5c7e94a0bcf0b0c12e63840599"
    ),
    "phase1_5_reinforce_2026-04-10T184909Z.json": (
        "7d4c7455203adf0637f14b9a5e913d34c36367dd9b879784d3826d24da793ade"
    ),
    "phase2_exhaust_2026-04-10T185634Z.json": (
        "77cea12482377478af69964ecef346cc363b30baddc83521fe421d84c00d299e"
    ),
    "phase2_5_final_burn_2026-04-10T190136Z.json": (
        "03cd28c060aee97302bf697ce0fdee910a533715bee5124778e0a36b42e159eb"
    ),
}

# Canonical load order = publication order. The loader returns runs
# in this order so the downstream reproducer does not need to sort.
RUN_FILES: tuple[str, ...] = (
    "phase1_bench_2026-04-10T183728Z.json",
    "phase1_5_reinforce_2026-04-10T184909Z.json",
    "phase2_exhaust_2026-04-10T185634Z.json",
    "phase2_5_final_burn_2026-04-10T190136Z.json",
)

_REQUIRED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        "experiment",
        "timestamp_utc",
        "backend",
        "job_ids",
        "wall_time_s",
        "n_circuits",
        "t_step",
        "circuits",
    }
)
_REQUIRED_META: frozenset[str] = frozenset(
    {
        "experiment",
        "n_qubits",
        "depth",
        "sector",
        "initial",
        "rep",
        "shots",
        "t_step",
    }
)


class DatasetIntegrityError(RuntimeError):
    """Raised when a dataset file does not match its published digest."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_keys(obj: dict[str, Any], required: frozenset[str], where: str) -> None:
    missing = required - obj.keys()
    if missing:
        raise ValueError(
            f"{where}: missing required keys {sorted(missing)}; "
            f"present keys = {sorted(obj.keys())}",
        )


def _parse_circuit(raw: dict[str, Any], idx: int) -> DlaParityCircuit:
    _require_keys(raw, frozenset({"meta", "counts"}), f"circuits[{idx}]")
    meta_raw = raw["meta"]
    if not isinstance(meta_raw, dict):
        raise ValueError(f"circuits[{idx}].meta: expected object, got {type(meta_raw).__name__}")
    _require_keys(meta_raw, _REQUIRED_META, f"circuits[{idx}].meta")
    sector = meta_raw["sector"]
    if sector not in ("even", "odd", "baseline"):
        raise ValueError(
            f"circuits[{idx}].meta.sector: expected 'even'|'odd'|'baseline', got {sector!r}",
        )
    counts_raw = raw["counts"]
    if not isinstance(counts_raw, dict):
        raise ValueError(
            f"circuits[{idx}].counts: expected object, got {type(counts_raw).__name__}",
        )
    counts: dict[str, int] = {}
    for k, v in counts_raw.items():
        if not isinstance(k, str) or not isinstance(v, int):
            raise ValueError(
                f"circuits[{idx}].counts[{k!r}]: expected str→int mapping, "
                f"got {type(k).__name__}→{type(v).__name__}",
            )
        counts[k] = v
    meta = DlaParityCircuitMeta(
        experiment=str(meta_raw["experiment"]),
        n_qubits=int(meta_raw["n_qubits"]),
        depth=int(meta_raw["depth"]),
        sector=sector,  # type: ignore[arg-type]
        initial=str(meta_raw["initial"]),
        rep=int(meta_raw["rep"]),
        shots=int(meta_raw["shots"]),
        t_step=float(meta_raw["t_step"]),
    )
    return DlaParityCircuit(meta=meta, counts=counts)


def _parse_run(raw: dict[str, Any], filename: str) -> DlaParityRun:
    _require_keys(raw, _REQUIRED_TOP_LEVEL, filename)
    circuits_raw = raw["circuits"]
    if not isinstance(circuits_raw, list):
        raise ValueError(
            f"{filename}.circuits: expected list, got {type(circuits_raw).__name__}",
        )
    circuits = tuple(_parse_circuit(c, i) for i, c in enumerate(circuits_raw))
    declared = int(raw["n_circuits"])
    if declared != len(circuits):
        raise ValueError(
            f"{filename}: n_circuits={declared} but circuits list has {len(circuits)} entries",
        )
    extra = {k: v for k, v in raw.items() if k not in _REQUIRED_TOP_LEVEL}
    job_ids_raw = raw["job_ids"]
    if not isinstance(job_ids_raw, list) or not all(isinstance(j, str) for j in job_ids_raw):
        raise ValueError(f"{filename}.job_ids: expected list[str]")
    return DlaParityRun(
        experiment=str(raw["experiment"]),
        timestamp_utc=str(raw["timestamp_utc"]),
        backend=str(raw["backend"]),
        job_ids=tuple(job_ids_raw),
        wall_time_s=float(raw["wall_time_s"]),
        n_circuits=declared,
        t_step=float(raw["t_step"]),
        circuits=circuits,
        extra=extra,
    )


def load_dla_parity_dataset(
    *,
    data_dir: Path | str | None = None,
    verify_integrity: bool = False,
) -> DlaParityDataset:
    """Load the published DLA-parity dataset.

    Parameters
    ----------
    data_dir:
        Directory holding the four run JSONs. Defaults to
        ``data/phase1_dla_parity`` at the repository root (data-directory
        name is part of the published URLs and kept for citation
        stability).
    verify_integrity:
        When True, compute SHA-256 on each JSON and raise
        :class:`DatasetIntegrityError` on a mismatch against
        :data:`PUBLISHED_SHA256`. Default False — integrity gating is
        opt-in because the published digests may lag behind a dataset
        refresh until a release is cut.

    Returns
    -------
    :class:`~scpn_quantum_control.dla_parity.schema.DlaParityDataset`
        The four runs in canonical publication order.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` or any expected run file is missing.
    DatasetIntegrityError
        If ``verify_integrity`` is True and any digest mismatches.
    ValueError
        If a JSON file is malformed or violates the schema.
    """
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"DLA-parity data directory not found: {root}")

    runs: list[DlaParityRun] = []
    for filename in RUN_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing DLA-parity run file: {path}")
        if verify_integrity:
            actual = _sha256(path)
            expected = PUBLISHED_SHA256[filename]
            if actual != expected:
                raise DatasetIntegrityError(
                    f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}",
                )
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{filename}: expected top-level JSON object")
        runs.append(_parse_run(raw, filename))

    return DlaParityDataset(runs=tuple(runs))
