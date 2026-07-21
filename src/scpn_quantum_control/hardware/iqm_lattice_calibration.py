# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM square-lattice calibration adapter
"""IQM square-lattice calibration → Kuramoto layout-cost inputs.

Adapter for the square-lattice layout-transfer preregistration
(``docs/campaigns/iqm_layout_transfer_square_lattice_prereg_2026-07-21.md``):
extracts the coupling graph and calibration-derived fidelities from an IQM
backend (fake or live) into plain data, and selects chain regions for the
heavy-hex-benchmarked layout optimiser
(:mod:`scpn_quantum_control.hardware.kuramoto_layout_optimiser`).

The module never imports ``iqm``: the backend is duck-typed (``num_qubits``,
``coupling_map.get_edges()``, ``error_profile``), so hermetic tests run
without the optional extra and a live backend plugs in unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

__all__ = [
    "LatticeCalibration",
    "ChainRegion",
    "lattice_calibration_from_backend",
    "enumerate_chain_regions",
    "best_chain_region",
]

#: Deterministic cap on enumerated simple paths per (n, lattice) call. The
#: enumeration records whether the cap was hit — never a silent truncation.
MAX_ENUMERATED_PATHS = 20000


@dataclass(frozen=True)
class LatticeCalibration:
    """Plain-data snapshot of an IQM lattice and its calibration.

    ``edge_fidelity`` maps an undirected edge ``(lo, hi)`` (0-based physical
    indices) to a two-qubit gate-fidelity proxy ``1 - depolarising_error``;
    ``readout_error`` maps each physical qubit to its mean readout error.
    """

    num_qubits: int
    edges: tuple[tuple[int, int], ...]
    edge_fidelity: dict[tuple[int, int], float]
    readout_error: dict[int, float]

    def neighbours(self, qubit: int) -> tuple[int, ...]:
        """Physical neighbours of ``qubit`` in ascending order."""
        out = [b for a, b in self.edges if a == qubit]
        out += [a for a, b in self.edges if b == qubit]
        return tuple(sorted(out))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot (edge keys become ``"lo-hi"`` strings)."""
        return {
            "num_qubits": self.num_qubits,
            "edges": [list(edge) for edge in self.edges],
            "edge_fidelity": {f"{a}-{b}": value for (a, b), value in self.edge_fidelity.items()},
            "readout_error": {str(q): value for q, value in self.readout_error.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LatticeCalibration:
        """Rebuild a snapshot written by :meth:`to_dict` (fails closed)."""
        try:
            edges: list[tuple[int, int]] = []
            for raw_edge in payload["edges"]:
                a, b = raw_edge
                edges.append((int(a), int(b)))
            edge_fidelity: dict[tuple[int, int], float] = {}
            for key, value in payload["edge_fidelity"].items():
                lo, hi = key.split("-")
                edge_fidelity[(int(lo), int(hi))] = float(value)
            calibration = cls(
                num_qubits=int(payload["num_qubits"]),
                edges=tuple(edges),
                edge_fidelity=edge_fidelity,
                readout_error={
                    int(q): float(value) for q, value in payload["readout_error"].items()
                },
            )
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError(f"malformed lattice-calibration payload: {error}") from error
        if set(calibration.edge_fidelity) != set(calibration.edges):
            raise ValueError(
                "malformed lattice-calibration payload: edge_fidelity keys must match edges"
            )
        return calibration


@dataclass(frozen=True)
class ChainRegion:
    """A candidate physical chain and its calibration score."""

    physical_qubits: tuple[int, ...]
    mean_gate_fidelity: float
    mean_readout_error: float


def _qubit_index(name: Any) -> int:
    """Map an IQM qubit label (``'QB7'`` or an int) to a 0-based index."""
    if isinstance(name, int):
        return name
    text = str(name)
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"unparseable IQM qubit label: {name!r}")
    return int(digits) - 1


def _undirected(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def lattice_calibration_from_backend(backend: Any) -> LatticeCalibration:
    """Extract :class:`LatticeCalibration` from an IQM backend object.

    Works with ``IQMFakeGarnet``-style fake backends and any live backend
    exposing the same trio: ``num_qubits``, ``coupling_map.get_edges()`` and
    ``error_profile`` with ``two_qubit_gate_depolarizing_error_parameters``
    and ``readout_errors``.
    """
    num_qubits = int(backend.num_qubits)
    if num_qubits <= 0:
        raise ValueError("backend reports a non-positive qubit count")

    raw_edges = {_undirected(int(a), int(b)) for a, b in backend.coupling_map.get_edges()}
    edges = tuple(sorted(raw_edges))

    profile = backend.error_profile
    edge_fidelity: dict[tuple[int, int], float] = {}
    for per_edge in profile.two_qubit_gate_depolarizing_error_parameters.values():
        for pair, error in per_edge.items():
            key = _undirected(_qubit_index(pair[0]), _qubit_index(pair[1]))
            value = 1.0 - float(error)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"two-qubit fidelity out of range for edge {key}: {value}")
            edge_fidelity[key] = value

    missing = [edge for edge in edges if edge not in edge_fidelity]
    if missing:
        raise ValueError(f"calibration missing two-qubit errors for edges: {missing[:4]}")

    readout_error: dict[int, float] = {}
    for name, errors in profile.readout_errors.items():
        rates = [float(v) for v in errors.values()]
        if not rates:
            raise ValueError(f"empty readout-error entry for qubit {name!r}")
        readout_error[_qubit_index(name)] = sum(rates) / len(rates)

    return LatticeCalibration(
        num_qubits=num_qubits,
        edges=edges,
        edge_fidelity=edge_fidelity,
        readout_error=readout_error,
    )


def _score(calibration: LatticeCalibration, path: tuple[int, ...]) -> ChainRegion:
    pairs = [_undirected(path[i], path[i + 1]) for i in range(len(path) - 1)]
    gate = sum(calibration.edge_fidelity[p] for p in pairs) / len(pairs)
    readout = sum(calibration.readout_error.get(q, 0.0) for q in path) / len(path)
    return ChainRegion(physical_qubits=path, mean_gate_fidelity=gate, mean_readout_error=readout)


def enumerate_chain_regions(
    calibration: LatticeCalibration,
    n: int,
    *,
    max_paths: int = MAX_ENUMERATED_PATHS,
) -> tuple[tuple[ChainRegion, ...], bool]:
    """Enumerate simple physical chains of length ``n`` deterministically.

    Returns the scored regions (canonical orientation, sorted by descending
    mean gate fidelity then lexicographically) and a flag that is ``True``
    when the ``max_paths`` cap stopped the enumeration early — callers must
    surface that flag rather than silently treating the result as complete.
    """
    if n < 2:
        raise ValueError("a chain region needs at least two qubits")
    if n > calibration.num_qubits:
        raise ValueError("chain longer than the lattice qubit count")
    if max_paths < 1:
        raise ValueError("max_paths must be positive")

    found: set[tuple[int, ...]] = set()
    truncated = False

    def extend(path: list[int]) -> None:
        nonlocal truncated
        if truncated:
            return
        if len(path) == n:
            forward = tuple(path)
            canonical = forward if forward <= forward[::-1] else forward[::-1]
            found.add(canonical)
            if len(found) >= max_paths:
                truncated = True
            return
        for nxt in calibration.neighbours(path[-1]):
            if nxt not in path:
                path.append(nxt)
                extend(path)
                path.pop()

    for start in range(calibration.num_qubits):
        extend([start])

    regions = sorted(
        (_score(calibration, path) for path in found),
        key=lambda r: (-r.mean_gate_fidelity, r.physical_qubits),
    )
    return tuple(regions), truncated


def best_chain_region(
    calibration: LatticeCalibration,
    n: int,
    *,
    max_paths: int = MAX_ENUMERATED_PATHS,
) -> ChainRegion:
    """Best chain of length ``n`` by mean two-qubit gate fidelity.

    Deterministic: ties break on the lexicographically smallest canonical
    qubit tuple. Raises if the lattice holds no chain of the requested
    length within the enumeration cap.
    """
    regions, truncated = enumerate_chain_regions(calibration, n, max_paths=max_paths)
    if not regions:
        detail = " (enumeration cap hit)" if truncated else ""
        raise ValueError(f"no chain of length {n} found on the lattice{detail}")
    return regions[0]
