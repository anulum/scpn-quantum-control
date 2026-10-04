# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — typed native non-gate provider observations
"""Preserve occupations, site readouts and spin samples in their native domains.

These immutable I/O records do not attest an installed SDK, physical execution,
calibration or a readout convention. Original native plans retain operator units.
Legacy count projections remain explicitly separate from native sample values.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Literal


def _integer(value: int, name: str, *, positive: bool = False) -> None:
    """Reject inexact, negative or optionally zero native dimensions/counts."""
    if type(value) is not int or value < int(positive):
        raise ValueError(
            f"{name} must be an exact {'positive' if positive else 'nonnegative'} integer"
        )


@dataclass(frozen=True)
class ModalitySemantics:
    """Original native plan and ordered modes, sites or annealing variables.

    Parameters
    ----------
    program_sha256
        SHA-256 of the exact UTF-8 encoded HAL plan.
    modality
        Native photonic, analog or annealing domain; no gate mapping is inferred.
    native_axes
        Ordered unique mode indices, site identifiers or variable labels.
    vartype
        Required SPIN/BINARY domain for annealing; None for other modalities.
    requested_target
        Exact optional caller pin. Absence does not attest a physical target.
    schema
        Exact supported companion version.

    Raises
    ------
    ValueError
        If version, digest, domain, axes or target identity is invalid.

    """

    program_sha256: str
    modality: Literal["photonic", "analog", "annealing"]
    native_axes: tuple[str | int, ...]
    vartype: Literal["SPIN", "BINARY"] | None = None
    requested_target: str | None = None
    schema: str = "provider_semantics.v1"

    def __post_init__(self) -> None:
        """Detach the native order and refuse ambiguous domain declarations."""
        if self.schema != "provider_semantics.v1":
            raise ValueError("unsupported provider semantics schema")
        if (
            not isinstance(self.program_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.program_sha256) is None
        ):
            raise ValueError("program_sha256 must be a SHA-256 digest")
        if self.modality not in {"photonic", "analog", "annealing"}:
            raise ValueError("unsupported native modality")
        axes = tuple(self.native_axes)
        if (
            not axes
            or any(
                (type(axis) is not int and not isinstance(axis, str))
                or (isinstance(axis, str) and not axis)
                or (type(axis) is int and axis < 0)
                for axis in axes
            )
            or len(set(axes)) != len(axes)
        ):
            raise ValueError(
                "native axes must be nonempty, unique string or nonnegative integer identifiers"
            )
        if (self.modality == "annealing" and self.vartype not in {"SPIN", "BINARY"}) or (
            self.modality != "annealing" and self.vartype is not None
        ):
            raise ValueError("vartype must match the native modality")
        target = self.requested_target
        if target is not None and (
            not isinstance(target, str)
            or not target
            or target.strip() != target
            or any(ord(char) < 32 or ord(char) == 127 for char in target)
        ):
            raise ValueError("requested_target must be an exact nonempty target name")
        object.__setattr__(self, "native_axes", axes)

    @property
    def n_qubits(self) -> int:
        """Return the HAL width as a mode/site/variable count, without qubit inference."""
        return len(self.native_axes)

    def require_source(self, program: str, n_qubits: int) -> None:
        """Require the exact native plan and HAL dimension before submission.

        Parameters
        ----------
        program
            Unchanged native encoded plan.
        n_qubits
            HAL dimension, representing modes/sites/variables for this domain.

        Raises
        ------
        ValueError
            If plan digest or dimension differs from the declaration.

        """
        if hashlib.sha256(
            program.encode("utf-8")
        ).hexdigest() != self.program_sha256 or n_qubits != len(self.native_axes):
            raise ValueError("native modality source digest or dimension differs")

    def to_payload(self) -> dict[str, object]:
        """Export detached native order without synthesizing gate measurements.

        Returns
        -------
        dict[str, object]
            Versioned native plan digest, domain, axes and optional target pin.

        """
        return {
            "schema": self.schema,
            "program_sha256": self.program_sha256,
            "modality": self.modality,
            "native_axes": list(self.native_axes),
            "vartype": self.vartype,
            "requested_target": self.requested_target,
        }


@dataclass(frozen=True)
class PhotonicSample:
    """One native Fock occupation outcome with its unchanged display label.

    Parameters
    ----------
    occupations
        Ordered nonnegative integer photon occupations per native mode.
    occurrences
        Nonnegative observed occurrence count.
    native_label
        Unchanged provider state label, without binary interpretation.

    Raises
    ------
    ValueError
        If occupation, count or label is invalid.

    """

    occupations: tuple[int, ...]
    occurrences: int
    native_label: str

    def __post_init__(self) -> None:
        """Freeze native occupation values without reducing them to binary bits."""
        occupations = tuple(self.occupations)
        if not occupations or not isinstance(self.native_label, str) or not self.native_label:
            raise ValueError("native photonic occupation and label must be nonempty")
        for occupation in occupations:
            _integer(occupation, "photonic occupation")
        _integer(self.occurrences, "photonic occurrences")
        object.__setattr__(self, "occupations", occupations)


@dataclass(frozen=True)
class PhotonicObservation:
    """Native occupation outcomes preserving mode order and exact occurrence totals.

    Parameters
    ----------
    request
        Stored source-bound photonic plan declaration.
    samples
        Native output order, occupations, display labels and occurrence counts.
    shots
        Positive observed count, equal to the occurrence sum.

    Raises
    ------
    ValueError
        If domain, width, labels or occurrence totals disagree.

    """

    request: ModalitySemantics
    samples: tuple[PhotonicSample, ...]
    shots: int
    counts: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        """Freeze exact occupations and maintain the legacy native-label count view."""
        _integer(self.shots, "photonic shots", positive=True)
        if self.request.modality != "photonic":
            raise ValueError("photonic observation requires a photonic request")
        samples = tuple(self.samples)
        counts: dict[str, int] = {}
        for sample in samples:
            if len(sample.occupations) != self.request.n_qubits:
                raise ValueError("native photonic sample width differs from mode count")
            if sample.native_label in counts:
                raise ValueError("native photonic labels must identify distinct outcomes")
            counts[sample.native_label] = sample.occurrences
        if not counts or sum(counts.values()) != self.shots:
            raise ValueError("native photonic occurrence total differs from shots")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "counts", MappingProxyType(counts))

    def to_payload(self) -> dict[str, object]:
        """Export original occupation values, labels and mode order.

        Returns
        -------
        dict[str, object]
            Typed native outcomes; occupations above one remain unchanged.

        """
        return {
            "schema": self.request.schema,
            "modality": "photonic",
            "request": self.request.to_payload(),
            "shots": self.shots,
            "samples": [
                {
                    "occupations": list(sample.occupations),
                    "occurrences": sample.occurrences,
                    "native_label": sample.native_label,
                }
                for sample in self.samples
            ],
        }


@dataclass(frozen=True)
class AnalogObservation:
    """Unchanged native analog site readouts in count or per-shot sample form.

    Parameters
    ----------
    request
        Stored analog plan with native site order.
    shots
        Positive exact sample count.
    raw_counts
        Optional original binary count channel; mutually exclusive with raw_samples.
    raw_samples
        Optional original ordered strings or integer bit sequences, including repeats.

    Raises
    ------
    ValueError
        If channel, native width, binary values or totals disagree.

    Notes
    -----
    Readout polarity and atom-loss interpretation remain explicitly unknown.

    """

    request: ModalitySemantics
    shots: int
    raw_counts: Mapping[str, int] | None = None
    raw_samples: tuple[str | tuple[int, ...], ...] | None = None
    counts: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        """Detach the original readout channel while preserving per-shot order."""
        _integer(self.shots, "analog shots", positive=True)
        if self.request.modality != "analog" or (self.raw_counts is None) == (
            self.raw_samples is None
        ):
            raise ValueError("analog observation requires one native readout channel")
        counts: dict[str, int] = {}
        if self.raw_counts is not None:
            for key, count in self.raw_counts.items():
                _analog_bits(key, self.request.n_qubits)
                _integer(count, "analog count")
                counts[key] = count
            object.__setattr__(self, "raw_counts", MappingProxyType(dict(counts)))
        else:
            assert self.raw_samples is not None
            samples = tuple(
                value if isinstance(value, str) else tuple(value) for value in self.raw_samples
            )
            for sample in samples:
                key = _analog_bits(sample, self.request.n_qubits)
                counts[key] = counts.get(key, 0) + 1
            object.__setattr__(self, "raw_samples", samples)
        if not counts or sum(counts.values()) != self.shots:
            raise ValueError("native analog sample total differs from shots")
        object.__setattr__(self, "counts", MappingProxyType(counts))

    def to_payload(self) -> dict[str, object]:
        """Export original native counts or ordered samples with unknown polarity.

        Returns
        -------
        dict[str, object]
            Readout channel preserving correlations and explicit site order.

        """
        return {
            "schema": self.request.schema,
            "modality": "analog",
            "request": self.request.to_payload(),
            "shots": self.shots,
            "readout_convention": "provider_native_unknown",
            "raw_counts": dict(self.raw_counts) if self.raw_counts is not None else None,
            "raw_samples": [
                value if isinstance(value, str) else list(value) for value in self.raw_samples
            ]
            if self.raw_samples is not None
            else None,
        }


def _analog_bits(value: str | tuple[int, ...], width: int) -> str:
    """Validate exact native binary readouts without truncation or polarity inference."""
    if isinstance(value, str):
        if any(bit not in "01" for bit in value):
            raise ValueError("analog native readout must contain binary values")
        key = value
    else:
        if any(type(bit) is not int or bit not in (0, 1) for bit in value):
            raise ValueError("analog native readout must contain exact binary integers")
        key = "".join(str(bit) for bit in value)
    if not key or len(key) != width:
        raise ValueError("analog native readout width differs from site order")
    return key


@dataclass(frozen=True)
class AnnealingSample:
    """One native ordered annealing sample with occurrences and optional energy.

    Parameters
    ----------
    values
        Native integer variable values, retaining SPIN -1 where returned.
    occurrences
        Nonnegative native occurrence count.
    energy
        Native energy in original model units, or None when not returned.

    Raises
    ------
    ValueError
        If values, occurrences or finite energy are invalid.

    """

    values: tuple[int, ...]
    occurrences: int
    energy: float | None

    def __post_init__(self) -> None:
        """Detach original integer values and retain unknown energy explicitly."""
        values = tuple(self.values)
        if not values or any(type(value) is not int for value in values):
            raise ValueError("native annealing values must be nonempty exact integers")
        _integer(self.occurrences, "annealing occurrences")
        if self.energy is not None:
            if isinstance(self.energy, bool) or not isinstance(self.energy, int | float):
                raise ValueError("native annealing energy must be finite numeric data or unknown")
            try:
                finite = isfinite(self.energy)
            except OverflowError as exc:
                raise ValueError(
                    "native annealing energy must be finite numeric data or unknown"
                ) from exc
            if not finite:
                raise ValueError("native annealing energy must be finite numeric data or unknown")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class NativeAnnealingRecord:
    """Exact original structured native record bytes with their storage layout.

    Parameters
    ----------
    dtype_description
        Unchanged native dtype descriptor text, retained without evaluating it.
    shape
        Original one-dimensional record-row shape.
    itemsize
        Native bytes per structured record.
    data
        Exact original C-order bytes, including energies and extra native fields.

    Raises
    ------
    ValueError
        If layout is empty, inexact or inconsistent with the original byte length.

    """

    dtype_description: str
    shape: tuple[int]
    itemsize: int
    data: bytes

    def __post_init__(self) -> None:
        """Bind original record shape to immutable bytes without dtype reinterpretation."""
        shape = tuple(self.shape)
        if (
            not isinstance(self.dtype_description, str)
            or not self.dtype_description
            or len(shape) != 1
        ):
            raise ValueError(
                "native annealing record requires its original one-dimensional layout"
            )
        _integer(shape[0], "native record rows", positive=True)
        _integer(self.itemsize, "native record itemsize", positive=True)
        if not isinstance(self.data, bytes) or len(self.data) != shape[0] * self.itemsize:
            raise ValueError("native annealing record original byte length differs")
        object.__setattr__(self, "shape", shape)

    def to_payload(self) -> dict[str, object]:
        """Export original record bytes with exact storage metadata and digest.

        Returns
        -------
        dict[str, object]
            Detached base64 bytes and their original dtype/shape/byte length.

        """
        return {
            "dtype_description": self.dtype_description,
            "shape": list(self.shape),
            "itemsize": self.itemsize,
            "data_base64": base64.b64encode(self.data).decode("ascii"),
            "data_sha256": hashlib.sha256(self.data).hexdigest(),
        }


@dataclass(frozen=True)
class AnnealingObservation:
    """Native SPIN/BINARY samples separate from the legacy bit count projection.

    Parameters
    ----------
    request
        Original BQM variable order and declared native domain.
    returned_axes
        Actual returned column order; may differ from request order.
    samples
        Original row order, native values, occurrences and optional energies.
    shots
        Positive exact occurrence sum.
    native_vartype
        Returned native domain when available, otherwise explicitly unknown.
    raw_record
        Optional original structured record bytes, preserving extra native fields.

    Raises
    ------
    ValueError
        If axes, domain, native values or occurrence totals disagree.

    """

    request: ModalitySemantics
    returned_axes: tuple[str | int, ...]
    samples: tuple[AnnealingSample, ...]
    shots: int
    native_vartype: Literal["SPIN", "BINARY"] | None = None
    raw_record: NativeAnnealingRecord | None = None
    counts: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        """Preserve native domain and order before forming a compatibility projection."""
        _integer(self.shots, "annealing shots", positive=True)
        axes, samples = tuple(self.returned_axes), tuple(self.samples)
        if (
            any(type(axis) is not int and not isinstance(axis, str) for axis in axes)
            or self.request.modality != "annealing"
            or len(axes) != len(self.request.native_axes)
            or set(axes) != set(self.request.native_axes)
        ):
            raise ValueError("native annealing axes differ from the original variable order")
        if self.native_vartype is not None and self.native_vartype != self.request.vartype:
            raise ValueError("native annealing domain differs from the original model")
        order = [axes.index(axis) for axis in self.request.native_axes]
        domain = {-1, 1} if self.request.vartype == "SPIN" else {0, 1}
        counts: dict[str, int] = {}
        for sample in samples:
            if len(sample.values) != len(axes) or any(
                value not in domain for value in sample.values
            ):
                raise ValueError("native annealing sample width or domain differs")
            key = "".join("1" if sample.values[index] == 1 else "0" for index in order)
            counts[key] = counts.get(key, 0) + sample.occurrences
        if not counts or sum(counts.values()) != self.shots:
            raise ValueError("native annealing occurrence total differs from shots")
        if self.raw_record is not None and self.raw_record.shape[0] != len(samples):
            raise ValueError("native annealing record rows differ from native samples")
        object.__setattr__(self, "returned_axes", axes)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "counts", MappingProxyType(counts))

    def to_payload(self) -> dict[str, object]:
        """Export native values and energies without replacing them by binary counts.

        Returns
        -------
        dict[str, object]
            Original native columns/rows and a separately named legacy projection.

        """
        return {
            "schema": self.request.schema,
            "modality": "annealing",
            "request": self.request.to_payload(),
            "shots": self.shots,
            "returned_axes": list(self.returned_axes),
            "native_vartype": self.native_vartype,
            "raw_record": self.raw_record.to_payload() if self.raw_record is not None else None,
            "samples": [
                {
                    "values": list(sample.values),
                    "occurrences": sample.occurrences,
                    "energy": sample.energy,
                }
                for sample in self.samples
            ],
            "legacy_count_projection": dict(self.counts),
        }


__all__ = [
    "ModalitySemantics",
    "PhotonicSample",
    "PhotonicObservation",
    "AnalogObservation",
    "AnnealingSample",
    "AnnealingObservation",
    "NativeAnnealingRecord",
]
