# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native modality data contracts
"""Exercise typed native I/O custody without claiming an optional SDK or device."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping, MutableMapping
from dataclasses import replace
from typing import Literal, cast

import pytest

from scpn_quantum_control.hardware.provider_modalities import (
    AnalogObservation,
    AnnealingObservation,
    AnnealingSample,
    ModalitySemantics,
    NativeAnnealingRecord,
    PhotonicObservation,
    PhotonicSample,
)


def test_photonic_occupation_custody_is_not_a_binary_measurement() -> None:
    """Two photons in one mode remain occupation 2 in the exported native record."""
    request = ModalitySemantics(
        program_sha256=hashlib.sha256(b"photonic-plan").hexdigest(),
        modality="photonic",
        native_axes=(0, 1),
    )
    observation = PhotonicObservation(
        request=request,
        shots=3,
        samples=(PhotonicSample((2, 0), 2, "|2,0>"), PhotonicSample((0, 2), 1, "|0,2>")),
    )
    payload = observation.to_payload()
    assert observation.counts == {"|2,0>": 2, "|0,2>": 1}
    assert payload["samples"] == [
        {"occupations": [2, 0], "occurrences": 2, "native_label": "|2,0>"},
        {"occupations": [0, 2], "occurrences": 1, "native_label": "|0,2>"},
    ]
    assert "measurement_map" not in payload
    with pytest.raises(ValueError, match="width"):
        replace(observation, samples=(PhotonicSample((2,), 3, "|2>"),))


def test_analog_samples_preserve_order_and_unknown_readout_convention() -> None:
    """Native per-shot correlations remain in input order with site identifiers."""
    request = ModalitySemantics(
        program_sha256=hashlib.sha256(b"analog-plan").hexdigest(),
        modality="analog",
        native_axes=("site-b", "site-a"),
    )
    observation = AnalogObservation(
        request=request,
        shots=3,
        raw_samples=((1, 0), "01", (1, 0)),
    )
    assert observation.counts == {"10": 2, "01": 1}
    assert observation.raw_samples == ((1, 0), "01", (1, 0))
    assert observation.to_payload()["readout_convention"] == "provider_native_unknown"
    with pytest.raises(ValueError, match="binary"):
        replace(observation, raw_samples=(cast(tuple[int, ...], (0.5, 1)),))


def test_annealing_spin_energy_and_native_column_order_survive_projection() -> None:
    """SPIN -1 survives even when the legacy compatibility counts contain zero."""
    request = ModalitySemantics(
        program_sha256=hashlib.sha256(b"spin-plan").hexdigest(),
        modality="annealing",
        native_axes=("a", "b"),
        vartype="SPIN",
    )
    observation = AnnealingObservation(
        request=request,
        shots=3,
        returned_axes=("b", "a"),
        native_vartype="SPIN",
        samples=(AnnealingSample((1, -1), 2, -0.25), AnnealingSample((-1, 1), 1, None)),
    )
    assert observation.counts == {"01": 2, "10": 1}
    payload = observation.to_payload()
    assert payload["returned_axes"] == ["b", "a"]
    assert payload["samples"] == [
        {"values": [1, -1], "occurrences": 2, "energy": -0.25},
        {"values": [-1, 1], "occurrences": 1, "energy": None},
    ]
    assert payload["legacy_count_projection"] == {"01": 2, "10": 1}
    with pytest.raises(ValueError, match="domain"):
        replace(observation, samples=(AnnealingSample((0, 1), 3, None),))


def test_modal_request_rejects_source_and_axis_aliasing() -> None:
    """Source digests and ordered unique axes are required before submission."""
    request = ModalitySemantics(
        program_sha256=hashlib.sha256(b"plan").hexdigest(),
        modality="analog",
        native_axes=(2, 0),
    )
    request.require_source("plan", 2)
    with pytest.raises(ValueError, match="source"):
        request.require_source("different-plan", 2)
    with pytest.raises(ValueError, match="axes"):
        replace(request, native_axes=(2, 2))


def test_native_energy_overflow_has_a_stable_contract_refusal() -> None:
    """An energy outside native finite numeric representation refuses cleanly."""
    with pytest.raises(ValueError, match="finite"):
        AnnealingSample((1, -1), 1, 10**400)


def test_returned_annealing_axes_do_not_alias_boolean_labels() -> None:
    """Boolean equality to integer zero cannot replace a native variable label."""
    request = ModalitySemantics(
        program_sha256=hashlib.sha256(b"binary-plan").hexdigest(),
        modality="annealing",
        native_axes=(0, 1),
        vartype="BINARY",
    )
    with pytest.raises(ValueError, match="axes"):
        AnnealingObservation(
            request,
            returned_axes=(False, 1),
            shots=1,
            samples=(AnnealingSample((0, 1), 1, None),),
        )


def _native_request(modality: Literal["photonic", "analog", "annealing"]) -> ModalitySemantics:
    """Declare two ordered native axes and the annealing domain when applicable."""
    return ModalitySemantics(
        hashlib.sha256(b"native-plan").hexdigest(),
        modality,
        ("b", "a"),
        vartype="SPIN" if modality == "annealing" else None,
    )


@pytest.mark.parametrize(
    "changes,diagnostic",
    [
        ({"schema": "provider_semantics.v2"}, "schema"),
        ({"program_sha256": "F" * 64}, "digest"),
        ({"program_sha256": 3}, "digest"),
        ({"modality": "unknown"}, "modality"),
        ({"native_axes": ()}, "axes"),
        ({"native_axes": ("", 1)}, "axes"),
        ({"native_axes": (False, 1)}, "axes"),
        ({"native_axes": (-1, 1)}, "axes"),
        ({"native_axes": (0.5, 1)}, "axes"),
        ({"native_axes": ([0], 1)}, "axes"),
        ({"native_axes": ("a", "a")}, "axes"),
        ({"vartype": "SPIN"}, "vartype"),
        ({"modality": "annealing"}, "vartype"),
        ({"requested_target": ""}, "target"),
        ({"requested_target": 3}, "target"),
        ({"requested_target": " padded"}, "target"),
        ({"requested_target": "native\x00target"}, "target"),
        ({"requested_target": "native\x7ftarget"}, "target"),
    ],
)
def test_modal_request_refuses_invalid_native_domain_or_identity(
    changes: dict[str, object],
    diagnostic: str,
) -> None:
    """Invalid declarations cannot become a native source-bound request."""
    request = _native_request("analog")
    with pytest.raises(ValueError, match=diagnostic):
        replace(
            request,
            schema=cast(str, changes.get("schema", request.schema)),
            program_sha256=cast(str, changes.get("program_sha256", request.program_sha256)),
            modality=cast(
                Literal["photonic", "analog", "annealing"],
                changes.get("modality", request.modality),
            ),
            native_axes=cast(
                tuple[str | int, ...], changes.get("native_axes", request.native_axes)
            ),
            vartype=cast(
                Literal["SPIN", "BINARY"] | None, changes.get("vartype", request.vartype)
            ),
            requested_target=cast(
                str | None, changes.get("requested_target", request.requested_target)
            ),
        )


def test_modal_payload_and_count_channels_detach_mutable_input() -> None:
    """External mutations leave stored site order, native counts and exports intact."""
    axes = ["b", "a"]
    request = replace(
        _native_request("analog"),
        native_axes=cast(tuple[str | int, ...], axes),
        requested_target="declared",
    )
    axes.reverse()
    raw = {"10": 2, "01": 1}
    observation = AnalogObservation(request, 3, raw_counts=raw)
    raw.clear()
    payload = observation.to_payload()
    assert request.native_axes == ("b", "a")
    assert request.to_payload() == {
        "schema": "provider_semantics.v1",
        "program_sha256": hashlib.sha256(b"native-plan").hexdigest(),
        "modality": "analog",
        "native_axes": ["b", "a"],
        "vartype": None,
        "requested_target": "declared",
    }
    assert observation.raw_counts == {"10": 2, "01": 1}
    assert payload["raw_counts"] == {"10": 2, "01": 1}
    assert payload["raw_samples"] is None
    with pytest.raises(ValueError, match="dimension"):
        request.require_source("native-plan", 1)
    with pytest.raises(TypeError):
        cast(MutableMapping[str, int], observation.counts)["10"] = 9


@pytest.mark.parametrize(
    "changes,diagnostic",
    [
        ({"occupations": ()}, "nonempty"),
        ({"native_label": ""}, "nonempty"),
        ({"native_label": 1}, "nonempty"),
        ({"occupations": (-1, 0)}, "integer"),
        ({"occupations": (1.5, 0)}, "integer"),
        ({"occupations": (True, 0)}, "integer"),
        ({"occurrences": -1}, "integer"),
        ({"occurrences": 1.5}, "integer"),
    ],
)
def test_photonic_sample_refuses_lossy_occupation_or_count(
    changes: dict[str, object],
    diagnostic: str,
) -> None:
    """Fock occupations remain exact integers rather than binary or rounded values."""
    sample = PhotonicSample((2, 0), 1, "|2,0>")
    with pytest.raises(ValueError, match=diagnostic):
        replace(
            sample,
            occupations=cast(tuple[int, ...], changes.get("occupations", sample.occupations)),
            occurrences=cast(int, changes.get("occurrences", sample.occurrences)),
            native_label=cast(str, changes.get("native_label", sample.native_label)),
        )


@pytest.mark.parametrize(
    "changes,diagnostic",
    [
        ({"shots": 0}, "integer"),
        ({"request": _native_request("analog")}, "photonic request"),
        ({"samples": ()}, "total"),
        ({"samples": (PhotonicSample((2, 0), 2, "|2,0>"),)}, "total"),
        (
            {"samples": (PhotonicSample((2, 0), 1, "same"), PhotonicSample((0, 2), 1, "same"))},
            "distinct",
        ),
    ],
)
def test_photonic_observation_requires_distinct_native_outcomes_and_totals(
    changes: dict[str, object],
    diagnostic: str,
) -> None:
    """Incomplete or ambiguously labelled native evidence never qualifies."""
    observation = PhotonicObservation(
        _native_request("photonic"), (PhotonicSample((2, 0), 1, "|2,0>"),), 1
    )
    with pytest.raises(ValueError, match=diagnostic):
        replace(
            observation,
            shots=cast(int, changes.get("shots", observation.shots)),
            request=cast(ModalitySemantics, changes.get("request", observation.request)),
            samples=cast(tuple[PhotonicSample, ...], changes.get("samples", observation.samples)),
        )


@pytest.mark.parametrize(
    "changes,diagnostic",
    [
        ({"shots": False}, "integer"),
        ({"request": _native_request("photonic")}, "one native"),
        ({"raw_samples": ("10",)}, "one native"),
        ({"raw_counts": None}, "one native"),
        ({"raw_counts": {}}, "total"),
        ({"raw_counts": {"10": 2}}, "total"),
        ({"raw_counts": {"10": -1}}, "integer"),
        ({"raw_counts": {"10": 1.0}}, "integer"),
        ({"raw_counts": {"1x": 1}}, "binary"),
        ({"raw_counts": {"1": 1}}, "width"),
        ({"raw_counts": {"": 1}}, "width"),
    ],
)
def test_analog_observation_requires_one_exact_native_count_channel(
    changes: dict[str, object],
    diagnostic: str,
) -> None:
    """Channels, native binary values and conserved totals remain explicit."""
    observation = AnalogObservation(_native_request("analog"), 1, raw_counts={"10": 1})
    with pytest.raises(ValueError, match=diagnostic):
        replace(
            observation,
            shots=cast(int, changes.get("shots", observation.shots)),
            request=cast(ModalitySemantics, changes.get("request", observation.request)),
            raw_counts=cast(
                Mapping[str, int] | None, changes.get("raw_counts", observation.raw_counts)
            ),
            raw_samples=cast(
                tuple[str | tuple[int, ...], ...] | None,
                changes.get("raw_samples", observation.raw_samples),
            ),
        )


@pytest.mark.parametrize(
    "samples,diagnostic",
    [
        (("1x",), "binary"),
        (("1",), "width"),
        (((2, 0),), "binary"),
        (((True, 0),), "binary"),
        (((1,),), "width"),
        ((), "total"),
    ],
)
def test_analog_per_shot_data_refuses_invalid_native_readouts(
    samples: tuple[str | tuple[int, ...], ...],
    diagnostic: str,
) -> None:
    """Per-shot arrays retain original values without padding or bit coercion."""
    with pytest.raises(ValueError, match=diagnostic):
        AnalogObservation(_native_request("analog"), 1, raw_samples=samples)


@pytest.mark.parametrize(
    "changes,diagnostic",
    [
        ({"values": ()}, "integers"),
        ({"values": (True, 0)}, "integers"),
        ({"values": (1.0, -1)}, "integers"),
        ({"occurrences": -1}, "integer"),
        ({"energy": True}, "finite"),
        ({"energy": "0.0"}, "finite"),
        ({"energy": float("nan")}, "finite"),
        ({"energy": float("inf")}, "finite"),
    ],
)
def test_annealing_samples_refuse_inexact_values_or_invalid_energy(
    changes: dict[str, object],
    diagnostic: str,
) -> None:
    """SPIN values, occurrences and optional native energy keep their domains."""
    sample = AnnealingSample((1, -1), 1, 0)
    with pytest.raises(ValueError, match=diagnostic):
        replace(
            sample,
            values=cast(tuple[int, ...], changes.get("values", sample.values)),
            occurrences=cast(int, changes.get("occurrences", sample.occurrences)),
            energy=cast(float | None, changes.get("energy", sample.energy)),
        )


@pytest.mark.parametrize(
    "changes,diagnostic",
    [
        ({"dtype_description": ""}, "layout"),
        ({"dtype_description": 3}, "layout"),
        ({"shape": (1, 1)}, "layout"),
        ({"shape": (0,)}, "integer"),
        ({"itemsize": False}, "integer"),
        ({"data": b"short"}, "byte length"),
        ({"data": bytearray(8)}, "byte length"),
    ],
)
def test_original_annealing_record_requires_exact_native_storage(
    changes: dict[str, object],
    diagnostic: str,
) -> None:
    """Original dtype text and immutable bytes must describe the same record."""
    record = NativeAnnealingRecord("native dtype text", (1,), 8, b"\x00" * 8)
    with pytest.raises(ValueError, match=diagnostic):
        replace(
            record,
            dtype_description=cast(
                str, changes.get("dtype_description", record.dtype_description)
            ),
            shape=cast(tuple[int], changes.get("shape", record.shape)),
            itemsize=cast(int, changes.get("itemsize", record.itemsize)),
            data=cast(bytes, changes.get("data", record.data)),
        )


def test_annealing_record_export_and_binary_rows_preserve_native_evidence() -> None:
    """Native repeated rows, energies and extra record bytes survive the projection."""
    request = replace(_native_request("annealing"), vartype="BINARY")
    record = NativeAnnealingRecord("uninterpreted native dtype", (2,), 8, b"\x01\x00" * 8)
    observation = AnnealingObservation(
        request,
        ("a", "b"),
        (AnnealingSample((1, 0), 2, 0), AnnealingSample((1, 0), 1, None)),
        3,
        native_vartype="BINARY",
        raw_record=record,
    )
    assert observation.counts == {"01": 3}
    payload = record.to_payload()
    assert payload["dtype_description"] == "uninterpreted native dtype"
    assert payload["shape"] == [2]
    assert payload["itemsize"] == 8
    assert payload["data_sha256"] == hashlib.sha256(record.data).hexdigest()
    assert base64.b64decode(str(payload["data_base64"])) == record.data
    assert observation.to_payload()["raw_record"] == payload
    assert AnalogObservation(_native_request("analog"), 1, raw_samples=("10",)).to_payload()[
        "raw_samples"
    ] == ["10"]


@pytest.mark.parametrize(
    "changes,diagnostic",
    [
        ({"shots": 0}, "integer"),
        ({"request": _native_request("analog")}, "axes"),
        ({"returned_axes": ("a",)}, "axes"),
        ({"returned_axes": ("a", "a")}, "axes"),
        ({"returned_axes": ("a", "foreign")}, "axes"),
        ({"returned_axes": ([0], "b")}, "axes"),
        ({"native_vartype": "BINARY"}, "domain"),
        ({"samples": (AnnealingSample((1,), 1, None),)}, "width or domain"),
        ({"samples": ()}, "total"),
        ({"shots": 2}, "total"),
        ({"raw_record": NativeAnnealingRecord("native dtype", (2,), 1, b"\x00\x00")}, "rows"),
    ],
)
def test_annealing_observation_refuses_axis_domain_or_record_drift(
    changes: dict[str, object],
    diagnostic: str,
) -> None:
    """Native sample evidence binds original variables, domain, counts and rows."""
    observation = AnnealingObservation(
        _native_request("annealing"),
        ("b", "a"),
        (AnnealingSample((1, -1), 1, None),),
        1,
    )
    with pytest.raises(ValueError, match=diagnostic):
        replace(
            observation,
            shots=cast(int, changes.get("shots", observation.shots)),
            request=cast(ModalitySemantics, changes.get("request", observation.request)),
            returned_axes=cast(
                tuple[str | int, ...], changes.get("returned_axes", observation.returned_axes)
            ),
            native_vartype=cast(
                Literal["SPIN", "BINARY"] | None,
                changes.get("native_vartype", observation.native_vartype),
            ),
            samples=cast(tuple[AnnealingSample, ...], changes.get("samples", observation.samples)),
            raw_record=cast(
                NativeAnnealingRecord | None, changes.get("raw_record", observation.raw_record)
            ),
        )
