# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — domain evidence provenance for sensing and entropy results
"""Provenance that domain evidence must carry, and what it does not establish.

Every sensing or domain result is a :class:`DomainEvidence`: the estimand it
measures, the kind of data (``synthetic``, ``proxy``, ``empirical`` or
``clinical``), the source, the identity of the data split, the null baseline
it is compared with, and the use authority of external data. The kind is fixed
at construction and survives export unchanged; two results are compared only
when estimand, kind and split agree. Statistical tests and signatures on random
bytes establish test outcomes and byte integrity, never source entropy. An
external dataset is not acquired without a recorded use authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Final, Literal

import numpy as np

from ..entropy.nist_sp800_22 import frequency_test, runs_test
from .nv_magnetometry_20T import NVFieldCalibration

EvidenceKind = Literal["synthetic", "proxy", "empirical", "clinical"]
EVIDENCE_KINDS: Final[tuple[EvidenceKind, ...]] = ("synthetic", "proxy", "empirical", "clinical")
DOMAIN_EVIDENCE_SCHEMA: Final[str] = "domain_evidence.v1"
DOMAIN_EVIDENCE_CLAIM_BOUNDARY: Final[str] = (
    "the value of one estimand on the stated data kind and split, against the stated null "
    "baseline; synthetic, proxy, empirical and clinical evidence are not interchangeable, and "
    "no domain efficacy follows without independent domain validation"
)
ENTROPY_CLAIM_BOUNDARY: Final[str] = (
    "statistical tests describe these bytes and a signature proves their integrity; neither "
    "establishes the entropy of the source nor certifies cryptographic randomness"
)


def split_digest(identifiers: Sequence[str]) -> str:
    """Return the identity of a data split as the SHA-256 of its sorted sample identifiers.

    Parameters
    ----------
    identifiers
        Sample identifiers in the split; their order does not matter.

    Returns
    -------
    str
        Hex digest.

    Raises
    ------
    ValueError
        If the split is empty or an identifier repeats.

    """
    if not identifiers:
        raise ValueError("a data split needs at least one sample identifier")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample identifiers in a split must be unique")
    payload = "\n".join(sorted(identifiers)).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DomainEvidence:
    """One domain result with the provenance it must carry.

    Attributes
    ----------
    estimand
        What is measured, in words, with units.
    value
        The measured value.
    kind
        ``synthetic``, ``proxy``, ``empirical`` or ``clinical``.
    source
        Where the data came from.
    split
        :func:`split_digest` of the evaluation split.
    null_baseline
        The value a null model gives for the same estimand and split.
    use_authority
        Licence or approval under which external data was used; ``None`` for data generated here.

    """

    estimand: str
    value: float
    kind: EvidenceKind
    source: str
    split: str
    null_baseline: float
    use_authority: str | None = None
    claim_boundary: str = DOMAIN_EVIDENCE_CLAIM_BOUNDARY
    schema: str = DOMAIN_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        """Refuse missing provenance, unknown kinds and external data without authority."""
        if self.kind not in EVIDENCE_KINDS:
            raise ValueError(f"evidence kind must be one of {EVIDENCE_KINDS}")
        for name in ("estimand", "source", "split"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required domain provenance")
        if not math.isfinite(self.value) or not math.isfinite(self.null_baseline):
            raise ValueError("value and null_baseline must be finite")
        if self.kind in ("empirical", "clinical") and not self.use_authority:
            raise ValueError(f"{self.kind} evidence needs a recorded use authority")
        if self.schema != DOMAIN_EVIDENCE_SCHEMA:
            raise ValueError("unexpected domain evidence schema")
        if self.claim_boundary != DOMAIN_EVIDENCE_CLAIM_BOUNDARY:
            raise ValueError("unexpected domain evidence claim boundary")

    @property
    def improvement_over_null(self) -> float:
        """Return ``value - null_baseline``; its sign convention belongs to the estimand."""
        return self.value - self.null_baseline

    def to_json(self) -> str:
        """Export with every provenance field; the kind is written as given.

        Returns
        -------
        str
            JSON text.

        """
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> DomainEvidence:
        """Read an export back; the same validation applies.

        Parameters
        ----------
        text
            Output of :meth:`to_json`.

        Returns
        -------
        DomainEvidence
            The evidence, with its original kind.

        """
        return cls(**json.loads(text))


def compare_evidence(first: DomainEvidence, second: DomainEvidence) -> float:
    """Return ``second.value - first.value`` only for like-for-like evidence.

    Parameters
    ----------
    first, second
        Results to compare.

    Returns
    -------
    float
        The difference of values.

    Raises
    ------
    ValueError
        If estimand, kind or split differ, however close the numbers are.

    """
    for name in ("estimand", "kind", "split"):
        if getattr(first, name) != getattr(second, name):
            raise ValueError(
                f"evidence differs in {name} ({getattr(first, name)!r} vs "
                f"{getattr(second, name)!r}); the results are not comparable"
            )
    return second.value - first.value


def nv_field_evidence(
    calibration: NVFieldCalibration, *, sample_ids: Sequence[str], null_baseline_tesla: float
) -> DomainEvidence:
    """Wrap an NV field calibration from a simulated ODMR scan as synthetic evidence.

    Parameters
    ----------
    calibration
        Result of ``calibrate_field_from_odmr`` on a simulated measurement.
    sample_ids
        Identifiers of the simulated scans in the evaluation split.
    null_baseline_tesla
        Absolute field error of the null estimator (for example, always guessing zero field).

    Returns
    -------
    DomainEvidence
        ``kind="synthetic"``: the measurement was simulated here.

    Raises
    ------
    ValueError
        If the calibration carries no true field to score against.

    """
    if not math.isfinite(calibration.abs_error_tesla):
        raise ValueError("synthetic NV evidence needs the true field of the simulation")
    return DomainEvidence(
        estimand="absolute axial-field error of the ODMR upper-resonance estimate, tesla",
        value=float(calibration.abs_error_tesla),
        kind="synthetic",
        source="simulate_odmr_measurement (Lorentzian ODMR model with Gaussian readout noise)",
        split=split_digest(sample_ids),
        null_baseline=float(null_baseline_tesla),
    )


@dataclass(frozen=True)
class EntropyEvidence:
    """What tests and a signature say about random bytes, and what they do not.

    Attributes
    ----------
    frequency_passed, runs_passed
        NIST SP 800-22 frequency and runs outcomes on the bytes.
    signature_valid
        Whether the supplied signature check accepted the bytes.
    source_entropy_qualified
        Always ``False``: tests and signatures cannot establish source entropy.
    claim_boundary
        :data:`ENTROPY_CLAIM_BOUNDARY`.

    """

    frequency_passed: bool
    runs_passed: bool
    signature_valid: bool
    source_entropy_qualified: bool = False
    claim_boundary: str = ENTROPY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Refuse any record that claims qualified source entropy."""
        if self.source_entropy_qualified:
            raise ValueError("tests and signatures cannot qualify source entropy")


def assess_random_bytes(
    payload: bytes, *, verify_signature: Callable[[bytes], bool]
) -> EntropyEvidence:
    """Test random bytes and check their signature without qualifying the source.

    Parameters
    ----------
    payload
        The bytes to assess (at least 16).
    verify_signature
        Signature check supplied by the caller.

    Returns
    -------
    EntropyEvidence
        Test outcomes and signature validity; source entropy stays unqualified.

    Raises
    ------
    ValueError
        If fewer than 16 bytes are given.

    """
    if len(payload) < 16:
        raise ValueError("at least 16 bytes are needed for the bit tests")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).astype(np.int8)
    return EntropyEvidence(
        frequency_passed=frequency_test(bits).passed,
        runs_passed=runs_test(bits).passed,
        signature_valid=bool(verify_signature(payload)),
    )


@dataclass(frozen=True)
class ExternalDatasetRequest:
    """A request for an external dataset and the authority to use it.

    Attributes
    ----------
    dataset_id
        The dataset's identifier.
    kind
        The kind the data would be (``proxy``, ``empirical`` or ``clinical``).
    use_authority
        The licence or approval that permits the use, or ``None``.

    """

    dataset_id: str
    kind: EvidenceKind
    use_authority: str | None


def acquire_external_dataset(
    request: ExternalDatasetRequest, fetch: Callable[[str], bytes]
) -> bytes:
    """Fetch an external dataset only when a use authority is recorded.

    Parameters
    ----------
    request
        What is requested and under which authority.
    fetch
        The caller's retrieval function; it is not called without authority.

    Returns
    -------
    bytes
        The fetched dataset.

    Raises
    ------
    PermissionError
        If no use authority is recorded; nothing is fetched.
    ValueError
        If the request names synthetic data, which is generated rather than acquired.

    """
    if request.kind == "synthetic":
        raise ValueError("synthetic data is generated here, not acquired")
    if not request.use_authority:
        raise PermissionError(
            f"dataset {request.dataset_id!r} has no recorded use authority; it is not acquired"
        )
    return fetch(request.dataset_id)


__all__ = [
    "DOMAIN_EVIDENCE_CLAIM_BOUNDARY",
    "DOMAIN_EVIDENCE_SCHEMA",
    "ENTROPY_CLAIM_BOUNDARY",
    "EVIDENCE_KINDS",
    "DomainEvidence",
    "EntropyEvidence",
    "EvidenceKind",
    "ExternalDatasetRequest",
    "acquire_external_dataset",
    "assess_random_bytes",
    "compare_evidence",
    "nv_field_evidence",
    "split_digest",
]
