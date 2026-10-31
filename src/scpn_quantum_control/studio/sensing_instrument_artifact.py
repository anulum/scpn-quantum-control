# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-owned sensing scores and public entropy metadata
"""Project bounded sensing and entropy evidence without exporting private samples.

The score owner supplies Gaussian-model information gain and posterior variance.
The entropy owner supplies statistical outcomes; a fixture signature is checked
by the existing platform verifier. Neither can certify source randomness. No
device is contacted, and an absent device record remains explicitly unavailable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scpn_studio_platform.seal import Ed25519Signer

from ..active_sensing_product import (
    ACTIVE_SENSING_CLAIM_BOUNDARY,
    demo_information_gain_candidates,
    score_expected_information_gain,
)
from ..sensing.domain_evidence import ENTROPY_CLAIM_BOUNDARY, EntropyEvidence, assess_random_bytes
from .committed_data import run_committed_data_cli

SENSING_INSTRUMENT_SCHEMA: Final[str] = "scpn_qc_studio_sensing_instrument_v1"
DEFAULT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data/studio/sensing_instrument_20261031.json"
)


def project_entropy_metadata(
    metadata: Mapping[str, object], assessment: EntropyEvidence | None
) -> dict[str, object]:
    """Return only public provenance and diagnostic declarations.

    Parameters
    ----------
    metadata
        Source declarations. Seeds, keys, samples and unrecognised fields are
        excluded even when supplied here. Missing provenance stays ``None``.
    assessment
        Source owner's diagnostic result, or ``None`` without samples.

    Returns
    -------
    dict[str, object]
        Allowlisted public metadata; entropy certification is always false.

    Raises
    ------
    ValueError
        If the source kind is unknown or a public declaration has the wrong type.

    """
    kind = metadata.get("source_kind", "unknown")
    if kind not in ("synthetic", "device", "unknown"):
        raise ValueError("source_kind must be synthetic, device or unknown")
    result: dict[str, object] = {"source_kind": kind}
    for name in ("case_id", "source", "extractor", "assumptions", "payload_sha256"):
        value = metadata.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be nonblank text or None")
        result[name] = value
    result["diagnostics"] = (
        None
        if assessment is None
        else {
            "frequency_passed": assessment.frequency_passed,
            "runs_passed": assessment.runs_passed,
            "signature_valid": assessment.signature_valid,
        }
    )
    result["source_entropy_qualified"] = False
    result["claim_boundary"] = ENTROPY_CLAIM_BOUNDARY
    return result


def build_sensing_instrument_artifact() -> dict[str, object]:
    """Build repeatable synthetic scores, signed fixture diagnostics and an absent device row.

    Returns
    -------
    dict[str, object]
        Source-owned public metadata. The fixed signing key is a public test
        vector, never a production identity or evidence of physical randomness.

    """
    signer = Ed25519Signer(
        "public-synthetic-fixture",
        Ed25519PrivateKey.from_private_bytes(
            hashlib.sha256(b"public-sensing-test-vector").digest()
        ),
    )
    verifier = signer.verifier()
    payload = np.random.default_rng(42).bytes(256)
    signature = signer.sign(payload)
    assessment = assess_random_bytes(
        payload, verify_signature=lambda message: verifier.verify(message, signature)
    )
    synthetic = project_entropy_metadata(
        {
            "case_id": "synthetic-signed-fixture",
            "source_kind": "synthetic",
            "source": "NumPy deterministic fixture; public test-vector signing identity",
            "extractor": "none",
            "assumptions": "Pseudorandom fixture only; no physical entropy assumption or qualification",
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        },
        assessment,
    )
    unavailable = project_entropy_metadata(
        {"case_id": "device-unavailable", "source_kind": "device"}, None
    )
    return {
        "schema": SENSING_INSTRUMENT_SCHEMA,
        "artifact_id": "studio-sensing-instrument-20261031",
        "generated_by": "python -m scpn_quantum_control.studio.sensing_instrument_artifact --write",
        "design_boundary": ACTIVE_SENSING_CLAIM_BOUNDARY,
        "design_model": "Scalar conjugate Gaussian model; posterior variance is model uncertainty, not empirical calibration",
        "candidates": [
            {
                **score_expected_information_gain(candidate, shots=32).to_dict(),
                "prior_variance": candidate.prior_variance,
                "sensitivity": candidate.sensitivity,
                "noise_variance": candidate.noise_variance,
            }
            for candidate in demo_information_gain_candidates()
        ],
        "entropy": [synthetic, unavailable],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed public metadata.

    Parameters
    ----------
    argv
        CLI arguments, or the process arguments when omitted.

    Returns
    -------
    int
        The shared committed-data writer's exit status.

    """
    return run_committed_data_cli(
        argv,
        build=build_sensing_instrument_artifact,
        default_path=DEFAULT_PATH,
        label="sensing instrument data",
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
