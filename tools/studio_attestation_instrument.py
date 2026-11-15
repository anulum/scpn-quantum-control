# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — committed data of the Studio attestation instrument
"""Build, write and check the committed data of the Studio attestation instrument.

The Studio verifies attested result-pack envelopes in the browser. This tool
builds the envelopes it shows, over the committed IBM Fez V2 aggregate summary,
and records the verdict of the Python owner for each, so the browser verdicts
have an independent reference. It also records the exact text of the owner's
canonical serialisation for edge values and the Python whitespace set.

Run ``python tools/studio_attestation_instrument.py --write`` after the owner
or the source summary changes, and ``--check`` to compare the committed data
with a fresh build.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

from scpn_quantum_control.attested_result_pack import (
    ATTESTED_RESULT_PACK_CLAIM_BOUNDARY,
    ATTESTED_RESULT_PACK_SCHEMA,
    AttestedEnvelope,
    build_unsigned_envelope,
    canonical_content_digest,
    default_claim_axes,
    verify_attested_envelope,
)
from scpn_quantum_control.studio.committed_data import run_committed_data_cli

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
ATTESTATION_INSTRUMENT_SCHEMA: Final[str] = "scpn_qc_studio_attestation_instrument_v1"
DEFAULT_PATH: Final[Path] = ROOT / "data/studio/attestation_instrument_20261115.json"
SOURCE_PATH: Final[str] = "results/ibm_hardware_v2_2026-03-29/full_results.json"
SOURCE_SHA256: Final[str] = "197510d92490ac3e6879f580ef5e75a4c7d6a1db6f326dfb3e48e2c5b227165c"
CLAIM_ID: Final[str] = "claim.ibm_fez.v2_equal_depth.2026-03-29"
CLAIM_AXES: Final[dict[str, object]] = default_claim_axes(
    claim_unit="ibm_fez V2 equal-depth fidelity and survival summary, 2026-03-29",
    honesty_axes=(
        "content_digest_only",
        "aggregate_summary_not_raw_counts",
        "no_provider_provenance",
        "unsigned",
        "unpromoted_pending_raw_count_review",
    ),
    backend_class="hardware_reported_summary",
)
BOUNDARY: Final[str] = (
    "VERIFIED means that an unsigned envelope is complete and that its digest matches its "
    "claim axes and content; it is not a signature, provider provenance or scientific "
    "validation, and these values are an aggregate summary pending raw-count review"
)


def load_source(root: Path = ROOT) -> dict[str, object]:
    """Read the pinned QPU summary and refuse any other bytes.

    Parameters
    ----------
    root
        Repository root holding the committed summary.

    Returns
    -------
    dict[str, object]
        The decoded summary.

    Raises
    ------
    RuntimeError
        If the file's SHA-256 differs from the pinned digest.

    """
    raw = (root / SOURCE_PATH).read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != SOURCE_SHA256:
        raise RuntimeError(f"{SOURCE_PATH} changed: sha256 {observed}, pinned {SOURCE_SHA256}")
    decoded: dict[str, object] = json.loads(raw)
    return decoded


def _changed_mean(content: dict[str, object]) -> dict[str, object]:
    """Return a copy whose ``A_even`` mean differs from the sealed value."""
    changed = json.loads(json.dumps(content))
    changed["A_even"]["mean"] = 0.9285
    return dict(changed)


def _case(
    case_id: str,
    title: str,
    change: str,
    envelope: AttestedEnvelope,
    *,
    require_signature: bool = False,
) -> dict[str, object]:
    """Record one envelope with the verdict of the Python owner."""
    verdict = verify_attested_envelope(envelope, require_signature=require_signature)
    return {
        "id": case_id,
        "title": title,
        "change": change,
        "require_signature": require_signature,
        "envelope": envelope.to_dict(),
        "python_verdict": verdict.to_dict(),
    }


_TAMPERED_CASE_TEXT: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "changed-value",
        "Value changed after sealing",
        "content.A_even.mean 0.9185 -> 0.9285, digest kept",
    ),
    (
        "rebound-axes",
        "Claim axes replaced",
        "claim_axes.backend_class -> hardware_verified, digest kept",
    ),
    ("stripped-axes", "Honesty axes removed", "claim_axes.honesty_axes emptied"),
    ("missing-digest", "Digest removed", "content_digest emptied"),
    (
        "unverified-signature",
        "Signature without a keyring",
        "signature field set to an opaque token",
    ),
)
"""Identifier, label and change text of each single-field tampering case, in display order."""


def build_cases(content: dict[str, object]) -> list[dict[str, object]]:
    """Seal the summary once and derive every refusal from that envelope.

    Parameters
    ----------
    content
        The pinned QPU summary.

    Returns
    -------
    list[dict[str, object]]
        Seven cases in display order, each with the owner's verdict.

    """
    sealed = build_unsigned_envelope(claim_id=CLAIM_ID, claim_axes=CLAIM_AXES, content=content)
    return [
        _case("intact", "Sealed summary", "none", sealed),
        *_tampered_cases(sealed, content),
        _case(
            "signature-required",
            "Signature required",
            "verified with require_signature",
            sealed,
            require_signature=True,
        ),
    ]


def _tampered_cases(
    sealed: AttestedEnvelope, content: dict[str, object]
) -> list[dict[str, object]]:
    """Derive the five refusals that alter one field of the sealed envelope.

    Parameters
    ----------
    sealed
        The envelope sealed over the pinned summary.
    content
        The pinned QPU summary the envelope was sealed over.

    Returns
    -------
    list[dict[str, object]]
        The changed-value, rebound-axes, stripped-axes, missing-digest and
        unverified-signature cases, in display order.

    """
    envelopes = (
        replace(sealed, content=_changed_mean(content)),
        replace(sealed, claim_axes={**CLAIM_AXES, "backend_class": "hardware_verified"}),
        replace(sealed, claim_axes={**CLAIM_AXES, "honesty_axes": []}),
        replace(sealed, content_digest=""),
        replace(sealed, signature="opaque-token-without-keyring"),
    )
    return [
        _case(*text, envelope)
        for text, envelope in zip(_TAMPERED_CASE_TEXT, envelopes, strict=True)
    ]


def _random_floats(count: int) -> list[float]:
    """Return seeded doubles: any finite bit pattern, then short decimals.

    A biased exponent below 2047 is always finite, so no draw is discarded;
    the rounded decimals cover the short spellings typical of measured values.
    """
    generator = random.Random(20261115)
    values: list[float] = []
    for _ in range(count):
        sign = generator.getrandbits(1) << 63
        bits = sign | (generator.randrange(2047) << 52) | generator.getrandbits(52)
        values.append(struct.unpack("<d", bits.to_bytes(8, "little"))[0])
    for _ in range(count):
        values.append(round(generator.uniform(-1000.0, 1000.0), generator.randrange(1, 9)))
    return values


def _oracle_contents() -> list[tuple[str, dict[str, object]]]:
    """Return labelled contents that exercise every branch of the serialisation."""
    floats = [
        0.0,
        -0.0,
        1.0,
        -7.25,
        0.1,
        0.0001,
        1e-05,
        1.5e-07,
        1e15,
        1e16,
        2.5e22,
        123456789.125,
    ]
    floats += [0.30000000000000004, 3.8812132998900355e-12, 5e-324, 1.7976931348623157e308]
    integers = [0, -1, 2**53, 2**53 + 1, 10**30, -(10**25)]
    strings = ["", 'quote " backslash \\', "tab\tnewline\nreturn\rbell\x07", "\x00\x1f\x7f"]
    strings += ["é€", "  ", "\U0001d538", "﻿"]
    return [
        ("edge floats", {"value": floats}),
        ("seeded random floats", {"value": _random_floats(32)}),
        ("integers", {"value": integers}),
        ("strings", {"value": strings}),
        ("key order", {"￿": 1, "\U0001d538": 2, "a": 3, "B": 4, "é": 5, "": 6}),
        ("nesting and literals", {"list": [[], {}, [1, 2.0, None, True, False]]}),
    ]


def build_oracle() -> list[dict[str, object]]:
    """Record the owner's canonical text and digest for each oracle content.

    Returns
    -------
    list[dict[str, object]]
        Label, content, the exact canonical text and the owner's digest.

    """
    rows: list[dict[str, object]] = []
    for label, content in _oracle_contents():
        bound = {"claim_axes": {}, "content": content}
        text = json.dumps(
            bound, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        rows.append(
            {
                "label": label,
                "content": content,
                "python_text": text,
                "digest": canonical_content_digest(content, {}),
            }
        )
    return rows


def build_attestation_instrument_artifact(root: Path = ROOT) -> dict[str, object]:
    """Build the committed data of the attestation instrument.

    Parameters
    ----------
    root
        Repository root holding the pinned summary.

    Returns
    -------
    dict[str, object]
        Source pin, cases with the owner's verdicts, the canonicalisation
        oracle, the Python whitespace set and the boundary.

    """
    return {
        "schema": ATTESTATION_INSTRUMENT_SCHEMA,
        "artifact_id": "studio-attestation-instrument-20261115",
        "generated_by": "python tools/studio_attestation_instrument.py --write",
        "verifier": "scpn_quantum_control.attested_result_pack.verify_attested_envelope",
        "envelope_schema": ATTESTED_RESULT_PACK_SCHEMA,
        "claim_boundary": ATTESTED_RESULT_PACK_CLAIM_BOUNDARY,
        "boundary": BOUNDARY,
        "source": {
            "path": SOURCE_PATH,
            "sha256": SOURCE_SHA256,
            "backend": "ibm_fez",
            "status": "aggregate summary, unpromoted pending raw-count review "
            "(results/IBM_HARDWARE_COMPLETE_AUDIT_2026-03-30.md)",
        },
        "cases": build_cases(load_source(root)),
        "canonical_oracle": build_oracle(),
        "python_whitespace": [code for code in range(0x110000) if chr(code).isspace()],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed attestation instrument data.

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
        build=build_attestation_instrument_artifact,
        default_path=DEFAULT_PATH,
        label="attestation instrument data",
        description=__doc__,
    )


if __name__ == "__main__":
    sys.exit(main())
