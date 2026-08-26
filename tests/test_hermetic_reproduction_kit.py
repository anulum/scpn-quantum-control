# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for hermetic reproduction kit
"""Real-surface tests for ``scpn_quantum_control.hermetic_reproduction_kit``."""

from __future__ import annotations

from typing import cast

import pytest

import scpn_quantum_control.hermetic_reproduction_kit as hermetic_reproduction_kit
from scpn_quantum_control.hermetic_reproduction_kit import (
    HERMETIC_REPRODUCTION_CLAIM_BOUNDARY,
    HERMETIC_REPRODUCTION_KIT_SCHEMA,
    DigestCheckResult,
    HermeticKitEntry,
    KitDigestSpec,
    assert_hermetic_kit_integrity,
    build_hermetic_reproduction_kit,
    fixture_payload,
    get_hermetic_kit_entry,
    iter_hermetic_kit_entries,
    list_hermetic_kit_entry_ids,
    probe_hermetic_kit_entry,
    sha256_hex_of,
    verify_digest,
    verify_kit_entry_digests,
)


def test_list_ids_and_core_entries() -> None:
    """Expose unique ordered identifiers and offline core rows."""
    ids = list_hermetic_kit_entry_ids()
    assert ids
    assert len(ids) == len(set(ids))
    core = iter_hermetic_kit_entries(core_only=True)
    assert core
    assert all(row.kind == "core_local" and not row.requires_qpu for row in core)
    assert all(row.digests for row in core)


def test_get_core_and_hardware_entries() -> None:
    """Distinguish core-local and hardware-gated catalogue records."""
    core = get_hermetic_kit_entry("kit:core.transform_algebra_smoke")
    assert core.kind == "core_local"
    assert core.argv
    assert core.claim_boundary == HERMETIC_REPRODUCTION_CLAIM_BOUNDARY
    assert core.digests[0].fixture_payload is not None

    hw = get_hermetic_kit_entry("kit:hardware.qpu_live_reproduction")
    assert hw.requires_qpu is True
    assert hw.reason


def test_get_rejects_blank_and_unknown() -> None:
    """Reject blank and unknown exact-entry lookups."""
    with pytest.raises(ValueError, match="non-empty"):
        get_hermetic_kit_entry("  ")
    with pytest.raises(ValueError, match="unknown hermetic kit entry_id"):
        get_hermetic_kit_entry("kit:invent.green")


def test_build_kit_zero_core_qpu() -> None:
    """Build an internally consistent kit without core QPU requirements."""
    kit = build_hermetic_reproduction_kit()
    assert kit["schema"] == HERMETIC_REPRODUCTION_KIT_SCHEMA
    assert kit["blank_entry_count"] == 0
    assert kit["core_requires_qpu_count"] == 0
    assert kit["entry_count"] == len(kit["entries"])  # type: ignore[arg-type]
    validated = assert_hermetic_kit_integrity(kit)
    assert validated["core_requires_qpu_count"] == 0


def test_sha256_and_verify_digest_paths() -> None:
    """Verify matching, blank, malformed, missing, and altered digests."""
    payload = fixture_payload("fixture:transform_algebra_smoke")
    digest = sha256_hex_of(payload)
    ok = verify_digest(
        label="fixture:transform_algebra_smoke",
        expected_sha256_hex=digest,
        content=payload,
    )
    assert ok.matched is True
    assert ok.refused is False

    with pytest.raises(ValueError, match="label must be a non-empty"):
        verify_digest(label="  ", expected_sha256_hex=digest, content=payload)

    blank = verify_digest(label="x", expected_sha256_hex="", content=payload)
    assert blank.refused is True
    assert blank.matched is False

    malformed = verify_digest(label="x", expected_sha256_hex="deadbeef", content=payload)
    assert malformed.refused is True

    missing = verify_digest(
        label="x",
        expected_sha256_hex=digest,
        content=None,
    )
    assert missing.refused is True

    mismatch = verify_digest(
        label="x",
        expected_sha256_hex=digest,
        content=b"tampered",
    )
    assert mismatch.matched is False
    assert mismatch.refused is True


def test_verify_kit_entry_digests_with_fixtures() -> None:
    """Check declared fixtures and caller-supplied content without execution."""
    results = verify_kit_entry_digests("kit:core.metamorphic_linearity")
    assert results
    assert all(item.matched for item in results)

    # entry without digests returns empty tuple
    empty = verify_kit_entry_digests("kit:optional.framework_parity_extra")
    assert empty == ()

    entry = get_hermetic_kit_entry("kit:core.metamorphic_linearity")
    bad = verify_kit_entry_digests(
        entry.entry_id,
        content_by_label={entry.digests[0].label: b"wrong"},
        use_fixture_payloads=False,
    )
    assert bad[0].matched is False


def test_verify_refuse_invent_green_entry_raises() -> None:
    """Refuse digest verification for invent-green catalogue rows."""
    with pytest.raises(ValueError, match="refuse_invent_green"):
        verify_kit_entry_digests("kit:refuse.invent_green_qpu_digest")


def test_probe_paths() -> None:
    """Report core eligibility and fail-closed unknown-entry outcomes."""
    core = probe_hermetic_kit_entry("kit:core.no_advantage_certificate")
    assert core["found"] is True
    assert core["core_eligible"] is True
    assert core["refused"] is False
    assert core["argv"]

    hw = probe_hermetic_kit_entry("kit:hardware.qpu_live_reproduction")
    assert hw["core_eligible"] is False
    assert hw["refused"] is True

    with pytest.raises(ValueError, match="unknown hermetic kit entry_id"):
        probe_hermetic_kit_entry("kit:missing")
    refused = probe_hermetic_kit_entry("kit:missing", unknown_policy="refuse")
    assert refused["refused"] is True
    assert refused["found"] is False
    with pytest.raises(ValueError, match="unknown_policy"):
        probe_hermetic_kit_entry(
            "kit:missing",
            unknown_policy="invent",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty"):
        probe_hermetic_kit_entry("")


def test_fixture_payload_and_sha256_type() -> None:
    """Reject unknown fixture labels and non-byte hash inputs."""
    with pytest.raises(ValueError, match="unknown fixture"):
        fixture_payload("fixture:nope")
    with pytest.raises(ValueError, match="non-empty"):
        fixture_payload("  ")
    with pytest.raises(TypeError, match="bytes"):
        sha256_hex_of("not-bytes")  # type: ignore[arg-type]


def test_digest_spec_and_entry_validation() -> None:
    """Enforce digest and catalogue-record construction invariants."""
    payload = b"abc\n"
    digest = sha256_hex_of(payload)
    no_fixture = KitDigestSpec(label="l", sha256_hex=digest)
    assert no_fixture.fixture_payload is None
    good = KitDigestSpec(label="l", sha256_hex=digest, fixture_payload=payload)
    assert good.sha256_hex == digest

    with pytest.raises(ValueError, match="label"):
        KitDigestSpec(label="", sha256_hex=digest)
    with pytest.raises(ValueError, match="64-character"):
        KitDigestSpec(label="l", sha256_hex="zz")
    with pytest.raises(ValueError, match="mismatch"):
        KitDigestSpec(label="l", sha256_hex=digest, fixture_payload=b"other")

    with pytest.raises(ValueError, match="entry_id"):
        HermeticKitEntry(
            entry_id="",
            kind="core_local",
            summary="s",
            argv=("echo", "x"),
            cwd=".",
            requires_qpu=False,
            digests=(good,),
        )
    with pytest.raises(ValueError, match="unknown kit entry kind"):
        HermeticKitEntry(
            entry_id="e",
            kind="nope",  # type: ignore[arg-type]
            summary="s",
            argv=("echo", "x"),
            cwd=".",
            requires_qpu=False,
            digests=(good,),
        )
    with pytest.raises(ValueError, match="summary"):
        HermeticKitEntry(
            entry_id="e",
            kind="core_local",
            summary="  ",
            argv=("echo", "x"),
            cwd=".",
            requires_qpu=False,
            digests=(good,),
        )
    with pytest.raises(ValueError, match="argv"):
        HermeticKitEntry(
            entry_id="e",
            kind="core_local",
            summary="s",
            argv=(),
            cwd=".",
            requires_qpu=False,
            digests=(good,),
        )
    with pytest.raises(ValueError, match="cwd"):
        HermeticKitEntry(
            entry_id="e",
            kind="core_local",
            summary="s",
            argv=("echo", "x"),
            cwd="",
            requires_qpu=False,
            digests=(good,),
        )
    with pytest.raises(ValueError, match="must not require QPU"):
        HermeticKitEntry(
            entry_id="e",
            kind="core_local",
            summary="s",
            argv=("echo", "x"),
            cwd=".",
            requires_qpu=True,
            digests=(good,),
        )
    with pytest.raises(ValueError, match="must not carry"):
        HermeticKitEntry(
            entry_id="e",
            kind="core_local",
            summary="s",
            argv=("echo", "x"),
            cwd=".",
            requires_qpu=False,
            digests=(good,),
            reason="nope",
        )
    with pytest.raises(ValueError, match="at least one digest"):
        HermeticKitEntry(
            entry_id="e",
            kind="core_local",
            summary="s",
            argv=("echo", "x"),
            cwd=".",
            requires_qpu=False,
            digests=(),
        )
    with pytest.raises(ValueError, match="require a non-empty reason"):
        HermeticKitEntry(
            entry_id="e",
            kind="hardware_gated",
            summary="s",
            argv=("echo", "x"),
            cwd=".",
            requires_qpu=True,
            digests=(),
            reason="",
        )


def test_digest_check_result_validation() -> None:
    """Reject invalid digest-result labels, messages, and states."""
    with pytest.raises(ValueError, match="label"):
        DigestCheckResult(
            label="",
            matched=True,
            expected_sha256_hex="a" * 64,
            actual_sha256_hex="a" * 64,
            message="m",
        )
    with pytest.raises(ValueError, match="message"):
        DigestCheckResult(
            label="l",
            matched=True,
            expected_sha256_hex="a" * 64,
            actual_sha256_hex="a" * 64,
            message="  ",
        )
    with pytest.raises(ValueError, match="cannot be marked matched"):
        DigestCheckResult(
            label="l",
            matched=True,
            expected_sha256_hex="a" * 64,
            actual_sha256_hex="a" * 64,
            message="m",
            refused=True,
        )


def test_assert_integrity_rejects_invalid() -> None:
    """Reject malformed rows and inconsistent aggregate counters."""
    with pytest.raises(ValueError, match="non-empty entries"):
        assert_hermetic_kit_integrity({"entries": []})
    with pytest.raises(ValueError, match="blank"):
        assert_hermetic_kit_integrity(
            {
                "entries": [{"entry_id": "", "kind": "core_local"}],
                "blank_entry_count": 0,
                "entry_count": 1,
                "core_requires_qpu_count": 0,
            }
        )
    with pytest.raises(ValueError, match="without reason"):
        assert_hermetic_kit_integrity(
            {
                "entries": [
                    {
                        "entry_id": "e",
                        "kind": "hardware_gated",
                        "requires_qpu": True,
                        "reason": "",
                    }
                ],
                "blank_entry_count": 0,
                "entry_count": 1,
                "core_requires_qpu_count": 0,
            }
        )
    with pytest.raises(ValueError, match="requiring QPU"):
        assert_hermetic_kit_integrity(
            {
                "entries": [
                    {
                        "entry_id": "e",
                        "kind": "core_local",
                        "requires_qpu": True,
                    }
                ],
                "blank_entry_count": 0,
                "entry_count": 1,
                "core_requires_qpu_count": 0,
            }
        )
    good = get_hermetic_kit_entry("kit:core.transform_algebra_smoke").to_dict()
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_hermetic_kit_integrity(
            {
                "entries": [good],
                "blank_entry_count": 1,
                "entry_count": 1,
                "core_requires_qpu_count": 0,
            }
        )
    with pytest.raises(ValueError, match="entry_count"):
        assert_hermetic_kit_integrity(
            {
                "entries": [good],
                "blank_entry_count": 0,
                "entry_count": 99,
                "core_requires_qpu_count": 0,
            }
        )
    with pytest.raises(ValueError, match="core_requires_qpu_count"):
        assert_hermetic_kit_integrity(
            {
                "entries": [good],
                "blank_entry_count": 0,
                "entry_count": 1,
                "core_requires_qpu_count": 1,
            }
        )
    with pytest.raises(ValueError, match="mapping"):
        assert_hermetic_kit_integrity(
            {
                "entries": ["x"],
                "blank_entry_count": 0,
                "entry_count": 1,
                "core_requires_qpu_count": 0,
            }
        )
    with pytest.raises(ValueError, match="blank"):
        assert_hermetic_kit_integrity(
            {
                "entries": [
                    {
                        "entry_id": "e",
                        "kind": "not-a-kind",
                        "reason": "r",
                    }
                ],
                "blank_entry_count": 0,
                "entry_count": 1,
                "core_requires_qpu_count": 0,
            }
        )


def test_catalogue_map_rejects_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject duplicate canonical entry identifiers."""
    row = get_hermetic_kit_entry("kit:core.transform_algebra_smoke")
    monkeypatch.setattr(hermetic_reproduction_kit, "_CANONICAL_ENTRIES", (row, row))
    with pytest.raises(RuntimeError, match="duplicate entry_id"):
        hermetic_reproduction_kit._catalogue_map()


def test_iter_kind_filter_and_to_dict() -> None:
    """Filter stable records and serialise public value objects."""
    all_rows = iter_hermetic_kit_entries()
    assert len(all_rows) == len(list_hermetic_kit_entry_ids())
    optional = iter_hermetic_kit_entries(kind="optional_extra")
    assert optional
    assert all(row.kind == "optional_extra" for row in optional)
    payload = get_hermetic_kit_entry("kit:core.transform_algebra_smoke").to_dict()
    assert cast(str, payload["entry_id"]).startswith("kit:core.")
    assert isinstance(payload["digests"], list)
    check = verify_digest(
        label="l",
        expected_sha256_hex=sha256_hex_of(b"z"),
        content=b"z",
    )
    assert check.to_dict()["matched"] is True
