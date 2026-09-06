# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — CORE-G02 corpus builder owner tests
"""Own the corpus builder itself, separately from the corpus it produces.

``tests/test_versioned_contract_custody.py`` uses the builder to prove the
frozen corpus still matches what the producers emit. This module owns the
builder's own behaviour: what it writes, what it refuses to invent, and that
its command-line entry point is the same code path a refresh actually runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from scpn_quantum_control import stable_core_product as scp
from tools.core_g02_contract_corpus import (
    COMPANION_MODULE,
    COMPANION_PENDING,
    COMPANION_SCHEMA,
    CORPUS_SCHEMA,
    EXECUTABLE,
    build_cases,
    build_manifest,
    case_manifest_entry,
    main,
    write_corpus,
)


class TestCases:
    """The case list is the reviewer's list, bound to real producers."""

    def test_every_case_carries_a_producer_reader_and_rationale(self) -> None:
        """A case without these cannot be reviewed or re-derived."""
        for case in build_cases():
            assert case.producer
            assert case.reader
            assert case.rationale
            assert case.expectation in {"accept", "reject"}

    def test_pending_cases_are_exactly_those_reading_the_companion(self) -> None:
        """A case may only be pending because the companion is absent."""
        pending = [case for case in build_cases() if case.status == COMPANION_PENDING]
        executable = [case for case in build_cases() if case.status == EXECUTABLE]

        assert pending
        assert executable
        for case in pending:
            assert case.reader.startswith(COMPANION_MODULE)
        for case in executable:
            assert not case.reader.startswith(COMPANION_MODULE)

    def test_case_identifiers_are_unique(self) -> None:
        """Identifiers are fixture filenames, so a collision would lose bytes."""
        identifiers = [case.case_id for case in build_cases()]

        assert len(identifiers) == len(set(identifiers))


class TestManifestEntries:
    """A manifest row must describe bytes that exist, or declare none."""

    def test_a_case_with_bytes_records_their_digest(self) -> None:
        """The digest is computed from the payload, never carried separately."""
        case = next(case for case in build_cases() if case.payload is not None)
        assert case.payload is not None

        entry = case_manifest_entry(case)

        assert entry["fixture"] == f"{case.case_id}.json"
        assert entry["fixture_sha256"] == scp.digest_stable_core_payload(case.payload)

    def test_a_case_without_bytes_records_no_fixture(self) -> None:
        """Type-pinning cases must not gain an invented payload."""
        case = next(case for case in build_cases() if case.payload is None)

        entry = case_manifest_entry(case)

        assert entry["fixture"] is None
        assert entry["fixture_sha256"] is None

    def test_the_manifest_names_its_schema_and_the_companion(self) -> None:
        """A corpus without its own version cannot be superseded cleanly."""
        manifest = build_manifest(build_cases())

        assert manifest["schema"] == CORPUS_SCHEMA
        assert manifest["companion_schema"] == COMPANION_SCHEMA
        assert manifest["model_schema_version"] == scp.STABLE_CORE_MODEL_SCHEMA_VERSION


class TestWriting:
    """Writing must produce exactly the bytes the manifest describes."""

    def test_write_corpus_writes_one_file_per_byte_case(self, tmp_path: Path) -> None:
        """Every declared fixture exists on disk with the declared digest.

        Parameters
        ----------
        tmp_path
            Fixture.

        """
        manifest = write_corpus(tmp_path)

        written = {path.name for path in tmp_path.iterdir()}
        declared = {row["fixture"] for row in manifest["cases"] if row["fixture"] is not None}

        assert declared <= written
        assert "manifest.json" in written
        for row in manifest["cases"]:
            if row["fixture"] is None:
                continue
            payload = json.loads((tmp_path / row["fixture"]).read_text(encoding="utf-8"))

            assert scp.digest_stable_core_payload(payload) == row["fixture_sha256"]

    def test_writing_twice_is_byte_identical(self, tmp_path: Path) -> None:
        """The corpus must be reproducible, not merely producible.

        Parameters
        ----------
        tmp_path
            Fixture.

        """
        first = tmp_path / "first"
        second = tmp_path / "second"

        write_corpus(first)
        write_corpus(second)

        for path in sorted(first.iterdir()):
            assert path.read_bytes() == (second / path.name).read_bytes()

    def test_main_writes_the_corpus_to_the_requested_directory(
        self, tmp_path: Path, capsys: object
    ) -> None:
        """The entry point is the same path a refresh runs.

        Parameters
        ----------
        tmp_path, capsys
            Fixtures.

        """
        destination = tmp_path / "corpus"

        assert main([str(destination)]) == 0

        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["schema"] == CORPUS_SCHEMA
        assert len(manifest["cases"]) == len(build_cases())
