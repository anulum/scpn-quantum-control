# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — custody corpus builder ownership
"""Own the corpus builder, separately from the corpus it produces.

``tests/test_versioned_contract_custody.py`` uses the builder to prove the
frozen corpus still matches what the producers emit. This module owns the
builder itself: how it classifies evidence, what it refuses to invent, and that
a manifest carrying the superseded internally-coded schema name is rejected
rather than quietly read.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_quantum_control import stable_core_product as scp
from tools import contract_custody_design_vectors as vectors
from tools.contract_custody_corpus import (
    CORPUS_SCHEMA,
    DESIGN_VECTOR,
    EXECUTED,
    SOURCE_FACT,
    SUPERSEDED_CORPUS_SCHEMAS,
    build_cases,
    build_manifest,
    case_manifest_entry,
    main,
    validate_manifest_schema,
    write_corpus,
)


class TestEvidenceClassification:
    """The three evidence classes must stay separable and honestly labelled."""

    def test_type_introspection_is_not_executed_reader_evidence(self) -> None:
        """A source fact has no reader and cannot qualify a binding."""
        case = next(
            case
            for case in build_cases()
            if case.case_id == "same_named_problem_types_remain_separable"
        )
        assert case.status == "source_fact"
        assert case.reader is None
        assert case.payload is None

    def test_fisher_mapping_names_the_called_analysis_not_a_hal_backend(self) -> None:
        """Expected-count analysis supplies no observed backend evidence."""
        case = next(
            case
            for case in build_cases()
            if case.case_id == "non_count_route_carries_no_measurement_mapping"
        )
        assert case.family == "Fidelity"
        assert case.reader == (
            "scpn_quantum_control.phase.qnode_circuit_differentiation."
            "phase_qnode_computational_basis_fisher_information"
        )

    def test_every_case_declares_its_evidence_class_and_reader_applicability(self) -> None:
        """Source facts omit readers; executable and proposed cases name them."""
        for case in build_cases():
            assert case.producer
            assert (case.reader is None) == (case.status == SOURCE_FACT)
            assert case.rationale
            assert case.expectation in {"accept", "reject"}
            assert case.status in {EXECUTED, DESIGN_VECTOR, SOURCE_FACT}

    def test_design_vectors_name_the_proposed_reader(self) -> None:
        """A vector is unexecuted because its reader is proposed, not built."""
        proposed = [case for case in build_cases() if case.status == DESIGN_VECTOR]

        assert proposed
        for case in proposed:
            assert case.reader is not None
            assert case.reader.endswith("validate_semantic_binding")
            assert case.producer == vectors.COMPANION_SCHEMA

    def test_executed_cases_do_not_name_the_proposed_companion_reader(self) -> None:
        """Guard reader names without claiming importability or actual execution."""
        executed = [case for case in build_cases() if case.status == EXECUTED]

        assert executed
        for case in executed:
            assert case.reader is not None
            assert "semantic_record" not in case.reader

    def test_case_identifiers_are_unique(self) -> None:
        """Identifiers are fixture filenames, so a collision would lose bytes."""
        identifiers = [case.case_id for case in build_cases()]

        assert len(identifiers) == len(set(identifiers))

    def test_no_identifier_encodes_an_internal_planning_code(self) -> None:
        """Descriptive naming is a Tier-0 rule and applies to fixture names."""
        for case in build_cases():
            lowered = case.case_id.lower()

            assert "core_g" not in lowered
            assert "core-g" not in lowered


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
        """Behaviour-pinning cases must not gain an invented payload."""
        case = next(case for case in build_cases() if case.payload is None)

        entry = case_manifest_entry(case)

        assert entry["fixture"] is None
        assert entry["fixture_sha256"] is None

    def test_the_manifest_declares_that_vectors_are_unexecuted(self) -> None:
        """The honesty of the corpus must survive being read by a stranger."""
        manifest = build_manifest(build_cases())

        assert manifest["schema"] == CORPUS_SCHEMA
        assert manifest["design_vectors_are_unexecuted"] is True
        assert manifest["model_schema_version"] == scp.STABLE_CORE_MODEL_SCHEMA_VERSION


class TestSchemaAdmission:
    """The superseded coded schema must be refused, not silently accepted."""

    def test_the_current_schema_is_accepted(self) -> None:
        """The descriptive successor is the only readable name."""
        assert validate_manifest_schema({"schema": CORPUS_SCHEMA}) == CORPUS_SCHEMA

    def test_the_superseded_coded_schema_is_refused_by_name(self) -> None:
        """Reading it would keep an internal identifier alive in artefacts."""
        superseded = next(iter(SUPERSEDED_CORPUS_SCHEMAS))

        with pytest.raises(ValueError, match="internal planning identifier"):
            validate_manifest_schema({"schema": superseded})

    def test_an_unrelated_schema_is_refused(self) -> None:
        """An unknown corpus version must not be read optimistically."""
        with pytest.raises(ValueError, match="unknown corpus schema"):
            validate_manifest_schema({"schema": "something_else.v1"})

    def test_a_manifest_without_a_schema_is_refused(self) -> None:
        """A missing schema is unknown, not a default."""
        with pytest.raises(ValueError, match="unknown corpus schema"):
            validate_manifest_schema({})


class TestWriting:
    """Writing must produce exactly the bytes the manifest describes."""

    def test_module_command_writes_reproducible_source_records(self, tmp_path: Path) -> None:
        """Run the actual module command and verify its captured planner bytes.

        Parameters
        ----------
        tmp_path
            Isolated destination; no repository fixture is overwritten.

        """
        result = subprocess.run(
            [sys.executable, "-m", "tools.contract_custody_corpus", str(tmp_path)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest == build_manifest(build_cases())
        assert result.stdout.strip() == f"{len(manifest['cases'])} cases written"
        source = json.loads(
            (tmp_path / "planner_preserves_null_request_beside_default.json").read_text(
                encoding="utf-8"
            )
        )
        assert source == vectors.planning_policy_source()

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

    def test_main_writes_the_corpus_to_the_requested_directory(self, tmp_path: Path) -> None:
        """The entry point is the same path a refresh runs.

        Parameters
        ----------
        tmp_path
            Fixture.

        """
        destination = tmp_path / "corpus"

        assert main([str(destination)]) == 0

        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))

        assert validate_manifest_schema(manifest) == CORPUS_SCHEMA
        assert len(manifest["cases"]) == len(build_cases())
