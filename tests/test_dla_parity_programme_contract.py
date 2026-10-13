# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — dla_parity.programme_contract tests
"""Tests of the protocol of the DLA-parity ordinary programme.

The document under test is the one the worker writes for the committed inputs.
Each case changes that real document in one stated way and requires the
validator, and where the shape is concerned the shipped JSON Schema, to refuse it.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.dla_parity import programme, programme_contract
from scpn_quantum_control.dla_parity.dataset import DEFAULT_DATA_DIR, RUN_FILES
from scpn_quantum_control.dla_parity.reproduce import (
    DEFAULT_PUBLISHED_SUMMARY,
    ReproductionTolerance,
)

JsonObject = dict[str, Any]
Change = Callable[[JsonObject], None]
SCHEMA = json.loads(
    (Path(programme_contract.__file__).parent / "programme_output_v1.schema.json").read_text(
        "utf-8"
    )
)


def _pin(path: Path) -> JsonObject:
    content = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


@pytest.fixture(scope="module")
def committed(tmp_path_factory: pytest.TempPathFactory) -> JsonObject:
    """Return the document the worker writes for the committed inputs."""
    request = tmp_path_factory.mktemp("contract") / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema": programme_contract.REQUEST_SCHEMA,
                "raw_run_files": [_pin(DEFAULT_DATA_DIR / name) for name in RUN_FILES],
                "published_summary": _pin(DEFAULT_PUBLISHED_SUMMARY),
                "tolerance": dataclasses.asdict(ReproductionTolerance()),
            }
        ),
        encoding="utf-8",
    )
    document: JsonObject = programme.run_request(request)
    reread: JsonObject = json.loads(programme_contract.serialise(document))
    return reread


def _schema_refuses(document: JsonObject) -> bool:
    validator = importlib.import_module("jsonschema").Draft202012Validator(SCHEMA)
    return bool(list(validator.iter_errors(document)))


def test_the_committed_document_satisfies_the_validator_and_the_schema(
    committed: JsonObject,
) -> None:
    """Require the real document to pass both checks unchanged."""
    programme_contract.validate_output_document(committed)
    assert not _schema_refuses(committed)
    refusal = {
        "schema": programme_contract.OUTPUT_SCHEMA,
        "outcome": "refused",
        "refusal": {"category": "digest_mismatch", "detail": "an input differs from its pin"},
    }
    programme_contract.validate_output_document(refusal)
    assert not _schema_refuses(refusal)


def _other_schema(d: JsonObject) -> None:
    d["schema"] = "something.else.v1"


def _unknown_outcome(d: JsonObject) -> None:
    d["outcome"] = "accepted"


def _extra_member(d: JsonObject) -> None:
    d["approved"] = True


def _record_dropped(d: JsonObject) -> None:
    del d["selection"]["records"][5]


def _record_repeated(d: JsonObject) -> None:
    d["selection"]["records"].insert(1, copy.deepcopy(d["selection"]["records"][0]))


def _records_reordered(d: JsonObject) -> None:
    records = d["selection"]["records"]
    records[0], records[1] = records[1], records[0]


def _unknown_reason(d: JsonObject) -> None:
    d["selection"]["records"][0]["reason"] = "approved"


def _count_inflated(d: JsonObject) -> None:
    d["selection"]["counts"]["selected"] += 1


def _depth_repeated(d: JsonObject) -> None:
    d["depth_summaries"][1]["depth"] = d["depth_summaries"][0]["depth"]


def _circuits_used_changed(d: JsonObject) -> None:
    d["result"]["n_circuits_used"] -= 1


def _peak_depth_unknown(d: JsonObject) -> None:
    d["result"]["peak_asymmetry_depth"] = 3


def _coverage_truncated(d: JsonObject) -> None:
    del d["comparison_coverage"]["scaled_rule"]


def _coverage_fabricated(d: JsonObject) -> None:
    d["comparison_coverage"]["independently_verified"] = "yes"


def _claim_dropped(d: JsonObject) -> None:
    del d["claims"][10]


def _claim_for_an_unreferenced_scalar(d: JsonObject) -> None:
    d["claims"].append(dict(d["claims"][0], name="mean_asymmetry_relative"))


def _identity_check_dropped(d: JsonObject) -> None:
    del d["identity_checks"][-1]


def _claim_passed_against_its_difference(d: JsonObject) -> None:
    d["claims"][3]["difference"] = 1.0


def _identity_check_passed_against_its_values(d: JsonObject) -> None:
    d["identity_checks"][0]["recomputed"] += 1


def _outcome_against_comparisons(d: JsonObject) -> None:
    d["outcome"] = "mismatch"


def _refusal_with_a_scalar(d: JsonObject) -> None:
    d["outcome"] = "refused"
    d["refusal"] = {"category": "digest_mismatch", "detail": "x"}


@pytest.mark.parametrize(
    ("change", "message", "shape"),
    [
        (_other_schema, "unknown schema identifier", True),
        (_unknown_outcome, "unknown outcome", True),
        (_extra_member, "missing or extra top-level member", True),
        (_record_dropped, "selection counts do not account", False),
        (_record_repeated, "records are not ordered and unique", False),
        (_records_reordered, "records are not ordered and unique", False),
        (_unknown_reason, "unknown selection reason", True),
        (_count_inflated, "selection counts do not account", False),
        (_depth_repeated, "depths are not ascending and unique", False),
        (_circuits_used_changed, "selected records, replicates and circuits used disagree", False),
        (_peak_depth_unknown, "peak depth is not a depth", False),
        (_coverage_truncated, "incomplete comparison coverage", True),
        (_coverage_fabricated, "incomplete comparison coverage", True),
        (_claim_dropped, "claims are not the complete population", False),
        (_claim_for_an_unreferenced_scalar, "claims are not the complete population", False),
        (_identity_check_dropped, "identity checks are not the complete population", False),
        (_claim_passed_against_its_difference, "contradicts its difference", False),
        (_identity_check_passed_against_its_values, "contradicts its values", False),
        (_outcome_against_comparisons, "outcome contradicts the comparisons", False),
        (_refusal_with_a_scalar, "a refusal carries a scalar", True),
    ],
)
def test_a_changed_document_is_refused_by_the_validator(
    committed: JsonObject, change: Change, message: str, shape: bool
) -> None:
    """Change the real document in one way and require the stated refusal.

    ``shape`` says whether the JSON Schema alone also refuses the change; the
    cases where it does not are the invariants the validator exists for.
    """
    document = copy.deepcopy(committed)
    change(document)
    with pytest.raises(ValueError, match=message):
        programme_contract.validate_output_document(document)
    assert _schema_refuses(document) is shape


@pytest.mark.parametrize(
    "refusal",
    [
        {"category": "approved", "detail": "x"},
        {"category": "digest_mismatch"},
        {"category": "digest_mismatch", "detail": 3},
        "digest_mismatch",
    ],
)
def test_a_malformed_refusal_is_refused(refusal: object) -> None:
    """Require an unknown category, a missing detail and a non-object refusal to be refused."""
    document = {
        "schema": programme_contract.OUTPUT_SCHEMA,
        "outcome": "refused",
        "refusal": refusal,
    }
    with pytest.raises(ValueError, match="malformed refusal"):
        programme_contract.validate_output_document(document)
    assert _schema_refuses(document)


@pytest.mark.parametrize(
    ("value", "number", "count"),
    [
        (0, True, True),
        (7, True, True),
        (-1, True, False),
        (0.5, True, False),
        (10**400, False, True),
        (float("inf"), False, False),
        (float("nan"), False, False),
        (True, False, False),
        ("1", False, False),
        (None, False, False),
    ],
)
def test_numbers_and_counts_of_the_protocol(value: object, number: bool, count: bool) -> None:
    """Classify values, among them an integer beyond binary64 and a boolean."""
    assert programme_contract.is_number(value) is number
    assert programme_contract.is_count(value) is count


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"a": 1, "a": 2}', "repeats a member name"),
        (b'{"a": {"b": 1, "b": 1}}', "repeats a member name"),
        (b'{"a": NaN}', "is not a JSON value"),
        (b'{"a": -Infinity}', "is not a JSON value"),
        (b"[1, 2]", "is not a JSON object"),
        (b"\xff\xfe", "is not UTF-8"),
        (b"{not json", "Expecting property name"),
        pytest.param(b"[" * 100_000 + b"]" * 100_000, "nested too deeply", id="deep-nesting"),
    ],
)
def test_the_reader_accepts_one_unambiguous_object_only(content: bytes, message: str) -> None:
    """Require a repeated member, a non-finite literal and a non-object to be refused."""
    with pytest.raises(ValueError, match=message):
        programme_contract.parse_json_object(content)


def test_the_reader_and_the_serialiser_round_trip_a_document() -> None:
    """Require sorted compact output, a refusal to write infinity, and an exact read back."""
    assert programme_contract.serialise({"b": 1, "a": [0.1, 2]}) == '{"a":[0.1,2],"b":1}\n'
    assert programme_contract.parse_json_object(b'{"a":[0.1,2],"b":1}\n') == {
        "a": [0.1, 2],
        "b": 1,
    }
    with pytest.raises(ValueError, match="Out of range float values"):
        programme_contract.serialise({"value": float("inf")})
