# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — claims compared only under compatible assumptions
"""Check comparison keys, criteria, the federated gate, filters and source custody."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path
from typing import cast

import pytest

import tools.studio_atlas_claim_comparison as comparison_tool
from scpn_quantum_control.atlas.comparability import (
    RECORD_DIGEST_SCHEMA,
    VALUES_SCHEMA,
    ComparabilityError,
    ComparableValue,
    Criterion,
    build_comparison_record,
    compare_values,
    filter_comparisons,
    parse_comparable_value,
    value_to_wire,
    verify_value_sources,
)
from scpn_quantum_control.studio_workspace.canonical import canonical_digest
from tools.studio_atlas_claim_comparison import (
    DEFAULT_PATH,
    VALUES_PATH,
    build_committed_claim_comparisons,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME = ("scpn-quantum-control", "1.2.0")
Json = dict[str, object]


def key(**overrides: object) -> Json:
    """Return a complete comparison key, with fields replaced.

    Parameters
    ----------
    **overrides : object
        Fields to replace.

    Returns
    -------
    Json
        The key.

    """
    return {
        "estimand": "parity leakage asymmetry",
        "unit": "probability",
        "model": "DLA parity circuit",
        "dataset": "campaign-a",
        "calibration": "epoch-1",
        "noise": "device noise as run",
        "uq_method": "Wilson 95 %",
        **overrides,
    }


def payload(**overrides: object) -> Json:
    """Return the JSON form of a home-producer value, with fields replaced.

    Parameters
    ----------
    **overrides : object
        Fields to replace.

    Returns
    -------
    Json
        The value's JSON form.

    """
    return {
        "value_id": "a",
        "producer": HOME[0],
        "producer_version": HOME[1],
        "key": key(),
        "estimate": 0.5,
        "interval": [0.4, 0.6],
        "negative": False,
        "source": {"path": "data/x.json", "sha256": "0" * 64},
        **overrides,
    }


def value(value_id: str, interval: list[float] | None, **overrides: object) -> ComparableValue:
    """Parse a home-producer value, with fields replaced.

    Parameters
    ----------
    value_id : str
        Identifier.
    interval : list[float] | None
        ``[low, high]`` or ``None``.
    **overrides : object
        Fields to replace.

    Returns
    -------
    ComparableValue
        The value.

    """
    return parse_comparable_value(payload(value_id=value_id, interval=interval, **overrides))


def test_atlas_claim_comparability_01() -> None:
    """Equal numbers with different estimands or units are non-comparable, never a contradiction."""
    left = value("a", [0.4, 0.6])
    estimand = value("b", [0.4, 0.6], key=key(estimand="total leakage"))
    unit = value("c", [0.4, 0.6], key=key(unit="percent"))

    by_estimand = compare_values(left, estimand, "interval_overlap", home=HOME)
    by_unit = compare_values(left, unit, "interval_overlap", home=HOME)

    assert (by_estimand.status, by_estimand.reasons) == (
        "non_comparable",
        ("estimand differs: parity leakage asymmetry / total leakage",),
    )
    assert (by_unit.status, by_unit.reasons) == (
        "non_comparable",
        ("unit differs: probability / percent",),
    )


def test_atlas_claim_comparability_02() -> None:
    """Matching keys are judged by the declared criterion."""
    left = value("a", [0.40, 0.46])
    overlapping = value("b", [0.45, 0.50])
    apart = value("c", [0.47, 0.50])
    positive = value("d", [0.01, 0.02])
    negative = value("e", [-0.02, -0.01])

    assert compare_values(left, overlapping, "interval_overlap", home=HOME).status == "agreement"
    disagreement = compare_values(left, apart, "interval_overlap", home=HOME)
    assert (disagreement.status, disagreement.reasons) == (
        "disagreement",
        ("the intervals do not overlap",),
    )
    same = compare_values(positive, value("f", [0.03, 0.04]), "same_sign", home=HOME)
    assert same.status == "agreement"
    opposite = compare_values(positive, negative, "same_sign", home=HOME)
    assert (opposite.status, opposite.reasons) == (
        "disagreement",
        ("the intervals exclude zero on opposite sides",),
    )
    zero = compare_values(positive, value("g", [-0.01, 0.01]), "same_sign", home=HOME)
    assert zero.status == "insufficient_evidence"


def test_atlas_claim_comparability_03() -> None:
    """A missing assumption or a missing interval is insufficient evidence."""
    unstated = value("b", [0.4, 0.6], key=key(calibration=None))
    result = compare_values(value("a", [0.4, 0.6]), unstated, "interval_overlap", home=HOME)
    assert (result.status, result.reasons) == (
        "insufficient_evidence",
        ("calibration is not stated for b",),
    )
    pointless = compare_values(
        value("a", [0.4, 0.6]), value("c", None), "interval_overlap", home=HOME
    )
    assert (pointless.status, pointless.reasons) == (
        "insufficient_evidence",
        ("interval_overlap needs an interval for both values",),
    )


def test_atlas_claim_comparability_04() -> None:
    """A comparison involving a negative record stays listed whatever the filter selects."""
    failed = value("replication", [0.4, 0.6], negative=True)
    other_unit = value("g", [0.4, 0.6], key=key(unit="%"))
    results = [
        compare_values(value("a", [0.4, 0.6]), failed, "interval_overlap", home=HOME),
        compare_values(
            value("b", [0.4, 0.46]), value("c", [0.47, 0.5]), "interval_overlap", home=HOME
        ),
        compare_values(
            value("d", [0.4, 0.6]), value("e", [0.4, 0.6]), "interval_overlap", home=HOME
        ),
        compare_values(value("f", [0.4, 0.6]), other_unit, "interval_overlap", home=HOME),
    ]
    shown, counter = filter_comparisons(results, ["non_comparable"])

    assert [(item.left, item.status) for item in shown] == [("f", "non_comparable")]
    assert [(item.left, item.right, item.negative, item.status) for item in counter] == [
        ("a", "replication", True, "agreement"),
        ("b", "c", False, "disagreement"),
    ]
    everything, none = filter_comparisons(results, ["agreement", "disagreement", "non_comparable"])
    assert len(everything) == 4
    assert none == ()


def test_atlas_claim_comparability_05() -> None:
    """Without accepted producer equivalence only the federated comparison is disabled."""
    local = value("a", [0.4, 0.6])
    external = value("x", [0.4, 0.6], producer="other-studio", producer_version="3.1")
    newer = value("y", [0.4, 0.6], producer_version="1.3.0")

    gated = compare_values(local, external, "interval_overlap", home=HOME)
    assert (gated.status, gated.reasons, gated.federated) == (
        "non_comparable",
        ("producer_equivalence_not_accepted: other-studio 3.1",),
        True,
    )
    plain = compare_values(local, value("b", [0.45, 0.5]), "interval_overlap", home=HOME)
    assert (plain.status, plain.federated) == ("agreement", False)
    accepted = compare_values(
        local,
        external,
        "interval_overlap",
        home=HOME,
        accepted=frozenset({("other-studio", "3.1")}),
    )
    assert (accepted.status, accepted.federated) == ("agreement", True)
    version_gate = compare_values(local, newer, "interval_overlap", home=HOME)
    assert version_gate.reasons == (
        "producer_equivalence_not_accepted: scpn-quantum-control 1.3.0",
    )


def test_an_unknown_criterion_is_refused() -> None:
    """Only the declared criteria judge a pair."""
    closeness = cast(Criterion, "closeness")
    with pytest.raises(ComparabilityError, match="unknown criterion"):
        compare_values(value("a", [0.4, 0.6]), value("b", [0.4, 0.6]), closeness, home=HOME)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"key": {"estimand": 1}}, r"\$\.key\.estimand must be non-empty text"),
        ({"key": {"colour": "red"}}, r"\$\.key must be an object of"),
        ({"key": "all"}, r"\$\.key must be an object of"),
        ({"interval": [1.0]}, r"\$\.interval must be \[low, high\]"),
        ({"interval": [2.0, 1.0]}, "must not be reversed"),
        ({"interval": [0.1, float("nan")]}, r"\$\.interval\[1\] must be a finite number"),
        ({"estimate": True}, r"\$\.estimate must be a finite number"),
        ({"negative": "no"}, r"\$\.negative must be a boolean"),
        ({"source": "data/x.json"}, r"\$\.source must be an object"),
        ({"value_id": ""}, r"\$\.value_id must be non-empty text"),
    ],
)
def test_a_malformed_value_is_refused(change: Json, message: str) -> None:
    """Each malformed field of a value is refused with its address.

    Parameters
    ----------
    change : Json
        Fields to replace.
    message : str
        Expected part of the refusal.

    """
    with pytest.raises(ComparabilityError, match=message):
        parse_comparable_value(payload(**change))
    with pytest.raises(ComparabilityError, match=r"\$ must be an object"):
        parse_comparable_value([])


D10 = "data/iqm_paper_replication/iqm_dla_d10_retest_primary_analysis_2026-07-22.json"


def sourced(root: Path, **overrides: object) -> ComparableValue:
    """Write a cited analysis below ``root`` and return a value citing it.

    Parameters
    ----------
    root : Path
        Repository root.
    **overrides : object
        Value fields to replace.

    Returns
    -------
    ComparableValue
        The value, whose numbers equal the cited fields.

    """
    path = root / "data/analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"result": {"diff": 0.25, "ci": [0.1, 0.4]}}), encoding="utf-8")
    source = {
        "path": "data/analysis.json",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "estimate": "result.diff",
        "low": "result.ci.0",
        "high": "result.ci.1",
    }
    fields: Json = {"estimate": 0.25, "interval": [0.1, 0.4], "source": source, **overrides}
    return parse_comparable_value(payload(**fields))


def test_a_value_equal_to_its_cited_fields_is_verified(tmp_path: Path) -> None:
    """Numbers equal to the cited committed fields pass; a value without numbers needs no pointer.

    Parameters
    ----------
    tmp_path : Path
        Repository root.

    """
    verify_value_sources([sourced(tmp_path)], tmp_path)
    bare = sourced(tmp_path)
    source = dict(bare.source)
    verify_value_sources(
        [parse_comparable_value(payload(estimate=None, interval=None, source=source))], tmp_path
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"estimate": 0.26}, "estimate differs from result.diff"),
        ({"interval": [0.1, 0.5]}, "high differs from result.ci.1"),
    ],
)
def test_a_value_that_differs_from_its_source_is_refused(
    tmp_path: Path, change: Json, message: str
) -> None:
    """A number that is not exactly the cited field is refused.

    Parameters
    ----------
    tmp_path : Path
        Repository root.
    change : Json
        Value fields to replace.
    message : str
        Expected part of the refusal.

    """
    with pytest.raises(ComparabilityError, match=message):
        verify_value_sources([sourced(tmp_path, **change)], tmp_path)


@pytest.mark.parametrize(
    ("source_change", "message"),
    [
        ({"sha256": "1" * 64}, "does not match its recorded sha256"),
        ({"sha256": "ABC"}, "source sha256 must be a lowercase digest"),
        ({"path": "../escape.json"}, "source path must be a safe relative path"),
        ({"path": "data/absent.json"}, "cannot be read"),
        ({"estimate": "result.missing"}, "pointer result.missing does not resolve"),
        ({"low": "result.ci.7"}, "pointer result.ci.7 does not resolve"),
        ({"high": "result.diff.0"}, "pointer result.diff.0 does not resolve"),
    ],
)
def test_an_unverifiable_source_is_refused(
    tmp_path: Path, source_change: Json, message: str
) -> None:
    """A wrong digest, an unsafe or absent path or an unresolved pointer is refused.

    Parameters
    ----------
    tmp_path : Path
        Repository root.
    source_change : Json
        Source fields to replace.
    message : str
        Expected part of the refusal.

    """
    base = sourced(tmp_path)
    source = {**dict(base.source), **source_change}
    with pytest.raises(ComparabilityError, match=message):
        verify_value_sources(
            [parse_comparable_value(payload(estimate=0.25, interval=[0.1, 0.4], source=source))],
            tmp_path,
        )


def test_a_number_without_a_pointer_or_a_non_json_source_is_refused(tmp_path: Path) -> None:
    """Each carried number needs its pointer, and the cited file must be JSON.

    Parameters
    ----------
    tmp_path : Path
        Repository root.

    """
    source = {key: name for key, name in sourced(tmp_path).source if key != "low"}
    with pytest.raises(ComparabilityError, match="low has no source pointer"):
        verify_value_sources(
            [parse_comparable_value(payload(estimate=0.25, interval=[0.1, 0.4], source=source))],
            tmp_path,
        )
    text = tmp_path / "data/text.json"
    text.write_text("not json", encoding="utf-8")
    plain = {"path": "data/text.json", "sha256": hashlib.sha256(text.read_bytes()).hexdigest()}
    with pytest.raises(ComparabilityError, match="is not JSON"):
        verify_value_sources(
            [parse_comparable_value(payload(estimate=None, interval=None, source=plain))],
            tmp_path,
        )


def test_the_d10_retest_difference_is_cited_exactly() -> None:
    """The committed d10 retest difference verifies against its own field."""
    path = REPO_ROOT / D10
    source = {
        "path": D10,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "estimate": "primary_sign_replication.difference_even_minus_odd",
    }
    recorded = json.loads(path.read_text(encoding="utf-8"))["primary_sign_replication"]
    retest = parse_comparable_value(
        payload(
            estimate=recorded["difference_even_minus_odd"],
            interval=None,
            negative=True,
            source=source,
        )
    )
    verify_value_sources([retest], REPO_ROOT)
    assert retest.negative is True


def declaration(root: Path, **overrides: object) -> Json:
    """Return a declaration of two cited values and the pairs to compare.

    Parameters
    ----------
    root : Path
        Repository root holding the cited analysis.
    **overrides : object
        Top-level fields to replace.

    Returns
    -------
    Json
        The declaration.

    """
    first = value_to_wire(sourced(root))
    second = {**first, "value_id": "b", "key": key(unit="percent")}
    return {
        "schema": VALUES_SCHEMA,
        "home": {"producer": HOME[0], "version": HOME[1]},
        "accepted": [{"producer": "other-studio", "version": "3.1"}],
        "values": [first, second],
        "pairs": [
            {"left": "a", "right": "a", "criterion": "interval_overlap"},
            {"left": "a", "right": "b", "criterion": "same_sign"},
        ],
        **overrides,
    }


def test_the_record_seals_verified_values_and_their_comparisons(tmp_path: Path) -> None:
    """The record holds the values, each declared comparison and a canonical digest.

    Parameters
    ----------
    tmp_path : Path
        Repository root.

    """
    record = build_comparison_record(declaration(tmp_path), tmp_path)

    sealed = dict(record)
    digest = sealed.pop("record_digest")
    assert digest == canonical_digest(RECORD_DIGEST_SCHEMA, sealed)
    assert record["accepted"] == [{"producer": "other-studio", "version": "3.1"}]
    comparisons = record["comparisons"]
    assert isinstance(comparisons, list)
    assert [(item["left"], item["right"], item["status"]) for item in comparisons] == [
        ("a", "a", "agreement"),
        ("a", "b", "non_comparable"),
    ]
    values = record["values"]
    assert isinstance(values, list)
    assert parse_comparable_value(values[0]) == sourced(tmp_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema": "other"}, "must be a scpn_qc_studio_comparable_values_v1 object"),
        ({"home": "here"}, r"\$\.home must be an object"),
        ({"pairs": {}}, r"\$\.accepted, \$\.values and \$\.pairs must be arrays"),
        ({"accepted": [{"producer": "x"}]}, r"\$\.accepted\[0\]\.version must be non-empty text"),
        ({"pairs": [3]}, r"\$\.pairs\[0\] must be an object"),
        (
            {"pairs": [{"left": "a", "right": "z", "criterion": "same_sign"}]},
            "names unknown values: z",
        ),
        (
            {"pairs": [{"left": "a", "right": "a", "criterion": "closeness"}]},
            r"\$\.pairs\[0\]\.criterion is not one of interval_overlap, same_sign",
        ),
    ],
)
def test_a_malformed_declaration_is_refused(tmp_path: Path, overrides: Json, message: str) -> None:
    """Each malformed part of a declaration is refused with its address.

    Parameters
    ----------
    tmp_path : Path
        Repository root.
    overrides : Json
        Fields to replace.
    message : str
        Expected part of the refusal.

    """
    with pytest.raises(ComparabilityError, match=message):
        build_comparison_record(declaration(tmp_path, **overrides), tmp_path)


def test_a_repeated_or_unverified_value_is_refused(tmp_path: Path) -> None:
    """Values must have distinct identifiers and equal their cited fields.

    Parameters
    ----------
    tmp_path : Path
        Repository root.

    """
    document = declaration(tmp_path)
    values = document["values"]
    assert isinstance(values, list)
    with pytest.raises(ComparabilityError, match="repeats a value_id"):
        build_comparison_record({**document, "values": [values[0], values[0]]}, tmp_path)
    changed = {**values[0], "estimate": 0.3}
    with pytest.raises(ComparabilityError, match="estimate differs"):
        build_comparison_record({**document, "values": [changed, values[1]]}, tmp_path)


def test_the_committed_record_is_current_and_compares_nothing_it_cannot() -> None:
    """The committed record equals a fresh build and keeps its honest verdicts."""
    committed = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(build_committed_claim_comparisons()))
    assert [
        (item["left"], item["status"], item["negative"]) for item in committed["comparisons"]
    ] == [
        ("d10-even-arm-leakage", "non_comparable", False),
        ("d10-asymmetry", "insufficient_evidence", True),
        ("powered-delta-8", "insufficient_evidence", False),
    ]


def test_the_comparison_cli_writes_and_checks(tmp_path: Path) -> None:
    """The committed-data CLI writes the record and then finds it current.

    Parameters
    ----------
    tmp_path : Path
        Output directory.

    """
    output = tmp_path / "comparisons.json"
    assert main(["--write", "--json-path", str(output)]) == 0
    assert main(["--check", "--json-path", str(output)]) == 0


def test_the_script_entry_point_checks_the_committed_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the tool as a script exits with the check status.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the process arguments.

    """
    monkeypatch.setattr(sys, "argv", ["studio_atlas_claim_comparison.py", "--check"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(
            str(REPO_ROOT / "tools/studio_atlas_claim_comparison.py"), run_name="__main__"
        )
    assert stopped.value.code == 0


def test_a_refused_declaration_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared number that differs from its source leaves the prior record byte for byte.

    Parameters
    ----------
    tmp_path : Path
        Holds the tampered declaration and the prior record.
    monkeypatch : pytest.MonkeyPatch
        Points the tool at the tampered declaration.

    """
    document = json.loads(VALUES_PATH.read_text(encoding="utf-8"))
    document["values"][0]["estimate"] = 0.5
    tampered = tmp_path / "values.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    prior = tmp_path / "comparisons.json"
    prior.write_bytes(DEFAULT_PATH.read_bytes())
    monkeypatch.setattr(comparison_tool, "VALUES_PATH", tampered)

    with pytest.raises(
        ComparabilityError, match="estimate differs from primary_sign_replication.leak_even"
    ):
        main(["--write", "--json-path", str(prior)])

    assert prior.read_bytes() == DEFAULT_PATH.read_bytes()
