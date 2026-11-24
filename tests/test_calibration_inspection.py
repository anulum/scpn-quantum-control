# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — calibration snapshot inspection, drift and plan judgement tests
"""Tests for dated calibration snapshots, their drift and the plans that use them."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest

import tools.studio_calibration_inspection as inspection_tool
from scpn_quantum_control.studio.calibration_inspection import (
    DECLARATION_SCHEMA,
    RECORD_DIGEST_SCHEMA,
    CalibrationInspectionError,
    CalibrationPlan,
    CalibrationSnapshot,
    FreshnessPolicy,
    PolicyRule,
    build_inspection_record,
    judge_plan,
    parse_calibration_snapshot,
    parse_freshness_policy,
    parse_submission_plan,
    read_committed_json,
    snapshot_drift,
)
from scpn_quantum_control.studio_workspace.canonical import canonical_digest
from tools.studio_calibration_inspection import (
    DECLARATION_PATH,
    DEFAULT_PATH,
    build_committed_calibration_inspection,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
Json = dict[str, object]


def snapshot_body(**overrides: object) -> Json:
    """Return a three-qubit snapshot document with two declared couplers.

    Parameters
    ----------
    **overrides : object
        Top-level or ``calibration`` fields to replace; ``calibration`` merges.

    Returns
    -------
    Json
        The document.

    """
    calibration: Json = {
        "num_qubits": 3,
        "edges": [[0, 1], [1, 2]],
        "edge_fidelity": {"0-1": 0.99, "1-2": 0.98},
        "readout_error": {"0": 0.01, "1": 0.02, "2": 0.03},
    }
    extra = overrides.pop("calibration", {})
    assert isinstance(extra, dict)
    calibration.update(extra)
    body: Json = {
        "source": "Example device",
        "date": "2026-09-23",
        "calibration_set_id": "set-a",
        "edge_fidelity_semantics": "cz gate fidelity",
        "calibration": calibration,
    }
    body.update(overrides)
    return body


def snapshot(**overrides: object) -> CalibrationSnapshot:
    """Parse :func:`snapshot_body` as a snapshot of target ``dev``.

    Parameters
    ----------
    **overrides : object
        Passed to :func:`snapshot_body`.

    Returns
    -------
    CalibrationSnapshot
        The snapshot.

    """
    body = snapshot_body(**overrides)
    name = f"data/{body['calibration_set_id']}.json"
    return parse_calibration_snapshot(body, target_id="dev", path=name, sha256="0" * 64)


def plan(
    *,
    target_id: str = "dev",
    block_on: str = "2026-09-23",
    calibration_set_id: str | None = "set-a",
) -> CalibrationPlan:
    """Return a plan for experiment ``exp``.

    Parameters
    ----------
    target_id : str
        Target the plan runs on.
    block_on : str
        Date of the block.
    calibration_set_id : str | None
        The set the plan names.

    Returns
    -------
    CalibrationPlan
        The plan.

    """
    return CalibrationPlan(
        plan_id="p1",
        target_id=target_id,
        path="data/p1.json",
        sha256="1" * 64,
        experiment_type="exp",
        block_on=block_on,
        created_at=f"{block_on}T10:00:00Z",
        calibration_set_id=calibration_set_id,
    )


def policy(*rules: PolicyRule) -> FreshnessPolicy:
    """Return a policy for experiment ``exp`` with the given rules.

    Parameters
    ----------
    *rules : PolicyRule
        The rules.

    Returns
    -------
    FreshnessPolicy
        The policy.

    """
    return FreshnessPolicy("exp", rules, (("path", "p"), ("sha256", "s"), ("section", "x")))


def test_calibration_snapshot_diff_01() -> None:
    """A missing readout or coupler reading stays unknown, never a perfect value."""
    partial = snapshot(
        calibration={"readout_error": {"0": 0.01, "2": 0.03}, "edge_fidelity": {"0-1": 0.99}}
    )

    assert dict(partial.readout_error) == {0: 0.01, 1: None, 2: 0.03}
    assert dict(partial.cz_fidelity) == {(0, 1): 0.99, (1, 2): None}
    drift = snapshot_drift(snapshot(), partial)
    assert [(item.element, item.change, item.after) for item in drift.changes] == [
        ("c1-2", "became_unknown", None),
        ("q1", "became_unknown", None),
    ]


def test_calibration_snapshot_diff_02() -> None:
    """A newer calibration of the target blocks a plan that still names the older set."""
    old = snapshot()
    new = snapshot(date="2026-09-24", calibration_set_id="set-b")
    later = plan(block_on="2026-09-24")

    judged = judge_plan(later, [old, new], policy(PolicyRule("request_names_calibration_set")), [])

    assert judged.status == "blocked"
    assert judged.reasons == ("calibration set set-a was superseded by set-b on 2026-09-24",)
    other = judge_plan(plan(target_id="dev2"), [old], policy(), [])
    assert other.reasons == ("calibration set set-a belongs to dev, the plan runs on dev2",)


def test_calibration_snapshot_diff_03() -> None:
    """Age at the bound is fresh, one day over is stale, and a later snapshot is refused."""
    rule = policy(PolicyRule("max_age_days", max_age_days=2))
    old = snapshot()

    at_bound = judge_plan(plan(block_on="2026-09-25"), [old], rule, [])
    over = judge_plan(plan(block_on="2026-09-26"), [old], rule, [])
    before = judge_plan(plan(block_on="2026-09-22"), [old], rule, [])

    assert (at_bound.status, at_bound.reasons) == ("admitted", ("max_age_days holds",))
    assert over.reasons == ("max_age_days: 3 days old, bound 2",)
    assert before.reasons == ("max_age_days: snapshot 2026-09-23 is dated after block 2026-09-22",)


def test_calibration_snapshot_diff_04() -> None:
    """A historical plan is judged as of its own block, so later calibrations never rewrite it."""
    old = snapshot()
    new = snapshot(date="2026-09-24", calibration_set_id="set-b")

    judged = judge_plan(
        plan(), [old, new], policy(PolicyRule("snapshot_date_equals_block_date")), []
    )

    assert judged.status == "admitted"
    assert judged.snapshot_path == "data/set-a.json"


def test_calibration_snapshot_diff_05() -> None:
    """A blocked plan keeps its named set and is never moved to another snapshot or device."""
    current = snapshot(date="2026-09-24", calibration_set_id="set-b")

    judged = judge_plan(plan(), [current], policy(), [])
    unnamed = judge_plan(plan(calibration_set_id=None), [current], None, [])

    assert (judged.status, judged.calibration_set_id, judged.snapshot_path) == (
        "blocked",
        "set-a",
        None,
    )
    assert judged.reasons == ("no admitted snapshot holds calibration set set-a",)
    assert unnamed.reasons == (
        "no freshness policy is declared for exp",
        "the plan names no calibration set",
    )


@pytest.mark.parametrize(
    ("rule", "named", "earlier", "reason"),
    [
        (PolicyRule("request_names_calibration_set"), None, [], "no set is named"),
        (
            PolicyRule("distinct_calibration_set", excluded_calibration_set_ids=("set-a",)),
            "set-a",
            [],
            "set-a is excluded by the cited source",
        ),
        (PolicyRule("distinct_calibration_set"), "set-a", ["set-a"], "used by an earlier plan"),
        (PolicyRule("required_couplers", couplers=((0, 2),)), "set-a", [], "0-2 not declared"),
        (PolicyRule("snapshot_date_equals_block_date"), "set-z", [], "cannot be judged"),
    ],
)
def test_each_unmet_rule_names_itself(
    rule: PolicyRule, named: str | None, earlier: list[str], reason: str
) -> None:
    """Every rule that fails is reported with its own name.

    Parameters
    ----------
    rule : PolicyRule
        The single rule.
    named : str | None
        The set the plan names.
    earlier : list[str]
        Sets of earlier plans.
    reason : str
        Expected part of the reason.

    """
    judged = judge_plan(plan(calibration_set_id=named), [snapshot()], policy(rule), earlier)

    assert judged.status == "blocked"
    assert any(rule.kind in item and reason in item for item in judged.reasons)


def test_a_date_mismatch_and_a_bound_without_value_are_reported() -> None:
    """A snapshot from another day is unmet, and a max-age rule without its bound is refused."""
    judged = judge_plan(
        plan(block_on="2026-09-24"),
        [snapshot()],
        policy(PolicyRule("snapshot_date_equals_block_date")),
        [],
    )
    assert judged.reasons == (
        "snapshot_date_equals_block_date: snapshot 2026-09-23, block 2026-09-24",
    )
    with pytest.raises(CalibrationInspectionError, match="needs its bound"):
        judge_plan(plan(), [snapshot()], policy(PolicyRule("max_age_days")), [])


def test_every_kind_of_drift_is_exact() -> None:
    """Changed, appeared, disappeared and became-known readings differ; an equal reading does not."""
    before = snapshot(calibration={"readout_error": {"0": 0.01, "2": 0.03}})
    after = snapshot(
        date="2026-09-24",
        calibration_set_id="set-b",
        calibration={
            "num_qubits": 4,
            "edges": [[0, 1], [2, 3]],
            "edge_fidelity": {"0-1": 0.97},
            "readout_error": {"0": 0.01, "1": 0.02, "2": 0.04},
        },
    )

    drift = snapshot_drift(before, after)

    assert drift.calibration_set_changed is True
    assert [(item.element, item.change) for item in drift.changes] == [
        ("c0-1", "changed"),
        ("c1-2", "disappeared"),
        ("c2-3", "appeared"),
        ("q1", "became_known"),
        ("q2", "changed"),
        ("q3", "appeared"),
    ]
    assert drift.unchanged == 1
    with pytest.raises(CalibrationInspectionError, match="earlier snapshot first"):
        snapshot_drift(after, before)
    other = parse_calibration_snapshot(snapshot_body(), target_id="x", path="p", sha256="0" * 64)
    with pytest.raises(CalibrationInspectionError, match="compares one target"):
        snapshot_drift(before, other)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source": ""}, "source must be non-empty text"),
        ({"date": "23.09.2026"}, "must be a YYYY-MM-DD date"),
        ({"date": "2026-02-30"}, "must be a calendar date"),
        ({"calibration": {"num_qubits": 0}}, "num_qubits must be positive"),
        ({"calibration": {"num_qubits": True}}, "non-negative integer"),
        ({"calibration": {"edges": [[0, 1, 2]]}}, r"edges\[0\] must be a pair"),
        ({"calibration": {"edges": [[1, 1]]}}, "two different qubits"),
        ({"calibration": {"edges": [[0, 7]]}}, "names an absent qubit"),
        ({"calibration": {"edges": [[0, 1], [0, 1]]}}, r"repeats \(0, 1\)"),
        ({"calibration": {"edges": "0-1"}}, "edges must be an array"),
        ({"calibration": {"edge_fidelity": {"0-2": 0.9}}}, "undeclared coupler 0-2"),
        ({"calibration": {"edge_fidelity": {"0-1": 1.5}}}, r"must lie in \[0, 1\]"),
        ({"calibration": {"readout_error": {"x": 0.1}}}, "absent qubit x"),
        ({"calibration": {"readout_error": {"5": 0.1}}}, "absent qubit 5"),
        ({"calibration": {"readout_error": {"0": float("nan")}}}, "finite number"),
        ({"calibration": {"readout_error": []}}, "readout_error must be an object"),
    ],
)
def test_a_malformed_snapshot_is_refused(overrides: Json, message: str) -> None:
    """Every malformed snapshot field is refused with its address.

    Parameters
    ----------
    overrides : Json
        Fields to replace.
    message : str
        Expected part of the refusal.

    """
    with pytest.raises(CalibrationInspectionError, match=message):
        snapshot(**overrides)


def write(root: Path, name: str, document: object) -> tuple[str, str]:
    """Write a JSON file below ``root`` and return its path and sha256.

    Parameters
    ----------
    root : Path
        Repository root.
    name : str
        Repository-relative path.
    document : object
        JSON content.

    Returns
    -------
    tuple[str, str]
        The path and digest.

    """
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document), encoding="utf-8")
    return name, hashlib.sha256(target.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("path", "digest", "message"),
    [
        ("/abs.json", "0" * 64, "safe relative path"),
        ("../up.json", "0" * 64, "safe relative path"),
        ("a\\b.json", "0" * 64, "safe relative path"),
        ("data/x.json", "ABC", "lowercase digest"),
        ("data/absent.json", "0" * 64, "cannot be read"),
        ("data/x.json", "0" * 64, "does not match its recorded sha256"),
    ],
)
def test_an_unverifiable_file_is_refused(
    tmp_path: Path, path: str, digest: str, message: str
) -> None:
    """Unsafe paths, malformed or wrong digests and absent files are refused.

    Parameters
    ----------
    tmp_path : Path
        Repository root.
    path : str
        Cited path.
    digest : str
        Cited digest.
    message : str
        Expected part of the refusal.

    """
    write(tmp_path, "data/x.json", {})
    with pytest.raises(CalibrationInspectionError, match=message):
        read_committed_json(path, digest, tmp_path, "$.f")


def test_a_non_json_file_is_refused(tmp_path: Path) -> None:
    """A cited file must be JSON.

    Parameters
    ----------
    tmp_path : Path
        Repository root.

    """
    (tmp_path / "x.json").write_text("not json", encoding="utf-8")
    digest = hashlib.sha256(b"not json").hexdigest()
    with pytest.raises(CalibrationInspectionError, match="is not JSON"):
        read_committed_json("x.json", digest, tmp_path, "$.f")


def policy_payload(root: Path, *rules: Json, **citation: object) -> Json:
    """Return a policy payload citing a document below ``root``.

    Parameters
    ----------
    root : Path
        Repository root.
    *rules : Json
        Rule payloads.
    **citation : object
        Citation fields to replace.

    Returns
    -------
    Json
        The payload.

    """
    document = root / "docs/policy.md"
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text("# Policy\n", encoding="utf-8")
    cited: Json = {
        "path": "docs/policy.md",
        "sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
        "section": "Admission",
        **citation,
    }
    return {"experiment_type": "exp", "citation": cited, "rules": list(rules)}


def test_a_policy_reads_its_excluded_sets_from_the_cited_source(tmp_path: Path) -> None:
    """Excluded sets come from the cited JSON field; every rule kind parses.

    Parameters
    ----------
    tmp_path : Path
        Repository root.

    """
    path, digest = write(tmp_path, "data/design.json", {"source": {"excluded": ["x", "y"]}})
    payload = policy_payload(
        tmp_path,
        {"kind": "snapshot_date_equals_block_date"},
        {
            "kind": "distinct_calibration_set",
            "excluded_from": {"path": path, "sha256": digest, "pointer": "source.excluded"},
        },
        {"kind": "required_couplers", "couplers": [[2, 7]]},
        {"kind": "request_names_calibration_set"},
        {"kind": "max_age_days", "max_age_days": 0},
    )

    parsed = parse_freshness_policy(payload, "$.p", tmp_path)

    assert [rule.kind for rule in parsed.rules] == [
        "snapshot_date_equals_block_date",
        "distinct_calibration_set",
        "required_couplers",
        "request_names_calibration_set",
        "max_age_days",
    ]
    assert parsed.rules[1].excluded_calibration_set_ids == ("x", "y")
    assert parsed.rules[2].couplers == ((2, 7),)
    assert parsed.rules[4].max_age_days == 0
    assert dict(parsed.citation)["section"] == "Admission"


@pytest.mark.parametrize(
    ("rules", "citation", "message"),
    [
        ([{"kind": "lifetime"}], {}, "kind is not one of"),
        ([], {}, "rules must not be empty"),
        ([{"kind": "required_couplers", "couplers": []}], {}, "couplers must not be empty"),
        ([{"kind": "max_age_days", "max_age_days": -1}], {}, "non-negative integer"),
        ([{"kind": "request_names_calibration_set"}], {"path": "../x.md"}, "safe relative path"),
        (
            [{"kind": "request_names_calibration_set"}],
            {"path": "docs/absent.md"},
            "cannot be read",
        ),
        (
            [{"kind": "request_names_calibration_set"}],
            {"sha256": "0" * 64},
            "does not match its sha256",
        ),
    ],
)
def test_a_malformed_policy_is_refused(
    tmp_path: Path, rules: list[Json], citation: Json, message: str
) -> None:
    """Unknown or empty rules and unverifiable citations are refused.

    Parameters
    ----------
    tmp_path : Path
        Repository root.
    rules : list[Json]
        Rule payloads.
    citation : Json
        Citation fields to replace.
    message : str
        Expected part of the refusal.

    """
    with pytest.raises(CalibrationInspectionError, match=message):
        parse_freshness_policy(policy_payload(tmp_path, *rules, **citation), "$.p", tmp_path)


@pytest.mark.parametrize(
    ("document", "pointer", "message"),
    [
        ({"source": {"excluded": ["x"]}}, "source.missing", "does not resolve"),
        ({"source": ["x"]}, "source.excluded", "does not resolve"),
        ({"source": {"excluded": [""]}}, "source.excluded", "must be non-empty text"),
    ],
)
def test_an_unresolved_exclusion_is_refused(
    tmp_path: Path, document: Json, pointer: str, message: str
) -> None:
    """The excluded sets must resolve to text in the cited JSON.

    Parameters
    ----------
    tmp_path : Path
        Repository root.
    document : Json
        Cited JSON.
    pointer : str
        Dotted pointer.
    message : str
        Expected part of the refusal.

    """
    path, digest = write(tmp_path, "data/design.json", document)
    rule: Json = {
        "kind": "distinct_calibration_set",
        "excluded_from": {"path": path, "sha256": digest, "pointer": pointer},
    }
    with pytest.raises(CalibrationInspectionError, match=message):
        parse_freshness_policy(policy_payload(tmp_path, rule), "$.p", tmp_path)


def test_a_submission_becomes_a_plan() -> None:
    """A submission's campaign, date, creation time and named set make the plan."""
    document = {
        "campaign": "exp",
        "date": "2026-09-23",
        "created_at": "2026-09-23T19:27:28.966Z",
        "calibration_set_id": None,
    }

    parsed = parse_submission_plan(
        document, target_id="dev", path="data/run_1.json", sha256="2" * 64
    )

    assert (parsed.plan_id, parsed.experiment_type, parsed.calibration_set_id) == (
        "run_1",
        "exp",
        None,
    )


def distinct_rule(root: Path) -> Json:
    """Return a distinct-set rule whose excluded sets come from a committed file.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    Json
        The rule, excluding ``set-z``.

    """
    path, digest = write(root, "data/excluded.json", {"ids": ["set-z"]})
    return {
        "kind": "distinct_calibration_set",
        "excluded_from": {"path": path, "sha256": digest, "pointer": "ids"},
    }


def declaration(root: Path, **overrides: object) -> Json:
    """Write two snapshots and two plans of one target and return their declaration.

    Parameters
    ----------
    root : Path
        Repository root.
    **overrides : object
        Top-level fields to replace.

    Returns
    -------
    Json
        The declaration.

    """
    first = write(root, "data/cal_a.json", snapshot_body())
    second = write(
        root, "data/cal_b.json", snapshot_body(date="2026-09-24", calibration_set_id="set-b")
    )
    runs = [
        write(
            root,
            f"data/run_{name}.json",
            {
                "campaign": "exp",
                "quantum_computer": "device",
                "date": day,
                "created_at": f"{day}T10:00:00Z",
                "calibration_set_id": calibration,
            },
        )
        for name, day, calibration in (("1", "2026-09-23", "set-a"), ("2", "2026-09-24", "set-a"))
    ]
    entry = [
        {"path": path, "sha256": digest, "target_id": "dev"} for path, digest in (first, second)
    ]
    body: Json = {
        "schema": DECLARATION_SCHEMA,
        "targets": [
            {
                "target_id": "dev",
                "snapshot_source": "Example device",
                "plan_quantum_computer": "device",
            }
        ],
        "policies": [policy_payload(root, distinct_rule(root))],
        "snapshots": entry,
        "plans": [{"path": path, "sha256": digest, "target_id": "dev"} for path, digest in runs],
    }
    body.update(overrides)
    return body


def test_the_record_maps_drifts_and_judges_every_plan(tmp_path: Path) -> None:
    """Snapshots, drift, references and plan verdicts are sealed with a canonical digest.

    Parameters
    ----------
    tmp_path : Path
        Repository root.

    """
    record = build_inspection_record(declaration(tmp_path), tmp_path)

    sealed = dict(record)
    digest = sealed.pop("record_digest")
    assert digest == canonical_digest(RECORD_DIGEST_SCHEMA, sealed)
    snapshots = record["snapshots"]
    plans = record["plans"]
    drifts = record["drifts"]
    assert isinstance(snapshots, list) and isinstance(plans, list) and isinstance(drifts, list)
    assert [
        (item["calibration_set_id"], item["superseded_by"], item["referenced_by"])
        for item in snapshots
    ] == [
        ("set-a", "set-b", ["run_1", "run_2"]),
        ("set-b", None, []),
    ]
    assert [(item["plan_id"], item["status"]) for item in plans] == [
        ("run_1", "admitted"),
        ("run_2", "blocked"),
    ]
    assert plans[1]["reasons"] == [
        "calibration set set-a was superseded by set-b on 2026-09-24",
        "distinct_calibration_set: set-a was used by an earlier plan",
    ]
    assert drifts[0]["unchanged"] == 5 and drifts[0]["calibration_set_changed"] is True


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema": "other"}, "must be a scpn_qc_studio_calibration_inspection_declaration_v1"),
        (
            {
                "targets": [
                    {"target_id": "dev", "snapshot_source": "s", "plan_quantum_computer": "d"}
                ]
                * 2
            },
            "targets repeats dev",
        ),
        (
            {"snapshots": [{"path": "data/cal_a.json", "sha256": "0" * 64, "target_id": "other"}]},
            "undeclared target",
        ),
        (
            {
                "targets": [
                    {
                        "target_id": "dev",
                        "snapshot_source": "Elsewhere",
                        "plan_quantum_computer": "device",
                    }
                ]
            },
            "is from Example device, not Elsewhere",
        ),
        (
            {
                "targets": [
                    {
                        "target_id": "dev",
                        "snapshot_source": "Example device",
                        "plan_quantum_computer": "other",
                    }
                ]
            },
            "ran on 'device', not other",
        ),
    ],
)
def test_a_malformed_declaration_is_refused(tmp_path: Path, change: Json, message: str) -> None:
    """Schema, targets and file ownership are checked before anything is judged.

    Parameters
    ----------
    tmp_path : Path
        Repository root.
    change : Json
        Top-level fields to replace.
    message : str
        Expected part of the refusal.

    """
    with pytest.raises(CalibrationInspectionError, match=message):
        build_inspection_record(declaration(tmp_path, **change), tmp_path)


def test_a_plan_naming_no_set_is_blocked_in_the_record(tmp_path: Path) -> None:
    """A submission without a calibration set is judged and blocked, never assigned one.

    Parameters
    ----------
    tmp_path : Path
        Repository root.

    """
    run = {
        "campaign": "exp",
        "quantum_computer": "device",
        "date": "2026-09-25",
        "created_at": "2026-09-25T10:00:00Z",
        "calibration_set_id": None,
    }
    path, digest = write(tmp_path, "data/run_3.json", run)
    document = declaration(tmp_path)
    plans = document["plans"]
    assert isinstance(plans, list)
    entry = {"path": path, "sha256": digest, "target_id": "dev"}

    record = build_inspection_record({**document, "plans": [*plans, entry]}, tmp_path)

    judged = record["plans"]
    assert isinstance(judged, list)
    assert (judged[2]["plan_id"], judged[2]["status"], judged[2]["calibration_set_id"]) == (
        "run_3",
        "blocked",
        None,
    )
    assert judged[2]["reasons"] == ["the plan names no calibration set"]


def test_repeated_policies_sets_and_dates_are_refused(tmp_path: Path) -> None:
    """Two policies of one type, a repeated set and two snapshots on one date are refused.

    Parameters
    ----------
    tmp_path : Path
        Repository root.

    """
    document = declaration(tmp_path)
    policies = document["policies"]
    snapshots = document["snapshots"]
    assert isinstance(policies, list) and isinstance(snapshots, list)
    with pytest.raises(CalibrationInspectionError, match="policies repeats exp"):
        build_inspection_record({**document, "policies": policies * 2}, tmp_path)
    with pytest.raises(CalibrationInspectionError, match="repeats calibration set set-a"):
        build_inspection_record({**document, "snapshots": [snapshots[0], snapshots[0]]}, tmp_path)
    same_day = write(tmp_path, "data/cal_c.json", snapshot_body(calibration_set_id="set-c"))
    entry = {"path": same_day[0], "sha256": same_day[1], "target_id": "dev"}
    with pytest.raises(CalibrationInspectionError, match="their order is not recorded"):
        build_inspection_record({**document, "snapshots": [snapshots[0], entry]}, tmp_path)


def test_the_committed_record_is_current_and_admits_every_real_epoch() -> None:
    """The committed record equals a fresh build; each real epoch ran on its own new calibration."""
    committed = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(build_committed_calibration_inspection()))
    plans = committed["plans"]
    assert len(plans) == 12
    assert {item["status"] for item in plans} == {"admitted"}
    assert all(len(item["reasons"]) == 4 for item in plans)
    snapshots = committed["snapshots"]
    assert [item["referenced_by"] for item in snapshots] == [[item["plan_id"]] for item in plans]
    assert snapshots[-1]["superseded_by"] is None
    assert all(len(item["qubits"]) == 20 and len(item["couplers"]) == 30 for item in snapshots)
    assert [item["calibration_set_changed"] for item in committed["drifts"]] == [True] * 11


def test_the_inspection_cli_writes_checks_and_refuses_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI writes and checks the record; a tampered declaration leaves the prior file intact.

    Parameters
    ----------
    tmp_path : Path
        Output directory.
    monkeypatch : pytest.MonkeyPatch
        Points the tool at a tampered declaration.

    """
    output = tmp_path / "record.json"
    assert main(["--write", "--json-path", str(output)]) == 0
    assert main(["--check", "--json-path", str(output)]) == 0
    document = json.loads(DECLARATION_PATH.read_text(encoding="utf-8"))
    document["snapshots"][0]["sha256"] = "0" * 64
    tampered = tmp_path / "declaration.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    before = output.read_bytes()
    monkeypatch.setattr(inspection_tool, "DECLARATION_PATH", tampered)
    with pytest.raises(CalibrationInspectionError, match="does not match its recorded sha256"):
        main(["--write", "--json-path", str(output)])
    assert output.read_bytes() == before


def test_the_script_entry_point_checks_the_committed_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the tool as a script exits with the check status.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the process arguments.

    """
    monkeypatch.setattr(sys, "argv", ["studio_calibration_inspection.py", "--check"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(
            str(REPO_ROOT / "tools/studio_calibration_inspection.py"), run_name="__main__"
        )
    assert stopped.value.code == 0
