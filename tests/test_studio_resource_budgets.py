# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — declared resource budgets, strict verdicts and bundle measurement tests
"""Tests for the declared Studio resource budgets and their strict verdicts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scpn_quantum_control.atlas import atlas_snapshot_digest, parse_atlas_snapshot
from tools.studio_resource_budgets import (
    BUDGETS_PATH,
    BUDGETS_SCHEMA,
    FROZEN_TIER_DIGESTS,
    PIN_SCHEMA,
    TIERS,
    Budget,
    BudgetError,
    initial_javascript_bytes,
    judge_budget,
    judge_budgets,
    load_budgets,
    parse_budgets,
    tier_node_id,
    tier_pin_text,
    tier_snapshot_document,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = REPO_ROOT / "data/studio/resource_budget_verdict_cases_20261123.json"
Json = dict[str, object]


def declaration(*budgets: object) -> Json:
    """Return a budget declaration.

    Parameters
    ----------
    *budgets : object
        Budget payloads.

    Returns
    -------
    Json
        The declaration.

    """
    return {"schema": BUDGETS_SCHEMA, "note": "n", "budgets": list(budgets)}


def lt(target: object = 500) -> Json:
    """Return a ``<`` budget payload.

    Parameters
    ----------
    target : object
        The target.

    Returns
    -------
    Json
        The payload.

    """
    return {"id": "lt", "metric": "m", "comparator": "<", "target": target, "unit": "ms"}


def test_every_shared_case_has_the_declared_verdict() -> None:
    """The shared cases, which the browser rule also runs, give the same verdicts here."""
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        budget = parse_budgets(declaration(case["budget"]))[0]
        assert judge_budget(budget, case["measured"]).status == case["status"], case


def test_workbench_resource_budgets_05() -> None:
    """A miss is listed as a failure; an absent measurement is unmeasured, never a pass."""
    budgets = parse_budgets(
        declaration(
            lt(),
            {"id": "rec", "metric": "m", "comparator": "record", "target": None, "unit": "b"},
            {"id": "ge", "metric": "m", "comparator": ">=", "target": 3, "unit": "n"},
        )
    )

    record = judge_budgets(budgets, {"lt": 500.0, "rec": 7})

    assert record["missed"] == ["lt"]
    assert record["unmeasured"] == ["ge"]
    verdicts = record["verdicts"]
    assert isinstance(verdicts, list)
    assert [(item["id"], item["status"], item["measured"]) for item in verdicts] == [
        ("lt", "fail", 500.0),
        ("rec", "recorded", 7.0),
        ("ge", "unmeasured", None),
    ]
    with pytest.raises(BudgetError, match="undeclared budgets: other"):
        judge_budgets(budgets, {"other": 1})


@pytest.mark.parametrize("measured", [float("nan"), float("inf"), True, "1"])
def test_a_measurement_that_is_not_a_finite_number_is_refused(measured: object) -> None:
    """Non-finite, boolean or text measurements cannot be judged.

    Parameters
    ----------
    measured : object
        The measurement.

    """
    budget = Budget("lt", "m", "<", 500.0, "ms")
    with pytest.raises(BudgetError, match="must be a finite number"):
        judge_budget(budget, measured)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"schema": "other"}, "must be a scpn_qc_studio_resource_budgets_v1 object"),
        (declaration(), "budgets must be a non-empty array"),
        (declaration("x"), r"budgets\[0\] must be an object"),
        (declaration({**lt(), "comparator": "~"}), "comparator is not one of"),
        (declaration({**lt(), "comparator": "record"}), "record-only budget has no target"),
        (declaration(lt(None)), "record-only budget has no target"),
        (declaration({**lt(), "unit": ""}), "non-empty id, metric and unit"),
        (declaration(lt("500")), "target must be a finite number"),
        (declaration(lt(), lt()), "repeats an id"),
    ],
)
def test_a_malformed_declaration_is_refused(document: object, message: str) -> None:
    """Every malformed part of a declaration is refused.

    Parameters
    ----------
    document : object
        The declaration.
    message : str
        Expected part of the refusal.

    """
    with pytest.raises(BudgetError, match=message):
        parse_budgets(document)


def test_the_committed_declaration_keeps_the_release_targets() -> None:
    """The committed budgets carry the release targets unchanged."""
    budgets = {budget.budget_id: budget for budget in load_budgets()}

    assert BUDGETS_PATH.is_file()
    assert (budgets["graph_query_p95"].comparator, budgets["graph_query_p95"].target) == (
        "<",
        500.0,
    )
    assert (budgets["worker_cancel_ack"].comparator, budgets["worker_cancel_ack"].target) == (
        "<",
        250.0,
    )
    assert budgets["initial_javascript"].target == 200 * 1024
    assert budgets["trajectory_points"].target == 100_000
    assert budgets["retained_memory"].comparator == "record"


def write_bundle(root: Path, html: str, files: dict[str, bytes]) -> Path:
    """Write a built-bundle directory.

    Parameters
    ----------
    root : Path
        Parent directory.
    html : str
        The entry page.
    files : dict[str, bytes]
        Assets by bundle-relative path.

    Returns
    -------
    Path
        The bundle directory.

    """
    dist = root / "dist"
    dist.mkdir()
    for name, data in files.items():
        (dist / name).parent.mkdir(parents=True, exist_ok=True)
        (dist / name).write_bytes(data)
    (dist / "index.html").write_text(html, encoding="utf-8")
    return dist


def test_initial_javascript_is_the_gzip_size_of_the_entry_and_its_preloads(tmp_path: Path) -> None:
    """Entry scripts and module preloads count once each; lazy chunks do not count.

    Parameters
    ----------
    tmp_path : Path
        Scratch directory.

    """
    entry, shared, lazy = b"import './shared.js';" * 50, b"export const a = 1;" * 80, b"x" * 9000
    html = (
        '<script type="module" crossorigin src="/assets/entry.js"></script>'
        '<link rel="modulepreload" crossorigin href="./assets/shared.js">'
        '<link rel="modulepreload" href="/assets/shared.js">'
    )
    dist = write_bundle(
        tmp_path,
        html,
        {"assets/entry.js": entry, "assets/shared.js": shared, "assets/lazy.js": lazy},
    )

    expected = sum(len(gzip.compress(data, compresslevel=9, mtime=0)) for data in (entry, shared))
    assert initial_javascript_bytes(dist) == expected


@pytest.mark.parametrize(
    ("html", "message"),
    [
        ("<html></html>", "names no script"),
        ('<script src="../outside.js"></script>', "outside the bundle"),
        ('<script src="https://cdn.example/x.js"></script>', "outside the bundle"),
        ('<script src="//cdn.example/x.js"></script>', "outside the bundle"),
        ('<script src="/assets/absent.js"></script>', "is not in the bundle"),
    ],
)
def test_an_unmeasurable_bundle_is_refused(tmp_path: Path, html: str, message: str) -> None:
    """An entry page without scripts, or naming scripts outside or absent from the bundle, is refused.

    Parameters
    ----------
    tmp_path : Path
        Scratch directory.
    html : str
        The entry page.
    message : str
        Expected part of the refusal.

    """
    dist = write_bundle(tmp_path, html, {})
    with pytest.raises(BudgetError, match=message):
        initial_javascript_bytes(dist)


@pytest.mark.parametrize(("tier", "frozen"), list(zip(TIERS, FROZEN_TIER_DIGESTS, strict=True)))
def test_each_tier_matches_the_frozen_browser_fixture(tier: tuple[int, int], frozen: str) -> None:
    """The Python tier has the digest the browser fixture pins, so both languages use one fixture.

    Parameters
    ----------
    tier : tuple[int, int]
        Nodes and relations.
    frozen : str
        The pinned snapshot digest.

    """
    nodes, relations = tier
    document = tier_snapshot_document(nodes, relations)

    assert atlas_snapshot_digest(parse_atlas_snapshot(document)) == frozen
    edges = document["edges"]
    assert isinstance(edges, list) and len(edges) == relations
    assert len({(edge["source"], edge["target"]) for edge in edges}) == relations


def test_a_tier_pin_carries_its_verified_digest() -> None:
    """The pin names the pin schema and the digest of its own snapshot."""
    pin = json.loads(tier_pin_text(100, 300))

    assert pin["schema"] == PIN_SCHEMA
    assert pin["snapshot_digest"] == FROZEN_TIER_DIGESTS[0]
    assert pin["snapshot"]["nodes"][0]["id"] == tier_node_id(0) == "definition:tier/n00000"


@pytest.mark.parametrize(("nodes", "relations"), [(10, 15), (3, 9)])
def test_an_impossible_tier_is_refused(nodes: int, relations: int) -> None:
    """Relations must be a multiple of the nodes, below their square.

    Parameters
    ----------
    nodes : int
        Node count.
    relations : int
        Relation count.

    """
    with pytest.raises(BudgetError, match="multiple of nodes"):
        tier_snapshot_document(nodes, relations)
