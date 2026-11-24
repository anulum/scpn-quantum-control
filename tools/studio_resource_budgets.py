# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — declared Studio resource budgets and their strict verdicts
"""Read the declared Studio resource budgets and judge measurements against them.

The budgets (``data/studio/resource_budgets_20261123.json``) are targets, not
results. A measurement either meets its target under the declared comparator
(``<``, ``<=`` or ``>=``) or it does not: there is no tolerance, no retry and no
relaxed second threshold. A missing measurement is *unmeasured*, never a pass,
and a record-only budget is *recorded* without a verdict.

:func:`initial_javascript_bytes` measures one budget from a built bundle: the
gzip-compressed size of every script the entry page loads before any lazy chunk.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
BUDGETS_PATH: Final[Path] = ROOT / "data/studio/resource_budgets_20261123.json"
BUDGETS_SCHEMA: Final = "scpn_qc_studio_resource_budgets_v1"
RECORD_SCHEMA: Final = "scpn_qc_studio_resource_budget_record_v1"

Comparator = Literal["<", "<=", ">=", "record"]
COMPARATORS: Final[tuple[Comparator, ...]] = ("<", "<=", ">=", "record")
Status = Literal["pass", "fail", "unmeasured", "recorded"]

_ENTRY_SCRIPT = re.compile(r'<script[^>]*\bsrc="([^"]+\.js)"')
_PRELOAD = re.compile(r'<link[^>]*\brel="modulepreload"[^>]*\bhref="([^"]+\.js)"')


class BudgetError(ValueError):
    """A budget declaration, a measurement or a built bundle is malformed."""


@dataclass(frozen=True, slots=True)
class Budget:
    """One declared budget.

    Attributes
    ----------
    budget_id
        Stable identifier.
    metric
        What is measured, in words.
    comparator
        How the measurement meets the target, or ``record`` for no verdict.
    target
        The target, or ``None`` for a record-only budget.
    unit
        Unit of the target and the measurement.

    """

    budget_id: str
    metric: str
    comparator: Comparator
    target: float | None
    unit: str


@dataclass(frozen=True, slots=True)
class Verdict:
    """The verdict on one budget.

    Attributes
    ----------
    budget
        The budget.
    measured
        The measurement, or ``None`` when it was not measured.
    status
        ``pass``, ``fail``, ``unmeasured`` or ``recorded``.

    """

    budget: Budget
    measured: float | None
    status: Status


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise BudgetError(f"{where} must be a finite number")
    return float(value)


def _budget(payload: object, where: str) -> Budget:
    if not isinstance(payload, Mapping):
        raise BudgetError(f"{where} must be an object")
    comparator = next((item for item in COMPARATORS if item == payload.get("comparator")), None)
    if comparator is None:
        raise BudgetError(f"{where}.comparator is not one of {', '.join(COMPARATORS)}")
    target = payload.get("target")
    if (comparator == "record") != (target is None):
        raise BudgetError(f"{where}: a record-only budget has no target, and every other has one")
    texts = [payload.get(key) for key in ("id", "metric", "unit")]
    if not all(isinstance(text, str) and text for text in texts):
        raise BudgetError(f"{where} needs a non-empty id, metric and unit")
    return Budget(
        str(texts[0]),
        str(texts[1]),
        comparator,
        None if target is None else _number(target, f"{where}.target"),
        str(texts[2]),
    )


def parse_budgets(document: object) -> tuple[Budget, ...]:
    """Parse a budget declaration.

    Parameters
    ----------
    document
        ``{schema, note, budgets}`` under :data:`BUDGETS_SCHEMA`.

    Returns
    -------
    tuple[Budget, ...]
        The budgets in declared order.

    Raises
    ------
    BudgetError
        The declaration is malformed or repeats an identifier.

    """
    if not isinstance(document, Mapping) or document.get("schema") != BUDGETS_SCHEMA:
        raise BudgetError(f"the declaration must be a {BUDGETS_SCHEMA} object")
    items = document.get("budgets")
    if not isinstance(items, list) or not items:
        raise BudgetError("$.budgets must be a non-empty array")
    budgets = tuple(_budget(item, f"$.budgets[{index}]") for index, item in enumerate(items))
    if len({budget.budget_id for budget in budgets}) != len(budgets):
        raise BudgetError("$.budgets repeats an id")
    return budgets


def load_budgets(path: Path = BUDGETS_PATH) -> tuple[Budget, ...]:
    """Read and parse the committed budget declaration.

    Parameters
    ----------
    path
        The declaration file.

    Returns
    -------
    tuple[Budget, ...]
        The budgets.

    """
    return parse_budgets(json.loads(path.read_text(encoding="utf-8")))


def judge_budget(budget: Budget, measured: object) -> Verdict:
    """Judge one measurement strictly against its budget.

    Parameters
    ----------
    budget
        The budget.
    measured
        A finite measurement, or ``None`` when it was not measured.

    Returns
    -------
    Verdict
        ``unmeasured`` without a measurement, ``recorded`` for a record-only
        budget, otherwise ``pass`` exactly when the comparator holds.

    Raises
    ------
    BudgetError
        The measurement is not a finite number.

    """
    if measured is None:
        return Verdict(budget, None, "unmeasured")
    value = _number(measured, budget.budget_id)
    target = budget.target
    if budget.comparator == "record" or target is None:
        return Verdict(budget, value, "recorded")
    holds = {
        "<": value < target,
        "<=": value <= target,
        ">=": value >= target,
    }[budget.comparator]
    return Verdict(budget, value, "pass" if holds else "fail")


def judge_budgets(
    budgets: Sequence[Budget], measurements: Mapping[str, object]
) -> dict[str, object]:
    """Judge every declared budget and list the missed ones.

    Parameters
    ----------
    budgets
        The declared budgets.
    measurements
        Measurements by budget id; an absent id is unmeasured.

    Returns
    -------
    dict[str, object]
        ``{schema, verdicts, missed, unmeasured}``. Every failure is listed in
        ``missed``; nothing is retried or relaxed.

    Raises
    ------
    BudgetError
        A measurement names no declared budget or is not a finite number.

    """
    known = {budget.budget_id for budget in budgets}
    stray = sorted(set(measurements) - known)
    if stray:
        raise BudgetError(f"measurements name undeclared budgets: {', '.join(stray)}")
    verdicts = [judge_budget(budget, measurements.get(budget.budget_id)) for budget in budgets]
    return {
        "schema": RECORD_SCHEMA,
        "verdicts": [
            {
                "id": verdict.budget.budget_id,
                "comparator": verdict.budget.comparator,
                "target": verdict.budget.target,
                "unit": verdict.budget.unit,
                "measured": verdict.measured,
                "status": verdict.status,
            }
            for verdict in verdicts
        ],
        "missed": [verdict.budget.budget_id for verdict in verdicts if verdict.status == "fail"],
        "unmeasured": [
            verdict.budget.budget_id for verdict in verdicts if verdict.status == "unmeasured"
        ],
    }


TIERS: Final[tuple[tuple[int, int], ...]] = (
    (100, 300),
    (1_000, 3_000),
    (10_000, 30_000),
    (10_000, 100_000),
)
"""The graph tiers of the budgets as (nodes, relations)."""
FROZEN_TIER_DIGESTS: Final[tuple[str, ...]] = (
    "179d85ee4f296f636223c60ccb27e8915be46fa712b77795a4067967778f6a95",
    "5fec283fb0ee18576daf602b5878eef9db435bd124de2aebe909ef03df1a71a7",
    "2d93d95b156099d611e2cc20320709b45a007b1a5fc25f6cfa47694fd922380a",
    "3b0e272eb561060e9eab78506364c433b81b099e7a1f802314f13f393a9991a6",
)
"""Snapshot digests of :data:`TIERS`; the browser fixture pins the same values."""
PIN_SCHEMA: Final = "scpn_qc_studio_atlas_snapshot_pin_v1"


def tier_node_id(index: int) -> str:
    """Return the identifier of tier node ``index``.

    Parameters
    ----------
    index
        Node index.

    Returns
    -------
    str
        The node identifier the browser fixture uses too.

    """
    return f"definition:tier/n{index:05d}"


def tier_snapshot_document(nodes: int, relations: int) -> dict[str, object]:
    """Return the snapshot document of one tier, identical to the browser fixture's.

    Parameters
    ----------
    nodes
        Node count.
    relations
        Relation count, a multiple of ``nodes`` below its square; each node points
        to the next ``relations // nodes`` nodes.

    Returns
    -------
    dict[str, object]
        The wire snapshot.

    Raises
    ------
    BudgetError
        The relation count is not a multiple of the node count below its square.

    """
    fan_out, remainder = divmod(relations, nodes)
    if remainder or fan_out >= nodes:
        raise BudgetError("relations must be a multiple of nodes, below nodes squared")
    null_reasons = {"source_digest": "tier", "validity_domain": "tier", "as_of": "tier"}
    return {
        "schema": "quantum_atlas_snapshot.v1",
        "extractor_version": "tier",
        "source_commit": "0" * 40,
        "evidence_set_hash": "a" * 64,
        "access_scope": "public",
        "nodes": [
            {
                "id": tier_node_id(index),
                "kind": "definition",
                "label": f"Tier node {index}",
                "source_ref": f"tier/n{index}",
                "source_digest": None,
                "validity_domain": None,
                "as_of": None,
                "access": "public",
                "status": "current",
                "evidence_refs": [],
                "null_reasons": dict(null_reasons),
            }
            for index in range(nodes)
        ],
        "edges": [
            {
                "source": tier_node_id(at // fan_out),
                "kind": "requires",
                "target": tier_node_id((at // fan_out + 1 + at % fan_out) % nodes),
                "relation_version": "atlas_relation.v1",
                "derivation_method": "declared",
                "source_ref": "tier/edges",
                "evidence_digest": None,
                "assumptions": [],
                "mapping": None,
                "as_of": None,
                "access": "public",
            }
            for at in range(relations)
        ],
        "tombstones": [],
        "exclusions": [],
    }


def tier_pin_text(nodes: int, relations: int) -> str:
    """Return the pin text the Atlas opens for one tier, with its verified digest.

    Parameters
    ----------
    nodes
        Node count.
    relations
        Relation count.

    Returns
    -------
    str
        The pin as compact JSON.

    """
    from scpn_quantum_control.atlas import atlas_snapshot_digest, parse_atlas_snapshot

    document = tier_snapshot_document(nodes, relations)
    digest = atlas_snapshot_digest(parse_atlas_snapshot(document))
    return json.dumps(
        {"schema": PIN_SCHEMA, "snapshot": document, "snapshot_digest": digest},
        separators=(",", ":"),
    )


def initial_javascript_size(html: str, read: Callable[[str], bytes | None]) -> int:
    """Return the gzip size of every script an entry page loads before any lazy chunk.

    Parameters
    ----------
    html
        The entry page.
    read
        Returns the bytes of a bundle-relative path, or ``None`` when absent.

    Returns
    -------
    int
        The summed size, after gzip at level 9, of each distinct entry script and
        module preload.

    Raises
    ------
    BudgetError
        The page names no script, or names one outside the bundle or absent.

    """
    names = set(_ENTRY_SCRIPT.findall(html)) | set(_PRELOAD.findall(html))
    if not names:
        raise BudgetError("the entry page names no script")
    sizes: dict[PurePosixPath, int] = {}
    for name in names:
        relative = PurePosixPath(name[1:] if name.startswith("/") else name)
        if ".." in relative.parts or "://" in name or name.startswith("//"):
            raise BudgetError(f"{name} is outside the bundle")
        data = read(relative.as_posix())
        if data is None:
            raise BudgetError(f"{name} is not in the bundle")
        sizes[relative] = len(gzip.compress(data, compresslevel=9, mtime=0))
    return sum(sizes.values())


def initial_javascript_bytes(dist: Path) -> int:
    """Return :func:`initial_javascript_size` of a built bundle directory.

    Parameters
    ----------
    dist
        A built bundle directory holding ``index.html``.

    Returns
    -------
    int
        The compressed initial JavaScript, in bytes.

    """

    def read(relative: str) -> bytes | None:
        path = dist / relative
        return path.read_bytes() if path.is_file() else None

    return initial_javascript_size((dist / "index.html").read_text(encoding="utf-8"), read)


__all__ = [
    "BUDGETS_PATH",
    "FROZEN_TIER_DIGESTS",
    "PIN_SCHEMA",
    "TIERS",
    "BUDGETS_SCHEMA",
    "COMPARATORS",
    "RECORD_SCHEMA",
    "Budget",
    "BudgetError",
    "Verdict",
    "initial_javascript_bytes",
    "initial_javascript_size",
    "judge_budget",
    "judge_budgets",
    "load_budgets",
    "parse_budgets",
    "tier_node_id",
    "tier_pin_text",
    "tier_snapshot_document",
]
