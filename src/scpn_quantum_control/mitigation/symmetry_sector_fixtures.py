# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- symmetry-sector mitigation fixture artefacts
"""Deterministic fixtures for the symmetry-sector mitigation planner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .symmetry_sector_compiler import (
    SymmetrySectorProblem,
    plan_symmetry_sector_mitigation,
)
from .symmetry_sector_replay import replay_symmetry_sector_counts

SCHEMA = "symmetry_sector_mitigation_fixture_v1"
DEFAULT_OUT_DIR = Path("data") / "symmetry_sector_mitigation"
DEFAULT_DOC_PATH = Path("docs") / "symmetry_sector_mitigation_fixtures.md"


def _json_ready(data: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON round-tripped object with lists instead of tuples."""

    payload: dict[str, Any] = json.loads(json.dumps(data, sort_keys=True))
    return payload


def fixture_problems() -> dict[str, SymmetrySectorProblem]:
    """Return canonical planner fixture inputs."""

    valid = SymmetrySectorProblem(
        n_qubits=4,
        coupling_matrix=(
            (0.0, 0.45, 0.0, 0.45),
            (0.45, 0.0, 0.45, 0.0),
            (0.0, 0.45, 0.0, 0.45),
            (0.45, 0.0, 0.45, 0.0),
        ),
        omega=(0.8, 0.9333333333, 1.0666666667, 1.2),
        initial_state="0011",
        measurement_basis="counts",
        has_raw_counts=True,
        has_noise_scaled_symmetry_observables=True,
    )
    valid_payload = asdict(valid)
    return {
        "eligible_counts_guess": valid,
        "blocked_missing_counts": SymmetrySectorProblem(
            **{**valid_payload, "has_raw_counts": False}
        ),
        "blocked_missing_guess_observables": SymmetrySectorProblem(
            **{**valid_payload, "has_noise_scaled_symmetry_observables": False}
        ),
        "blocked_nonsymmetric_coupling": SymmetrySectorProblem(
            **{
                **valid_payload,
                "coupling_matrix": (
                    (0.0, 0.45, 0.0, 0.0),
                    (0.0, 0.0, 0.45, 0.0),
                    (0.0, 0.45, 0.0, 0.45),
                    (0.45, 0.0, 0.45, 0.0),
                ),
            }
        ),
    }


def fixture_payload() -> dict[str, Any]:
    """Build the deterministic fixture payload."""

    fixtures = []
    for fixture_id, problem in fixture_problems().items():
        plan = plan_symmetry_sector_mitigation(problem)
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "problem": asdict(problem),
                "plan": plan.to_dict(),
            }
        )
    return _json_ready(
        {
            "schema": SCHEMA,
            "hardware_submission": False,
            "fixtures": fixtures,
            "replay_fixtures": replay_fixture_rows(),
            "claim_boundary": (
                "Planner fixtures prove deterministic eligibility/blocker outputs. "
                "Replay fixtures prove offline raw-count accounting. They do not mutate circuits, "
                "submit hardware jobs, or prove hardware improvement."
            ),
        }
    )


def replay_fixture_rows() -> list[dict[str, Any]]:
    """Return deterministic raw-count replay fixture rows."""

    problems = fixture_problems()
    rows: list[dict[str, Any]] = []
    applied = replay_symmetry_sector_counts(
        problems["eligible_counts_guess"],
        {"0011": 40, "0000": 10, "0001": 5, "1110": 7},
    )
    rows.append(
        {
            "fixture_id": "replay_counts_postselection_expansion",
            "status": applied.status,
            "result": applied.to_dict(),
        }
    )
    try:
        replay_symmetry_sector_counts(problems["blocked_missing_counts"], {"0011": 10})
    except ValueError as exc:
        rows.append(
            {
                "fixture_id": "replay_blocked_missing_counts",
                "status": "blocked",
                "blockers": [str(exc)],
            }
        )
    else:
        raise AssertionError("blocked replay fixture unexpectedly succeeded")
    return rows


def normalised_json(data: dict[str, Any]) -> str:
    """Return deterministic JSON text for fixture comparison and writing."""

    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, data: dict[str, Any]) -> str:
    """Write deterministic JSON and return its SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = normalised_json(data)
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def fixture_markdown(data: dict[str, Any]) -> str:
    """Render a public fixture summary."""

    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- scpn-quantum-control -- symmetry-sector mitigation fixtures -->",
        "",
        "# Symmetry-Sector Mitigation Planner Fixtures",
        "",
        "These no-QPU fixtures lock the planner contract before execution-path integration.",
        "",
        "| Fixture | Status | Primitives | Blockers |",
        "|---|---|---|---|",
    ]
    for fixture in data["fixtures"]:
        plan = fixture["plan"]
        lines.append(
            "| `{fixture}` | `{status}` | {primitives} | {blockers} |".format(
                fixture=fixture["fixture_id"],
                status=plan["status"],
                primitives=", ".join(plan["primitives"]) or "none",
                blockers="; ".join(plan["blockers"]) or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Raw-count replay fixtures",
            "",
            "| Fixture | Status | Applied primitives | Deferred primitives | Blockers |",
            "|---|---|---|---|---|",
        ]
    )
    for fixture in data["replay_fixtures"]:
        if fixture["status"] == "applied":
            result = fixture["result"]
            applied = ", ".join(result["applied_primitives"]) or "none"
            deferred = ", ".join(result["deferred_primitives"]) or "none"
            blockers = "; ".join(result["blockers"]) or "none"
        else:
            applied = "none"
            deferred = "none"
            blockers = "; ".join(fixture["blockers"]) or "none"
        lines.append(
            "| `{fixture}` | `{status}` | {applied} | {deferred} | {blockers} |".format(
                fixture=fixture["fixture_id"],
                status=fixture["status"],
                applied=applied,
                deferred=deferred,
                blockers=blockers,
            )
        )
    lines.extend(
        [
            "",
            "## Reproducibility gate",
            "",
            "Regenerate and compare the fixtures with:",
            "",
            "```bash",
            "scpn-bench symmetry-sector-mitigation-gate",
            "```",
            "",
            "## Claim boundary",
            "",
            str(data["claim_boundary"]),
        ]
    )
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> str:
    """Write text and return its SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
