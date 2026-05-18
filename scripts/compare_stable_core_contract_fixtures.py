#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stable core contract fixture comparator
"""Compare regenerated stable-core contract fixtures.

The comparator is deterministic: it regenerates fixture JSON and Markdown
payloads into a temporary directory (unless explicit ``--actual-*`` paths are
provided) and compares deterministic outputs against committed artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from scpn_quantum_control.stable_core import (
    Problem,
    build_backend,
    build_experiment,
    build_problem,
    build_result,
    classical_reference_backend,
    hardware_replay_backend,
    pennylane_backend,
    pulser_surrogate_backend,
    qiskit_backend,
    qutip_backend,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED_JSON = REPO_ROOT / "data" / "stable_core" / "stable_core_contract_fixtures.json"
DEFAULT_EXPECTED_MARKDOWN = REPO_ROOT / "docs" / "stable_core_contract_fixtures.md"
DEFAULT_ACTUAL_JSON_NAME = "stable_core_contract_fixtures.json"
DEFAULT_ACTUAL_MARKDOWN_NAME = "stable_core_contract_fixtures.md"
SCHEMA_VERSION = "stable_core_contract_fixtures_v1"


def write_stable_core_contract_fixtures(
    *,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, str]:
    """Write deterministic stable-core contract fixture artifacts."""

    payload = stable_core_contract_fixtures_payload()
    json_text = stable_core_contract_fixtures_json(payload)
    markdown_text = stable_core_contract_fixtures_markdown(payload)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")

    return {
        "json_sha256": _sha256(json_text),
        "markdown_sha256": _sha256(markdown_text),
    }


def stable_core_contract_fixtures_payload() -> dict[str, Any]:
    """Build deterministic fixture payloads from stable-core contracts."""

    problems: tuple[Problem, ...] = (
        build_problem(
            problem_id="ring4",
            coupling_matrix=(
                (0.0, 0.45, 0.0, 0.45),
                (0.45, 0.0, 0.45, 0.0),
                (0.0, 0.45, 0.0, 0.45),
                (0.45, 0.0, 0.45, 0.0),
            ),
            omega=(0.8, 0.9, 1.1, 1.2),
            initial_state="0011",
            metadata={"source": "fixture", "domain": "stable-core"},
        ),
        build_problem(
            problem_id="chain3",
            coupling_matrix=(
                (0.0, 0.55, 0.0),
                (0.55, 0.0, 0.55),
                (0.0, 0.55, 0.0),
            ),
            omega=(1.0, 0.95, 1.05),
            initial_state="010",
            metadata={"source": "fixture", "domain": "stable-core", "lane": "chain"},
        ),
    )

    backends = (
        classical_reference_backend(metadata={"role": "baseline"}),
        hardware_replay_backend(metadata={"role": "planner-replay"}),
        qiskit_backend(metadata={"role": "adapter"}),
        qiskit_backend(
            backend_id="qiskit-runtime-live",
            hardware_submission_allowed=True,
            metadata={"role": "hardware-path", "mode": "fixture"},
        ),
        qutip_backend(metadata={"role": "open-system"}),
        pennylane_backend(metadata={"role": "autodiff"}),
        pulser_surrogate_backend(metadata={"role": "analog"}),
    )

    experiments = (
        build_experiment(
            experiment_id="exp-ring4-order-classical",
            problem=problems[0],
            backend=backends[0],
            objective="order_parameter",
            seed=17,
            shots=1024,
        ),
        build_experiment(
            experiment_id="exp-ring4-parity-replay",
            problem=problems[0],
            backend=backends[1],
            objective="parity_leakage",
            seed=23,
            shots=512,
        ),
        build_experiment(
            experiment_id="exp-chain3-mitigation-qiskit",
            problem=problems[1],
            backend=backends[2],
            objective="mitigation_replay",
            seed=31,
        ),
        build_experiment(
            experiment_id="exp-chain3-fim-qutip",
            problem=problems[1],
            backend=build_backend(
                backend_id="qutip-dynamics-fallback",
                kind="qutip",
                capabilities=("order_parameter", "hamiltonian_dynamics", "lindblad"),
                metadata={"role": "fim-fallback"},
            ),
            objective="order_parameter",
            seed=41,
            shots=256,
            metadata={"fixture": "fim"},
        ),
        build_experiment(
            experiment_id="exp-ring4-control-live",
            problem=problems[0],
            backend=build_backend(
                backend_id="classical-control",
                kind="classical_reference",
                capabilities=("order_parameter", "parity", "fim", "control"),
                metadata={"role": "control"},
            ),
            objective="control_cost",
            seed=7,
            metadata={"control_profile": "l2"},
        ),
        build_experiment(
            experiment_id="exp-ring4-order-qiskit-live",
            problem=problems[0],
            backend=backends[3],
            objective="order_parameter",
            seed=53,
            metadata={"preregistration_id": "fixture-001"},
        ),
    )

    results = (
        build_result(
            experiment_id=experiments[0].experiment_id,
            backend_id=experiments[0].backend.backend_id,
            status="succeeded",
            observables={"order_parameter": 0.742},
            artifacts=("artifacts/exp-ring4-order-classical.json",),
        ),
        build_result(
            experiment_id=experiments[1].experiment_id,
            backend_id=experiments[1].backend.backend_id,
            status="succeeded",
            observables={"parity_leakage": 0.91},
        ),
        build_result(
            experiment_id=experiments[2].experiment_id,
            backend_id=experiments[2].backend.backend_id,
            status="blocked",
            observables={},
            blockers=("mitigation replay requires calibrated parity observables",),
        ),
        build_result(
            experiment_id=experiments[3].experiment_id,
            backend_id=experiments[3].backend.backend_id,
            status="failed",
            observables={},
            blockers=("offline fixture path forbids hardware execution",),
            metadata={"fallback_mode": "fixture-only"},
        ),
        build_result(
            experiment_id=experiments[4].experiment_id,
            backend_id=experiments[4].backend.backend_id,
            status="succeeded",
            observables={"control_cost": 0.27},
            artifacts=("artifacts/exp-ring4-control-live.json",),
            metadata={"control_profile": "l2"},
        ),
        build_result(
            experiment_id=experiments[5].experiment_id,
            backend_id=experiments[5].backend.backend_id,
            status="blocked",
            observables={},
            blockers=("live qiskit path intentionally disabled in stable-core fixtures",),
            metadata={"mode": "rehearsal"},
        ),
    )

    payload = {
        "schema": SCHEMA_VERSION,
        "hardware_submission": False,
        "claim_boundary": (
            "Stable-core contract fixtures are deterministic shape checks only. "
            "They do not execute on hardware or invoke external providers."
        ),
        "problems": [problem.to_dict() for problem in problems],
        "backends": [backend.to_dict() for backend in backends],
        "experiments": [experiment.to_dict() for experiment in experiments],
        "results": [result.to_dict() for result in results],
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def stable_core_contract_fixtures_json(payload: dict[str, Any]) -> str:
    """Return deterministic JSON text for stable-core contract fixtures."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def stable_core_contract_fixtures_markdown(payload: dict[str, Any]) -> str:
    """Render a deterministic markdown summary for fixture artifacts."""

    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- scpn-quantum-control -- stable core contract fixtures -->",
        "",
        "# Stable Core Contract Fixtures",
        "",
        "These no-QPU, no-network fixtures lock stable core contract payloads.",
        "",
        "## Fixture summary",
        "",
        f"- Schema: `{payload['schema']}`",
        f"- Hardware submission enabled in fixtures: `{payload['hardware_submission']}`",
        "",
        "## Problems",
        "",
        "| Problem ID | Kind | Qubits | Initial state | Coupling matrix | Omega | Metadata |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in payload["problems"]:
        lines.append(
            "| `{problem_id}` | `{kind}` | {n_qubits} | `{initial_state}` | `{coupling_matrix}` | "
            "`{omega}` | `{metadata}` |".format(
                problem_id=row["problem_id"],
                kind=row["kind"],
                n_qubits=row["n_qubits"],
                initial_state=row["initial_state"],
                coupling_matrix=json.dumps(row["coupling_matrix"]),
                omega=json.dumps(row["omega"]),
                metadata=json.dumps(row["metadata"], sort_keys=True),
            )
        )

    lines.extend(
        [
            "",
            "## Backends",
            "",
            "| Backend ID | Kind | Capabilities | Hardware submission | Metadata |",
            "|---|---|---|---|---|",
        ]
    )
    for row in payload["backends"]:
        lines.append(
            "| `{backend_id}` | `{kind}` | {capabilities} | `{hardware_submission_allowed}` | "
            "`{metadata}` |".format(
                backend_id=row["backend_id"],
                kind=row["kind"],
                capabilities=", ".join(row["capabilities"]),
                hardware_submission_allowed=row["hardware_submission_allowed"],
                metadata=json.dumps(row["metadata"], sort_keys=True),
            )
        )

    lines.extend(
        [
            "",
            "## Experiments",
            "",
            "| Experiment ID | Problem ID | Backend ID | Objective | Seed | Shots | Metadata |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["experiments"]:
        lines.append(
            "| `{experiment_id}` | `{problem_id}` | `{backend_id}` | `{objective}` | `{seed}` | "
            "`{shots}` | `{metadata}` |".format(
                experiment_id=row["experiment_id"],
                problem_id=row["problem"]["problem_id"],
                backend_id=row["backend"]["backend_id"],
                objective=row["objective"],
                seed=row["seed"],
                shots=row["shots"],
                metadata=json.dumps(row["metadata"], sort_keys=True),
            )
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Experiment ID | Backend ID | Status | Observables | Blockers | Artifacts |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in payload["results"]:
        lines.append(
            "| `{experiment_id}` | `{backend_id}` | `{status}` | `{observables}` | "
            "{blockers} | `{artifacts}` |".format(
                experiment_id=row["experiment_id"],
                backend_id=row["backend_id"],
                status=row["status"],
                observables=json.dumps(row["observables"], sort_keys=True),
                blockers=(", ".join(row["blockers"]) if row["blockers"] else "`none`"),
                artifacts=", ".join(row["artifacts"]) if row["artifacts"] else "`none`",
            )
        )

    lines.extend(
        [
            "",
            "## Reproducibility gate",
            "",
            "Regenerate and compare these fixtures with:",
            "",
            "```bash",
            "python scripts/run_stable_core_contract_gate.py",
            "```",
            "",
            "## Claim boundary",
            "",
            str(payload["claim_boundary"]),
        ]
    )
    return "\n".join(lines) + "\n"


def compare_stable_core_contract_fixtures(
    *,
    expected_json_path: Path,
    expected_markdown_path: Path,
    actual_json_path: Path | None = None,
    actual_markdown_path: Path | None = None,
) -> dict[str, Any]:
    """Compare committed stable-core contract artifacts with regenerated ones."""

    blockers: list[str] = []

    if (actual_json_path is None) != (actual_markdown_path is None):
        raise ValueError("--actual-json and --actual-markdown must be provided together")

    if actual_json_path is not None and actual_markdown_path is not None:
        actual_json = actual_json_path
        actual_markdown = actual_markdown_path
        generated_digests = None
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            actual_json = tmp_dir / DEFAULT_ACTUAL_JSON_NAME
            actual_markdown = tmp_dir / DEFAULT_ACTUAL_MARKDOWN_NAME
            generated_digests = write_stable_core_contract_fixtures(
                json_path=actual_json,
                markdown_path=actual_markdown,
            )
            return {
                "schema": SCHEMA_VERSION,
                "valid": _comparison_result(
                    expected_json_path=expected_json_path,
                    expected_markdown_path=expected_markdown_path,
                    actual_json_path=actual_json,
                    actual_markdown_path=actual_markdown,
                    blockers=blockers,
                ),
                "blockers": tuple(blockers),
                "digests": {
                    "expected_json_sha256": _sha256_path(expected_json_path),
                    "expected_markdown_sha256": _sha256_path(expected_markdown_path),
                    "actual_json_sha256": generated_digests["json_sha256"],
                    "actual_markdown_sha256": generated_digests["markdown_sha256"],
                },
                "expected_paths": {
                    "json": str(expected_json_path),
                    "markdown": str(expected_markdown_path),
                },
                "actual_paths": {
                    "json": str(actual_json),
                    "markdown": str(actual_markdown),
                },
            }

    return {
        "schema": SCHEMA_VERSION,
        "valid": _comparison_result(
            expected_json_path=expected_json_path,
            expected_markdown_path=expected_markdown_path,
            actual_json_path=actual_json,
            actual_markdown_path=actual_markdown,
            blockers=blockers,
        ),
        "blockers": tuple(blockers),
        "digests": {
            "expected_json_sha256": _sha256_path(expected_json_path),
            "expected_markdown_sha256": _sha256_path(expected_markdown_path),
            "actual_json_sha256": _sha256_path(actual_json),
            "actual_markdown_sha256": _sha256_path(actual_markdown),
        },
        "expected_paths": {
            "json": str(expected_json_path),
            "markdown": str(expected_markdown_path),
        },
        "actual_paths": {
            "json": str(actual_json),
            "markdown": str(actual_markdown),
        },
    }


def _comparison_result(
    *,
    expected_json_path: Path,
    expected_markdown_path: Path,
    actual_json_path: Path,
    actual_markdown_path: Path,
    blockers: list[str],
) -> bool:
    """Populate blockers and report whether artifacts match exactly."""

    valid = True
    expected_json = _load_json(expected_json_path, blockers)
    actual_json = _load_json(actual_json_path, blockers)
    expected_markdown = _normalise_markdown(_load_text(expected_markdown_path, blockers))
    actual_markdown = _normalise_markdown(_load_text(actual_markdown_path, blockers))

    if expected_json is not None and actual_json is not None:
        if expected_json.get("schema") != SCHEMA_VERSION:
            blockers.append("stable core fixture schema mismatch")
            valid = False
        if _normalised_json(expected_json) != _normalised_json(actual_json):
            blockers.append("stable-core fixture JSON artifacts differ from committed version")
            valid = False

    if (
        expected_markdown is not None
        and actual_markdown is not None
        and expected_markdown != actual_markdown
    ):
        blockers.append("stable-core fixture Markdown artifacts differ from committed version")
        valid = False

    if not expected_json_path.exists() or not actual_json_path.exists():
        valid = False
    if not expected_markdown_path.exists() or not actual_markdown_path.exists():
        valid = False

    if blockers:
        valid = False
    return valid


def _normalise_markdown(text: str) -> str:
    """Normalise markdown line endings for deterministic comparisons."""

    return text.replace("\r\n", "\n")


def _normalised_json(payload: dict[str, Any]) -> str:
    """Canonical JSON text for deterministic drift checks."""

    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _load_json(path: Path, blockers: list[str]) -> dict[str, Any] | None:
    """Load JSON payload with deterministic blocker recording."""

    try:
        return json.loads(_load_text(path, blockers))
    except json.JSONDecodeError as exc:
        blockers.append(f"{path} must contain valid JSON: {exc}")
        return None


def _load_text(path: Path, blockers: list[str]) -> str:
    """Load text with deterministic blocker recording."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        blockers.append(f"unable to read {path}: {exc}")
        return ""


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    try:
        return _sha256(path.read_text(encoding="utf-8"))
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for stable-core contract fixture comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-json", type=Path, default=DEFAULT_EXPECTED_JSON)
    parser.add_argument("--expected-markdown", type=Path, default=DEFAULT_EXPECTED_MARKDOWN)
    parser.add_argument("--actual-json", type=Path)
    parser.add_argument("--actual-markdown", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON payload")
    args = parser.parse_args(argv)

    payload = compare_stable_core_contract_fixtures(
        expected_json_path=args.expected_json,
        expected_markdown_path=args.expected_markdown,
        actual_json_path=args.actual_json,
        actual_markdown_path=args.actual_markdown,
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"stable core contract fixture comparison valid: {payload['valid']}")
        for blocker in payload["blockers"]:
            print(f"  blocker: {blocker}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
