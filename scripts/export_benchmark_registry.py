# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S5 benchmark registry export
"""Export the public benchmark-harness registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scpn_quantum_control.benchmark_harness.registry import benchmark_registry_payload

DATE = "2026-05-06"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s5_benchmark_harness"
DOC_PATH = REPO_ROOT / f"docs/campaigns/benchmark_harness_registry_{DATE}.md"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Harness Registry",
        "",
        "This registry distinguishes implemented public benchmark harnesses from planned entries. Planned rows are visible so the roadmap is transparent, but they are not treated as available benchmark results.",
        "",
        "## Counts",
        f"- Implemented: `{payload['implemented_count']}`",
        f"- Planned: `{payload['planned_count']}`",
        f"- Blocked: `{payload['blocked_count']}`",
        "",
        "## Families",
    ]
    for family in payload["families"]:
        lines.extend(
            [
                "",
                f"### {family['benchmark_id']}",
                f"- Title: {family['title']}",
                f"- Status: `{family['status']}`",
                f"- Public API: `{family['public_api']}`",
                f"- Command: `{family['command']}`",
                f"- Dataset: `{family['dataset']}`",
                f"- Generated artefact: `{family['generated_artifact']}`",
                f"- Baseline: {family['baseline']}",
                f"- Claim boundary: {family['claim_boundary']}",
            ]
        )
        if family.get("blocker"):
            lines.append(f"- Blocker: {family['blocker']}")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse the benchmark registry export CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    parser.add_argument("--implemented-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Write the benchmark registry JSON and Markdown artefacts."""
    args = parse_args()
    payload = benchmark_registry_payload(include_planned=not args.implemented_only)
    json_path = args.out_dir / f"benchmark_registry_{DATE}.json"
    sha_json = _write_json(json_path, payload)
    sha_md = _write_text(args.doc_path, _markdown(payload))
    print(f"wrote {json_path.relative_to(REPO_ROOT)} sha256={sha_json}")
    print(f"wrote {args.doc_path.relative_to(REPO_ROOT)} sha256={sha_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
