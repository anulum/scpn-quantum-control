# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — committed data of the Studio benchmark databank
"""Build, write and check the committed pin of the Studio benchmark databank.

The Studio shows the canonical Kuramoto tier benchmarks of the CI and local
environments without rerunning them. The browser reads both committed
artefacts (``scpn-quantum-control.tier-benchmark.v1``) itself; this tool pins
what it must find there: the SHA-256 of each file, the owner's payload digest
(checked here with the owner's own function) and the generation time, beside
the boundary the databank states.

Run ``python tools/studio_benchmark_databank.py --write`` after a benchmark
artefact changes, and ``--check`` to compare the committed pin with a fresh
build.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from oscillatools.accel.tier_benchmark import payload_digest
from scpn_quantum_control.studio.committed_data import run_committed_data_cli

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
BENCHMARK_DATABANK_SCHEMA: Final[str] = "scpn_qc_studio_benchmark_databank_v1"
TIER_BENCHMARK_SCHEMA: Final[str] = "scpn-quantum-control.tier-benchmark.v1"
DEFAULT_PATH: Final[Path] = ROOT / "data/studio/benchmark_databank_20261115.json"
SOURCES: Final[dict[str, str]] = {
    "ci": "docs/benchmarks/tiers/kuramoto_tiers.ci.json",
    "local": "docs/benchmarks/tiers/kuramoto_tiers.local.json",
}
BOUNDARY: Final[str] = (
    "each environment is one host's measurement: medians of a few repeats after warm-up, "
    "not a competitive or production claim (production_claim_allowed is false in both "
    "artefacts); the CI baseline was measured on Python 3.12 before the CI job moved to "
    "Python 3.15, and its re-baseline is pending"
)


def pin_source(name: str, root: Path = ROOT) -> dict[str, object]:
    """Pin one tier artefact after the owner's checks.

    Parameters
    ----------
    name
        Environment key in :data:`SOURCES`.
    root
        Repository root holding the artefact.

    Returns
    -------
    dict[str, object]
        Path, file SHA-256, payload digest and generation time.

    Raises
    ------
    ValueError
        If the artefact has another schema, allows a production claim or its
        payload digest differs from the owner's recomputation.

    """
    raw = (root / SOURCES[name]).read_bytes()
    artifact: dict[str, Any] = json.loads(raw)
    if artifact.get("schema_version") != TIER_BENCHMARK_SCHEMA:
        raise ValueError(f"{SOURCES[name]} is not a {TIER_BENCHMARK_SCHEMA} artefact")
    if artifact.get("production_claim_allowed") is not False:
        raise ValueError(f"{SOURCES[name]} allows a production claim; the databank shows none")
    if payload_digest(artifact) != artifact["payload_sha256"]:
        raise ValueError(f"{SOURCES[name]} payload digest does not match its content")
    return {
        "path": SOURCES[name],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": artifact["payload_sha256"],
        "generated_utc": artifact["generated_utc"],
    }


def build_ascii_oracle() -> list[dict[str, object]]:
    """Record the tier owner's serialisation of non-ASCII text, which it escapes.

    The tier owner hashes ``json.dumps`` with Python's default
    ``ensure_ascii=True``; both committed artefacts are ASCII, so these samples
    give the browser's escaping an oracle the artefacts cannot.

    Returns
    -------
    list[dict[str, object]]
        Label, content and the exact text the owner's serialisation writes.

    """
    contents: list[tuple[str, dict[str, object]]] = [
        (
            "strings",
            {"value": ["é€", "\u2028\u2029", "\U0001d538", "\ufeff", 'quote " \\', "\x00\x7f"]},
        ),
        ("key order", {"\uffff": 1, "\U0001d538": 2, "é": 3, "a": 4, "Z": 0.5}),
    ]
    return [
        {
            "label": label,
            "content": content,
            "python_text": json.dumps(content, sort_keys=True, separators=(",", ":")),
        }
        for label, content in contents
    ]


def build_benchmark_databank(root: Path = ROOT) -> dict[str, object]:
    """Build the committed pin of the benchmark databank.

    Parameters
    ----------
    root
        Repository root holding the tier artefacts.

    Returns
    -------
    dict[str, object]
        Schema, identity, boundary, the pin of both environments and the
        escaping oracle.

    """
    return {
        "schema": BENCHMARK_DATABANK_SCHEMA,
        "artifact_id": "studio-benchmark-databank-20261115",
        "generated_by": "python tools/studio_benchmark_databank.py --write",
        "benchmark_schema": TIER_BENCHMARK_SCHEMA,
        "boundary": BOUNDARY,
        "sources": {name: pin_source(name, root) for name in SOURCES},
        "ascii_oracle": build_ascii_oracle(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed benchmark databank.

    Parameters
    ----------
    argv
        CLI arguments, or the process arguments when omitted.

    Returns
    -------
    int
        The shared committed-data writer's exit status.

    """
    return run_committed_data_cli(
        argv,
        build=build_benchmark_databank,
        default_path=DEFAULT_PATH,
        label="benchmark databank data",
        description=__doc__,
    )


if __name__ == "__main__":
    sys.exit(main())
