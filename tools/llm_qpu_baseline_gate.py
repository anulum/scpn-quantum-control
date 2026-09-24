# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — offline LLM-QPU baseline evidence gates
"""Capture baseline refusals and failures without importing provider SDKs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_MAX_OUTPUT_CHARS = 16_384
_TOKEN_MARKERS = ("TOKEN", "SECRET", "API_KEY", "PASSWORD", "CREDENTIAL")


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        raise ValueError("baseline checkout has no git HEAD")
    return result.stdout.strip()


def verify_source_anchor(
    repo: Path, *, expected_head: str, source_sha256: Mapping[str, str]
) -> dict[str, object]:
    """Refuse a stale commit or any changed audited source byte sequence."""
    root = repo.resolve(strict=True)
    current_head = _git_head(root)
    failures: list[str] = []
    if current_head != expected_head:
        failures.append("HEAD differs from audited baseline")
    for relative, expected_digest in sorted(source_sha256.items()):
        path = (root / relative).resolve(strict=False)
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(f"audited source unavailable: {relative}")
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected_digest:
            failures.append(f"audited source changed: {relative}")
    return {
        "status": "REFUSED" if failures else "PASS",
        "expected_head": expected_head,
        "current_head": current_head,
        "checked_sources": len(source_sha256),
        "failures": failures,
    }


def dependency_report(
    module_names: Sequence[str], *, python: Path = Path(sys.executable), isolated: bool = False
) -> dict[str, object]:
    """Report missing optional SDK modules as BLOCKED, never as a skip or pass."""
    code = (
        "import importlib.util,json,sys;"
        "missing=[];"
        "\nfor name in sys.argv[1:]:"
        "\n try: found=importlib.util.find_spec(name) is not None"
        "\n except (ImportError,ModuleNotFoundError,ValueError): found=False"
        "\n if not found: missing.append(name)"
        "\nprint(json.dumps(missing))"
    )
    command = [str(python), *(["-I", "-S"] if isolated else []), "-c", code, *module_names]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        return {
            "status": "BLOCKED",
            "missing_modules": list(module_names),
            "reason": "probe failed",
        }
    missing = json.loads(result.stdout)
    return {"status": "BLOCKED" if missing else "PASS", "missing_modules": missing}


def scrubbed_environment(home: Path, inherited: Mapping[str, str]) -> dict[str, str]:
    """Construct a small subprocess environment without credential-bearing keys."""
    allowed = {"PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR"}
    clean = {
        key: value
        for key, value in inherited.items()
        if key.upper() in allowed and not any(marker in key.upper() for marker in _TOKEN_MARKERS)
    }
    clean.update({"HOME": str(home), "PYTHONNOUSERSITE": "1"})
    return clean


def run_baseline_command(
    command: Sequence[str], *, cwd: Path, report_path: Path, timeout: int = 30
) -> dict[str, Any]:
    """Preserve a real command failure and its cause in an atomic JSON report."""
    if not command:
        raise ValueError("baseline command must not be empty")
    with tempfile.TemporaryDirectory(prefix="llm-qpu-baseline-") as home_name:
        environment = scrubbed_environment(Path(home_name), os.environ)
        try:
            process = subprocess.run(
                list(command),
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            report: dict[str, Any] = {
                "status": "PASS" if process.returncode == 0 else "FAIL",
                "command": list(command),
                "exit_code": process.returncode,
                "stdout": process.stdout[-_MAX_OUTPUT_CHARS:],
                "stderr": process.stderr[-_MAX_OUTPUT_CHARS:],
            }
        except subprocess.TimeoutExpired:
            report = {
                "status": "BLOCKED",
                "command": list(command),
                "exit_code": None,
                "reason": "baseline command timed out",
            }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=report_path.parent, delete=False
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    temporary.replace(report_path)
    return report


def offline_container_probe() -> dict[str, object]:
    """Require a credential-free, networkless Docker namespace for W00 probes."""
    with tempfile.TemporaryDirectory(prefix="llm-qpu-offline-") as home_name:
        home = Path(home_name)
        child = (
            "import json,os,pathlib,socket;"
            "markers=('TOKEN','SECRET','API_KEY','PASSWORD','CREDENTIAL');"
            "exposed=[k for k in os.environ if any(m in k.upper() for m in markers)];"
            "vault=pathlib.Path.home()/'.config/scpn-quantum-control/credentials.md';"
            "routes=pathlib.Path('/proc/net/route').read_text().splitlines()[1:];"
            "default=any(len(row.split())>2 and row.split()[1]=='00000000' for row in routes);"
            "network=False;"
            "\ntry: socket.create_connection(('198.51.100.1',443),timeout=1)"
            "\nexcept OSError: pass"
            "\nelse: network=True"
            "\nprint(json.dumps({'exposed':exposed,'vault_exists':vault.exists(),"
            "'default_route':default,'network_connected':network}))"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", child],
            env=scrubbed_environment(home, os.environ),
            cwd=home,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    if result.returncode:
        return {"status": "BLOCKED", "reason": "offline subprocess failed"}
    evidence = json.loads(result.stdout)
    return {
        "status": "PASS" if not any(evidence.values()) else "REFUSED",
        **evidence,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Docker-only W00 offline isolation check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("offline-container-probe",))
    arguments = parser.parse_args(argv)
    if arguments.operation == "offline-container-probe":
        report = offline_container_probe()
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "PASS" else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
