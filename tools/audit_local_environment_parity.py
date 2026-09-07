# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — local environment parity report
"""Report where this workstation has drifted from what CI pins.

A local run only means something if it ran against the environment CI runs
against. The pins move: locks get regenerated, action versions get bumped, a
runner image ships a newer default. None of that is visible from inside a test
run — it shows up as a failure that reproduces on one machine and not the other,
or worse, as a pass on one machine and a failure on the other.

This is an instrument, not a gate. It reads the repository as the source of
truth — the workflows for interpreter and tool pins, the hash-locked
requirements for distributions, ``rust-toolchain.toml`` for the toolchain — and
prints every axis where the live environment says something different. It is
deliberately not wired into ``tools/preflight.py``: a workstation legitimately
differs from a runner on the optional tiers, and a push should not be blocked by
that.

Two axes cannot be closed from here and the report says so rather than passing
them silently. The local interpreter is a different build of the same CPython
version than ``actions/setup-python`` ships, and ``juliapkg`` reuses a system
Julia when one is on PATH while a clean runner downloads its own.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from shutil import which
from typing import Final

import yaml

#: The hash-locked set every CI test job installs.
BASE_LOCK: Final[str] = "requirements-ci-py312-linux.txt"

#: Matches a pinned distribution in a requirement lock, extras included.
PINNED_DISTRIBUTION: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s\\]+)", re.M
)

#: Actions whose ``version`` input pins a tool this workstation also provides.
TOOL_ACTIONS: Final[dict[str, str]] = {"pnpm/action-setup": "pnpm"}


@dataclass(frozen=True)
class Divergence:
    """One axis on which the workstation and CI disagree."""

    axis: str
    expected: str
    observed: str

    def __str__(self) -> str:
        """Render the divergence as one reviewable line."""
        return f"{self.axis}: CI {self.expected}, local {self.observed}"


def _normalise(name: str) -> str:
    """Return a distribution name in the form PyPI compares."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _command_version(executable: str, *arguments: str) -> str | None:
    """Return the first line an executable prints, or None when it is absent."""
    path = which(executable)
    if path is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603
            [path, *arguments], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return output[0].strip() if output else None


def workflow_pins(repo: Path) -> tuple[str | None, dict[str, str]]:
    """Return the interpreter version CI pins most, and its pinned tool versions."""
    interpreters: Counter[str] = Counter()
    tools: dict[str, str] = {}
    for path in sorted((repo / ".github" / "workflows").glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        jobs = document.get("jobs") if isinstance(document, dict) else None
        if not isinstance(jobs, dict):
            continue
        for job in jobs.values():
            steps = job.get("steps") if isinstance(job, dict) else None
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                uses, inputs = step.get("uses"), step.get("with")
                if not isinstance(uses, str) or not isinstance(inputs, dict):
                    continue
                action = uses.split("@")[0]
                if action == "actions/setup-python":
                    version = inputs.get("python-version")
                    if isinstance(version, str) and not version.startswith("${{"):
                        interpreters[version] += 1
                tool = TOOL_ACTIONS.get(action)
                version = inputs.get("version")
                if tool is not None and version is not None:
                    tools[tool] = str(version)
    most_common = interpreters.most_common(1)
    return (most_common[0][0] if most_common else None), tools


def interpreter_divergences(pinned: str | None) -> list[Divergence]:
    """Compare the running interpreter against the version CI pins."""
    if pinned is None:
        return []
    local = f"{sys.version_info.major}.{sys.version_info.minor}"
    if local != pinned:
        return [Divergence("python", pinned, local)]
    return []


def distribution_divergences(repo: Path) -> list[Divergence]:
    """Compare every pinned distribution against what is installed."""
    lock = repo / BASE_LOCK
    if not lock.is_file():
        return [Divergence("base lock", BASE_LOCK, "missing")]
    installed = {
        _normalise(dist.metadata["Name"]): dist.version
        for dist in metadata.distributions()
        if dist.metadata["Name"]
    }
    divergences: list[Divergence] = []
    for name, version in PINNED_DISTRIBUTION.findall(lock.read_text(encoding="utf-8")):
        key = _normalise(name)
        present = installed.get(key)
        if present is None:
            divergences.append(Divergence(f"package {key}", version, "absent"))
        elif present != version:
            divergences.append(Divergence(f"package {key}", version, present))
    return divergences


def tool_divergences(pinned: dict[str, str]) -> list[Divergence]:
    """Compare the pinned Node-side tools against the ones on PATH."""
    divergences: list[Divergence] = []
    for tool, version in sorted(pinned.items()):
        observed = _command_version(tool, "--version")
        if observed is None:
            divergences.append(Divergence(tool, version, "not on PATH"))
        elif observed != version:
            divergences.append(Divergence(tool, version, observed))
    return divergences


def toolchain_divergences(repo: Path) -> list[Divergence]:
    """Compare the declared Rust toolchain against the installed one."""
    declaration = repo / "rust-toolchain.toml"
    if not declaration.is_file():
        return [Divergence("rust-toolchain.toml", "declared", "missing")]
    document = tomllib.loads(declaration.read_text(encoding="utf-8"))
    toolchain = document.get("toolchain", {})
    divergences: list[Divergence] = []

    if _command_version("rustc", "--version") is None:
        divergences.append(Divergence("rustc", str(toolchain.get("channel")), "not on PATH"))
        return divergences

    for kind, key in (("component", "components"), ("target", "targets")):
        listing = _rustup_listing(kind)
        for wanted in toolchain.get(key, []):
            if wanted not in listing:
                divergences.append(Divergence(f"rust {kind} {wanted}", "installed", "absent"))
    return divergences


def _rustup_listing(kind: str) -> str:
    """Return what rustup reports as installed for a component or target."""
    rustup = which("rustup")
    if rustup is None:
        return ""
    try:
        completed = subprocess.run(  # noqa: S603
            [rustup, kind, "list", "--installed"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout


def engine_divergences() -> list[Divergence]:
    """Report whether the native engine CI builds is importable here."""
    try:
        import scpn_quantum_engine  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001
        return [Divergence("native engine", "built and installed", f"{type(error).__name__}")]
    if getattr(scpn_quantum_engine, "__file__", None) is None:
        return [Divergence("native engine", "built and installed", "no module file")]
    return []


def audit(repo: Path) -> list[Divergence]:
    """Collect every axis on which this workstation differs from CI."""
    interpreter, tools = workflow_pins(repo)
    return [
        *interpreter_divergences(interpreter),
        *distribution_divergences(repo),
        *tool_divergences(tools),
        *toolchain_divergences(repo),
        *engine_divergences(),
    ]


def main(argv: list[str] | None = None) -> int:
    """Print the parity report and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    arguments = parser.parse_args(argv)

    divergences = audit(arguments.repo)
    print("Axes this report cannot close, by construction:")
    print("    interpreter build: setup-python ships a GCC build; uv ships a Clang one")
    print("    julia: juliapkg reuses a system interpreter here, downloads one on a runner")
    if divergences:
        print(f"\n{len(divergences)} divergence(s) from what CI pins:")
        for divergence in divergences:
            print(f"    {divergence}")
        return 1
    print("\nEvery checked axis matches what CI pins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
