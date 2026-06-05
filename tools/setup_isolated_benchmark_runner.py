# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Isolated benchmark runner setup helper.
"""Prepare a self-hosted GitHub Actions runner for isolated benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LABELS = ("self-hosted", "linux", "isolated-benchmark")
RUNNER_VERSION = "2.330.0"


@dataclass(frozen=True)
class IsolatedRunnerSetupPlan:
    """Concrete setup plan for the benchmark runner host."""

    repo: str
    runner_dir: Path
    runner_name: str
    labels: tuple[str, ...]
    runner_version: str = RUNNER_VERSION

    @property
    def archive_name(self) -> str:
        """Return the expected Linux x64 GitHub Actions runner archive name."""

        return f"actions-runner-linux-x64-{self.runner_version}.tar.gz"

    @property
    def download_url(self) -> str:
        """Return the GitHub Actions runner release archive URL."""

        return (
            "https://github.com/actions/runner/releases/download/"
            f"v{self.runner_version}/{self.archive_name}"
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready setup metadata."""

        return {
            "repo": self.repo,
            "runner_dir": str(self.runner_dir),
            "runner_name": self.runner_name,
            "labels": list(self.labels),
            "runner_version": self.runner_version,
            "download_url": self.download_url,
            "platform": platform.platform(),
            "claim_boundary": (
                "Runner setup only; benchmark claims require a later CI artefact "
                "classified as isolated_affinity."
            ),
        }


def build_runner_setup_plan(
    *, repo: str, runner_dir: Path, runner_name: str, labels: tuple[str, ...] = DEFAULT_LABELS
) -> IsolatedRunnerSetupPlan:
    """Build a deterministic isolated-runner setup plan."""

    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise ValueError("isolated benchmark runner setup currently supports Linux x64 only")
    if "self-hosted" not in labels or "linux" not in labels or "isolated-benchmark" not in labels:
        raise ValueError("labels must include self-hosted, linux, and isolated-benchmark")
    return IsolatedRunnerSetupPlan(
        repo=repo,
        runner_dir=runner_dir,
        runner_name=runner_name,
        labels=labels,
    )


def install_runner(plan: IsolatedRunnerSetupPlan) -> None:
    """Download, configure, and install the self-hosted runner service."""

    plan.runner_dir.mkdir(parents=True, exist_ok=True)
    archive_path = plan.runner_dir / plan.archive_name
    if not archive_path.exists():
        urllib.request.urlretrieve(plan.download_url, archive_path)  # noqa: S310
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract(archive, plan.runner_dir)
    token = _registration_token(plan.repo)
    subprocess.run(
        (
            str(plan.runner_dir / "config.sh"),
            "--unattended",
            "--url",
            f"https://github.com/{plan.repo}",
            "--token",
            token,
            "--name",
            plan.runner_name,
            "--labels",
            ",".join(plan.labels),
            "--work",
            "_work",
        ),
        cwd=plan.runner_dir,
        check=True,
    )
    subprocess.run(
        ("sudo", str(plan.runner_dir / "svc.sh"), "install"), cwd=plan.runner_dir, check=True
    )
    subprocess.run(
        ("sudo", str(plan.runner_dir / "svc.sh"), "start"), cwd=plan.runner_dir, check=True
    )


def _registration_token(repo: str) -> str:
    output = subprocess.check_output(
        ("gh", "api", f"repos/{repo}/actions/runners/registration-token"),
        text=True,
    )
    payload = json.loads(output)
    return str(payload["token"])


def _safe_extract(archive: tarfile.TarFile, target_dir: Path) -> None:
    """Extract a tar archive after rejecting path traversal entries."""

    target_root = target_dir.resolve()
    for member in archive.getmembers():
        destination = (target_dir / member.name).resolve()
        try:
            destination.relative_to(target_root)
        except ValueError:
            raise ValueError(f"unsafe runner archive member: {member.name}") from None
    archive.extractall(target_dir)  # nosec B202 - members are path-validated above.


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="anulum/scpn-quantum-control")
    parser.add_argument(
        "--runner-dir",
        type=Path,
        default=Path.home() / "actions-runner-scpn-qc-isolated",
    )
    parser.add_argument("--runner-name", default=f"{platform.node()}-scpn-qc-isolated")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)

    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    plan = build_runner_setup_plan(
        repo=args.repo,
        runner_dir=args.runner_dir,
        runner_name=args.runner_name,
        labels=labels,
    )
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    if args.install:
        install_runner(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
