#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — provision a self-hosted isolated-benchmark runner
#
# Registers a GitHub Actions self-hosted runner with the labels the CI workflow
# `differentiable-isolated-benchmark` requires (self-hosted, linux,
# isolated-benchmark), installs it as a systemd service, and pins the reserved
# benchmark core to the performance governor. Run on a dedicated, idle Linux
# host (not a developer workstation) so the isolation classifier promotes the
# evidence to `isolated_affinity`.
#
# Prerequisites: bash, curl, tar, sha256sum, sudo, and an authenticated `gh`
# (GitHub CLI) with admin access to the repository.
#
# Usage:
#   scripts/provision_isolated_benchmark_runner.sh \
#       [--repo anulum/scpn-quantum-control] \
#       [--name scpn-isolated-bench-01] \
#       [--work-dir "$HOME/actions-runner"] \
#       [--reserved-core 0]
set -euo pipefail

# Pinned GitHub Actions runner release.
# Verified: https://api.github.com/repos/actions/runner/releases/latest
RUNNER_VERSION="2.335.1"
RUNNER_SHA256="4ef2f25285f0ae4477f1fe1e346db76d2f3ebf03824e2ddd1973a2819bf6c8cf"

REPO="anulum/scpn-quantum-control"
RUNNER_NAME="scpn-isolated-bench-01"
WORK_DIR="${HOME}/actions-runner"
RESERVED_CORE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --name) RUNNER_NAME="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --reserved-core) RESERVED_CORE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for tool in curl tar sha256sum gh sudo; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 1; }
done

archive="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
url="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${archive}"

echo "==> Preparing runner directory: ${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if [[ ! -f "${archive}" ]]; then
  echo "==> Downloading runner ${RUNNER_VERSION}"
  curl -fsSL -o "${archive}" "${url}"
fi

echo "==> Verifying SHA-256"
echo "${RUNNER_SHA256}  ${archive}" | sha256sum --check --status \
  || { echo "SHA-256 verification FAILED for ${archive}" >&2; exit 1; }

if [[ ! -x "./config.sh" ]]; then
  echo "==> Extracting runner"
  tar xzf "${archive}"
fi

echo "==> Requesting a registration token for ${REPO}"
token="$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq '.token')"
[[ -n "${token}" ]] || { echo "failed to obtain a registration token" >&2; exit 1; }

echo "==> Configuring runner ${RUNNER_NAME}"
./config.sh \
  --unattended \
  --replace \
  --url "https://github.com/${REPO}" \
  --token "${token}" \
  --name "${RUNNER_NAME}" \
  --labels "self-hosted,linux,isolated-benchmark" \
  --work "_work"

echo "==> Installing the runner as a systemd service"
sudo ./svc.sh install
sudo ./svc.sh start

echo "==> Pinning cpu${RESERVED_CORE} to the performance governor"
governor_path="/sys/devices/system/cpu/cpu${RESERVED_CORE}/cpufreq/scaling_governor"
if [[ -w "${governor_path}" ]] || sudo test -w "${governor_path}"; then
  echo performance | sudo tee "${governor_path}" >/dev/null
  echo "    cpu${RESERVED_CORE} governor: $(cat "${governor_path}")"
else
  echo "    WARNING: ${governor_path} is not writable; set the governor manually." >&2
fi

echo
echo "Runner registered. Verify readiness, then dispatch the workflow:"
echo "  python scripts/check_isolated_benchmark_host.py --reserved-core ${RESERVED_CORE}"
echo "  gh workflow run ci.yml --repo ${REPO}"
echo
echo "For a fully reserved core, add 'isolcpus=${RESERVED_CORE} nohz_full=${RESERVED_CORE}'"
echo "to the kernel command line and reboot before benchmarking."
