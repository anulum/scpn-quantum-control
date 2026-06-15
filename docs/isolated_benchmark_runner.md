# Isolated Benchmark Runner

SPDX-License-Identifier: AGPL-3.0-or-later

Differentiable-benchmark evidence is promoted to `isolated_affinity` only when it
runs on a self-hosted GitHub Actions runner with a reserved CPU, a recorded
governor and frequency, and low host load. Without such a runner the
`differentiable-isolated-benchmark` job in `.github/workflows/ci.yml` cannot
execute, and every committed benchmark stays `functional_non_isolated`. This
runbook registers that runner and confirms the host is configured to produce
isolated evidence.

Run the runner on a **dedicated, idle Linux host** — not a developer workstation.
A workstation under load reports a load average well above the `1.0` threshold
and is rejected by the readiness check.

## 1. Check host readiness

```bash
python scripts/check_isolated_benchmark_host.py --reserved-core 0
```

The check exits non-zero and lists blockers when the reserved core is not on the
`performance` governor, the governor or frequency is unreadable, or the host
load average exceeds `1.0`. Fix every blocker before registering the runner.

## 2. Provision the runner

```bash
scripts/provision_isolated_benchmark_runner.sh \
    --repo anulum/scpn-quantum-control \
    --name scpn-isolated-bench-01 \
    --reserved-core 0
```

The script pins GitHub Actions runner `v2.335.1`
(`sha256:4ef2f25285f0ae4477f1fe1e346db76d2f3ebf03824e2ddd1973a2819bf6c8cf`,
verified at the releases API), downloads and SHA-256-verifies the archive,
requests a short-lived registration token through the authenticated `gh` CLI,
configures the runner with the `self-hosted,linux,isolated-benchmark` labels,
installs it as a systemd service, and sets the reserved core to the
`performance` governor.

For a fully reserved core, add `isolcpus=0 nohz_full=0` to the kernel command
line and reboot before benchmarking, so the scheduler keeps other work off the
benchmark core.

## 3. Dispatch the isolated benchmark

The job is gated on `workflow_dispatch`:

```bash
gh workflow run ci.yml --repo anulum/scpn-quantum-control
```

It pins the work to the reserved core with `taskset -c 0 chrt -f 1`, runs
`scripts/run_differentiable_benchmark_evidence.py`, asserts the artefact
classifies as `isolated_affinity`, and uploads it. Commit the small JSON/CSV
summary as the isolated benchmark evidence; keep the raw artefact under the
ignored results tree.

## How classification works

`scpn_quantum_control.benchmarks.differentiable_evidence.BenchmarkIsolationMetadata`
promotes a run to `isolated_affinity` only when it sees a self-hosted
isolated-benchmark runner, a CPU affinity, a recorded governor and frequency,
low host load before and after, and no heavy concurrent jobs. The readiness
checker reuses the same `1.0` load threshold, so a host that passes the check
will not be downgraded for load or governor reasons at run time.
