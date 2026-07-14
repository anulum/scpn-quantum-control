# Real-Time Runtime

SPDX-License-Identifier: AGPL-3.0-or-later

`scpn_quantum_control.control.realtime_runtime` provides software timing
contracts for bounded control loops. It has two surfaces:

1. A fixed-period control-loop runtime (`run_realtime_control_loop`,
   `RealtimeRuntimeConfig`, SLA evaluation) for second/millisecond-scale loops.
2. A **sub-microsecond outer-loop tracker** (`SubMicrosecondTracker`) for
   high-rate jitter and deadline telemetry.

Both are deterministic software-timing surfaces. Neither is an intra-shot
hardware-latency claim; a downstream sub-50 ns trigger path is covered by RTL
assertions in the consumer, not by this runtime.

## Fixed-period control loop

`RealtimeRuntimeConfig` defines the period, execution deadline, start-jitter
budget, tolerated miss count, and scheduling mode. With `align_to_period=True`,
the runtime waits for `start + index * sample_period_s` before each step. With
alignment disabled, the caller drives tick timing immediately, while the same
schedule remains in each `RealtimeTickRecord` for drift telemetry.

```python
from scpn_quantum_control import (
    RealtimeRuntimeConfig,
    RealtimeSLAConfig,
    VirtualRealtimeClock,
    evaluate_realtime_sla,
    run_realtime_control_loop,
)

clock = VirtualRealtimeClock()
durations_s = (0.00052, 0.00061, 0.00074)


def step(index: int) -> dict[str, float]:
    clock.advance(durations_s[index])
    return {"command_norm": float(index + 1)}


result = run_realtime_control_loop(
    len(durations_s),
    step,
    config=RealtimeRuntimeConfig(
        sample_period_s=0.0012,
        deadline_s=0.00095,
        jitter_budget_s=0.00015,
    ),
    clock=clock,
)
report = evaluate_realtime_sla(
    result,
    sla=RealtimeSLAConfig(
        max_latency_s=0.001,
        max_jitter_s=0.00025,
        p95_latency_s=0.001,
        p99_latency_s=0.001,
    ),
)
assert report.compliant
```

Execution latency is `finish_s - actual_start_s`. The first tick has zero
jitter; later ticks use
`abs((actual_start_s[i] - actual_start_s[i-1]) - sample_period_s)`. Jitter at
or below the configured budget is recorded as zero. A tick misses when either
its latency exceeds `deadline_s` or its remaining jitter exceeds the budget,
and the loop raises once the cumulative miss count exceeds
`max_missed_deadlines`. Step metrics are copied into a read-only mapping after
finite-value validation.

`evaluate_realtime_sla` reports maximum latency and jitter, linear-interpolated
p95/p99 latency, and the miss rate. `enforce_realtime_sla` returns the same
report when compliant and raises with every breach reason otherwise.

## Sub-microsecond tracker

`SubMicrosecondTracker` records integer-nanosecond cycle samples and reports
inter-cycle jitter percentiles against a target period together with a
deadline-miss count.

```python
from scpn_quantum_control.control.realtime_runtime import (
    CycleSample,
    SubMicrosecondTracker,
)

tracker = SubMicrosecondTracker(target_rate_hz=100_000)  # 10_000 ns period
tracker.record(CycleSample(cycle_id=0, start_ns=0, end_ns=3_000, deadline_ns=5_000))
tracker.record(CycleSample(cycle_id=1, start_ns=10_200, end_ns=13_100, deadline_ns=15_200))
report = tracker.report()
# report.jitter_p99_ns, report.jitter_max_ns, report.deadline_misses, ...
```

### Definitions

- **Jitter** of cycle `i > 0` is `abs((start_ns[i] - start_ns[i-1]) - target_period_ns)`,
  where `target_period_ns = 1e9 / target_rate_hz`. The first observed cycle has
  zero jitter.
- A cycle **misses its deadline** when `end_ns > deadline_ns`.
- Recent jitter samples are held in a bounded ring of `ring_buffer_capacity`
  entries for percentile estimation. `cycles_observed` and `deadline_misses` are
  running counters and stay exact across ring overwrites; `window_size` reports
  how many jitter samples backed the percentiles.

`SubMicrosecondReport` fields: `jitter_p50_ns`, `jitter_p95_ns`,
`jitter_p99_ns`, `jitter_max_ns`, `deadline_misses`, `cycles_observed`,
`target_period_ns`, `window_size`.

### Batch summary

`summarise_cycle_samples(start_ns, end_ns, deadline_ns, *, target_rate_hz)`
computes the same report from arrays of timestamps in one pass. It is the path
used by the throughput benchmark and by callers that buffer timestamps and
summarise a window periodically.

## Acceleration and parity

Percentile and summary computation dispatch to the Rust kernel
(`scpn_quantum_engine.sub_us_jitter_percentiles`,
`sub_us_tracker_summary`) when the engine is installed, falling back to a NumPy
implementation otherwise. The Rust kernel replicates NumPy's branchful linear
interpolation (`numpy.quantile(..., method="linear")`), so the two paths are
**bit-true identical**; this is asserted over random inputs in
`tests/test_sub_us_tracker.py`.

The dispatch is deliberately tolerant of an absent engine, an older engine
without these exports, or a native input rejection: all three conditions use
the same NumPy reference path through the public tracker and batch APIs.

## Measured throughput

Per-record wall-time, median of 9 repeats, release build, produced by
`scripts/bench_sub_us_tracker.py` (artefact:
`results/sub_us_tracker_benchmark.json`).

| window size | Rust kernel | NumPy fallback | Python `record()` |
|---|---|---|---|
| 1 024 | 16.9 ns | 182.0 ns | 212.3 ns |
| 16 384 | 24.1 ns | 43.3 ns | 225.0 ns |
| 65 536 | 27.1 ns | 37.3 ns | 253.6 ns |
| 262 144 | 35.1 ns | 63.7 ns | 228.0 ns |

The Rust batch kernel is the fastest measured backend and stays under the
100 ns/record budget across all window sizes; it therefore sits at the top of
the dispatch chain. The per-call `record()` convenience path is bounded by
Python call overhead (~0.25 µs), which is a small fraction of the 10 µs cycle
budget at 100 kHz.

These figures are classified `functional_non_isolated`: they come from a shared
workstation with no reserved cores. An `isolated_affinity` figure requires a
reserved-core run on the self-hosted benchmark runner; the artefact records host
load, CPU governor, scheduling affinity, and runtime versions for traceability.

## Consumers

The tracker is consumed by downstream pulsed-control telemetry as the
microsecond-scale outer-loop complement to the FPGA-side fast path.
