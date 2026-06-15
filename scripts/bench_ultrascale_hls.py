# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — UltraScale+ HLS codegen benchmark
"""Benchmark for the pulse → UltraScale+ HLS code generator (QUA-C.4).

* ``quantise`` — the Q-format quantisation, Rust kernel vs the pure-Python
  reference (the polyglot comparison; quantisation is the per-sample hot path).
* ``codegen`` — end-to-end ``pulse_to_vivado_hls`` latency for a 10^4-sample
  waveform (the bundle-generation acceptance target).

Shared-workstation runs are ``functional_non_isolated`` evidence only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from scpn_quantum_control.codegen import pulse_to_vivado_hls
from scpn_quantum_control.codegen import ultrascale_hls as hls

_RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "ultrascale_hls_benchmark.json"


def _engine():
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "quantise_q_format"):
            return engine
    except ImportError:
        pass
    return None


def _median_ns(fn, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1e9)
    return statistics.median(timings)


def _waveform(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.sin(np.pi * t) ** 2


def _quantise_rows(repeats: int, n_samples: int) -> dict:
    engine = _engine()
    values = _waveform(n_samples).tolist()
    python_ns = _median_ns(lambda: hls._python_quantise(values, 8, 16), repeats)
    rust_ns = (
        _median_ns(lambda: engine.quantise_q_format(values, 8, 16), repeats) if engine else None
    )
    return {
        "n_samples": n_samples,
        "python_ns": python_ns,
        "rust_ns": rust_ns,
        "speedup": (python_ns / rust_ns) if rust_ns else None,
    }


def _codegen_rows(repeats: int, n_samples: int) -> dict:
    wave = _waveform(n_samples)
    median_ms = _median_ns(lambda: pulse_to_vivado_hls(wave, 125e6, "zu3eg"), repeats) / 1e6
    return {
        "n_samples": n_samples,
        "median_ms": median_ms,
        "target_ms": 10.0,
        "within_target": median_ms <= 10.0,
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run(repeats: int) -> dict:
    load_before = os.getloadavg()
    quantise = _quantise_rows(repeats, 10_000)
    codegen = _codegen_rows(max(3, repeats // 2), 10_000)
    load_after = os.getloadavg()
    return {
        "benchmark": "ultrascale_hls",
        "evidence_class": "functional_non_isolated",
        "evidence_note": (
            "Shared-workstation run with no reserved cores. Functional/regression "
            "evidence only; an isolated_affinity figure requires a reserved-core run."
        ),
        "command": "python scripts/bench_ultrascale_hls.py",
        "repeats": repeats,
        "quantise": quantise,
        "codegen": codegen,
        "host": {
            "cpu_model": _cpu_model(),
            "cpu_count_logical": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": load_after,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "engine_available": _engine() is not None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()

    result = run(args.repeats)
    _RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print(f"evidence_class: {result['evidence_class']}")
    q = result["quantise"]
    speed = f"{q['speedup']:.1f}x" if q["speedup"] else "n/a"
    print(
        f"quantise (10^4): python={q['python_ns'] / 1e6:.3f}ms  "
        f"rust={(q['rust_ns'] or 0) / 1e6:.3f}ms  speedup={speed}"
    )
    c = result["codegen"]
    print(f"codegen  (10^4): {c['median_ms']:.3f} ms  (target ≤ {c['target_ms']} ms)")
    print(f"written: {_RESULT_PATH}")


if __name__ == "__main__":
    main()
