# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — ML-DSA-65 benchmark
"""Benchmark for the FIPS 204 ML-DSA-65 signer (QUA-C.2).

* ``ntt`` — the negacyclic NTT, Rust kernel vs the pure-Python reference (the
  polyglot comparison; the NTT is the dominant lattice hot path).
* ``operations`` — key generation, signing, and verification latency.

Shared-workstation runs are ``functional_non_isolated`` evidence only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

from scpn_quantum_control.crypto import ml_dsa

_RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "ml_dsa_benchmark.json"


def _engine():
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "ml_dsa_ntt"):
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


def _ntt_rows(repeats: int) -> dict:
    engine = _engine()
    rng = random.Random(2026)
    poly = [rng.randrange(ml_dsa.Q) for _ in range(256)]
    python_ns = _median_ns(lambda: ml_dsa._ntt_python(poly), repeats)
    rust_ns = _median_ns(lambda: engine.ml_dsa_ntt(poly), repeats) if engine else None
    return {
        "python_ns": python_ns,
        "rust_ns": rust_ns,
        "speedup": (python_ns / rust_ns) if rust_ns else None,
    }


def _operation_rows(repeats: int) -> list[dict]:
    seed = bytes(range(32))
    pair = ml_dsa.key_gen(seed)
    message = b"capacitor-bank discharge authorisation"
    sig = ml_dsa.sign(pair.secret_key, message)
    rows = [
        {
            "operation": "key_gen",
            "median_ms": _median_ns(lambda: ml_dsa.key_gen(seed), repeats) / 1e6,
        },
        {
            "operation": "sign",
            "median_ms": _median_ns(lambda: ml_dsa.sign(pair.secret_key, message), repeats) / 1e6,
        },
        {
            "operation": "verify",
            "median_ms": _median_ns(lambda: ml_dsa.verify(pair.public_key, message, sig), repeats)
            / 1e6,
        },
    ]
    return rows


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
    ntt = _ntt_rows(repeats)
    operations = _operation_rows(max(3, repeats // 3))
    load_after = os.getloadavg()
    return {
        "benchmark": "ml_dsa_65",
        "evidence_class": "functional_non_isolated",
        "evidence_note": (
            "Shared-workstation run with no reserved cores. Functional/regression "
            "evidence only; an isolated_affinity figure requires a reserved-core run."
        ),
        "command": "python scripts/bench_ml_dsa.py",
        "repeats": repeats,
        "ntt": ntt,
        "operations": operations,
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
    n = result["ntt"]
    speed = f"{n['speedup']:.1f}x" if n["speedup"] else "n/a"
    print(
        f"NTT: python={n['python_ns'] / 1000:.2f}us  rust={(n['rust_ns'] or 0) / 1000:.2f}us  speedup={speed}"
    )
    for row in result["operations"]:
        print(f"  {row['operation']:8s} {row['median_ms']:.2f} ms")
    print(f"written: {_RESULT_PATH}")


if __name__ == "__main__":
    main()
