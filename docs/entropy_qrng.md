# Quantum Random-Number Generation

SPDX-License-Identifier: AGPL-3.0-or-later

The `scpn_quantum_control.entropy` package provides a streaming quantum
random-number generator with the NIST SP 800-22 Revision 1a statistical test
suite and the FIPS 140-2 Annex C power-up tests. Entropy originates from Qiskit
Aer measurement circuits; bias is removed by Von Neumann debiasing.

## Streaming harness

```python
from scpn_quantum_control.entropy import QRNGStream

qrng = QRNGStream("bell_pair", register_qubits=64, debias=True, seed=2026)
bits = qrng.sample(40_000)            # uint8 array of exactly 40 000 bits (0/1)
report = qrng.health_check()          # FIPS + NIST + entropy diagnostics
assert report.healthy
for chunk in qrng.stream(1_000):      # indefinite fixed-size chunks
    ...
    break
```

`QRNGStream.health_check` returns an `EntropyHealthReport` carrying the FIPS
verdict, a subset of NIST P-values, the Shannon and min-entropy per bit, and the
measured bit rate.

## Quantum entropy sources

`AerQuantumEntropySource` exposes three measurement circuits:

| Source | Circuit | Aer method |
|---|---|---|
| `xy_measurement` | Hadamard register measured in Z | stabilizer |
| `bell_pair` | one member of each `|Φ⁺⟩` pair read out | stabilizer |
| `phase_estimation` | Hadamard test with controlled-phase kickback | statevector |

The two Clifford circuits use Aer's exact stabilizer simulator and scale to
large registers; `phase_estimation` is non-Clifford and its register is capped
to keep the statevector tractable. Entropy model reference: Bell, Pironio,
Christensen et al., *Device-independent randomness from a single measurement on
a Bell state*, Physical Review Letters 121, 100403 (2018).

## NIST SP 800-22 Revision 1a

All fifteen tests are implemented per the publication and return a
`NistTestResult` (name, P-value(s), pass/fail at `alpha`, statistic, details):
frequency (monobit), block frequency, runs, longest run of ones, binary matrix
rank, discrete Fourier transform (spectral), non-overlapping and overlapping
template matching, Maurer's universal, linear complexity (Berlekamp-Massey),
serial, approximate entropy, cumulative sums, random excursions, and random
excursions variant. The complementary error function and incomplete gamma
function come from SciPy.

Correctness is validated against the publication's worked-example P-values
(frequency `0.527089`, block frequency `0.801252`, serial `0.808792`/`0.670320`,
approximate entropy `0.261961`, cumulative sums `0.4116588`) and against known
references for the Berlekamp-Massey and GF(2)-rank sub-components, plus a
full-suite behavioural check that passes on cryptographic-random input and
rejects biased and periodic input.

## FIPS 140-2 Annex C

`fips_140_2_tests` runs the monobit, poker, runs, and long-run power-up tests on
a fixed 20 000-bit window with the published acceptance intervals;
`enforce_fips_140_2` fails closed on the first violation.

## Acceleration

The linear-complexity hot path (Berlekamp-Massey, an O(n²) loop) dispatches to a
Rust kernel that returns the exact LFSR length and is therefore bit-true with
the NumPy reference; the P-value special functions stay in SciPy. The monobit,
runs, and per-block longest-run statistics also have Rust kernels.

Measured per-block wall-time (release build, median of 7, produced by
`scripts/bench_qrng_entropy.py`, artefact `results/qrng_entropy_benchmark.json`,
`functional_non_isolated`):

| block size | Python | Rust | speed-up |
|---|---|---|---|
| 500 | 29.6 ms | 0.11 ms | 265× |
| 2 000 | 363.8 ms | 1.60 ms | 227× |
| 5 000 | 2 290 ms | 7.04 ms | 325× |

These figures come from a shared workstation with no reserved cores; an
`isolated_affinity` figure requires a reserved-core run on the self-hosted
benchmark runner.

## Consumers

The harness supplies entropy to downstream pulsed-control formal-verification
fuzzing campaigns and stochastic Petri-net seeds.
