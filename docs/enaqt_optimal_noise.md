# ENAQT Optimal-Noise Scan

## Abstract

This module provides a bounded local simulator for environment-assisted quantum
transport (ENAQT). It scans a finite set of local dephasing rates and maximises
the population irreversibly transferred into a target sink by a fixed time.
The committed evidence contains one disordered-chain intermediate optimum and
two negative controls. This is scenario-specific transport evidence, not a
universal noise optimum or a synchronisation, biological, hardware, or
consciousness result.

## Introduction and literature boundary

Plenio and Huelga showed that local dephasing can improve excitation transport
in selected dissipative quantum networks. Mohseni and co-authors studied the
corresponding environment-assisted transfer mechanism in a model of the FMO
photosynthetic complex. Those results motivate a transport-efficiency scan;
they do not license replacing efficiency with a phase estimator or transferring
the conclusion to Kuramoto synchronisation, BKT physics, consciousness, or a
physical noise-control policy.

- M. B. Plenio and S. F. Huelga, “Dephasing-assisted transport: quantum
  networks and biomolecules,” *New Journal of Physics* 10, 113019 (2008),
  [doi:10.1088/1367-2630/10/11/113019](https://doi.org/10.1088/1367-2630/10/11/113019),
  [arXiv:0807.4902](https://arxiv.org/abs/0807.4902).
- M. Mohseni, P. Rebentrost, S. Lloyd, and A. Aspuru-Guzik,
  “Environment-assisted quantum walks in photosynthetic energy transfer,”
  *Journal of Chemical Physics* 129, 174106 (2008),
  [doi:10.1063/1.3002335](https://doi.org/10.1063/1.3002335),
  [arXiv:0805.2741](https://arxiv.org/abs/0805.2741).

## Methodology

The model uses a single-excitation site basis augmented by orthogonal sink and
loss states. For a real symmetric hopping matrix `K` and site energies ω, the
network Hamiltonian is

\[
H = \operatorname{diag}(\omega) + K.
\]

Each site has a local dephasing jump operator with rate γ. The target has an
irreversible sink jump with rate κ, and every site may recombine into a loss
state with rate μ. The full trace-preserving Lindblad equation is propagated
with the exponential action of a matrix-free generator.

For a chosen horizon `T`, transfer efficiency is the final sink population

\[
\eta(\gamma;T)=\langle s\rvert\rho_\gamma(T)\lvert s\rangle.
\]

The scanner reports the best sampled γ, the exact zero-dephasing endpoint, the
largest-scanned-γ endpoint, and whether the best grid
point is strictly interior and exceeds both endpoints by a configured minimum.
The largest finite dephasing rate is deliberately called “high noise,” not a
classical limit.

```python
import numpy as np

from scpn_quantum_control.analysis import enaqt_scan

K = np.zeros((4, 4), dtype=np.float64)
for site in range(3):
    K[site, site + 1] = K[site + 1, site] = 1.0

result = enaqt_scan(
    K,
    np.array([0.0, 3.0, -2.0, 1.0]),
    gamma_range=np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]),
    t_evolve=10.0,
)

assert result.has_intermediate_optimum
assert result.optimal_gamma == 3.0
```

`optimal_r`, `r_values`, `coherent_r`, and `classical_r` remain read-only
compatibility aliases. They now return transport-efficiency values; the
`classical_r` name does not assert that the finite high-noise endpoint is a
classical limit. New code should use the explicit efficiency fields.

## Results

Regenerate the committed JSON and Markdown evidence with:

```bash
PYTHONPATH=src:oscillatools/src python scripts/run_enaqt_evidence.py
```

| Scenario | γ* | η(0) | η(γ*) | High-noise η | Ratio | Interior? |
|---|---:|---:|---:|---:|---:|---|
| Disordered four-site chain | 3 | 0.0522739 | 0.1765650 | 0.0114666 | 3.37769 | yes |
| Uniform three-site chain | 0 | 0.8195421 | 0.8195421 | 0.0714405 | 1 | no |
| Disconnected target | 0 | 0 | 0 | 0 | 0 | no |

The first row demonstrates the intended intermediate-noise effect on one frozen
finite model. The second shows that dephasing can be strictly detrimental. The
third checks that dephasing does not create a transport path absent from the
Hamiltonian. Every row is replayed and digest-bound in
`data/enaqt_product/enaqt_evidence.json`.

## Conclusion and control boundary

The implementation closes a local simulator and evidence lane. It does not
expose a noise setpoint controller: an optimum depends on the network, horizon,
sink/loss rates, dephasing model, and sampled grid. Using a simulated γ* to alter
a provider, QPU, laboratory noise source, biological system, or plant requires
a separately authorised and calibrated control protocol. No such protocol is
implemented here.

QFI/QNG geometry and chimera/multiscale synchronisation targets remain separate
scientific lanes. This transport scan neither consumes
their observables nor promotes a transport optimum into a geometry or
synchronisation result.
