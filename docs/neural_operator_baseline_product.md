# Neural-Operator Forecast Baselines

SPDX-License-Identifier: AGPL-3.0-or-later

`scpn_quantum_control.neural_operator_baseline_product` composes the existing
classical neural-operator and observed-synchronisation forecast surfaces without
adding another model or solver.

## Bounded product

The report:

- verifies the committed neural-operator evidence schema, host-independent cost
  arithmetic, disabled production-claim flag, and payload digest;
- labels `training_flops` as a one-time training estimate,
  `surrogate_flops_per_query` as a per-inference estimate, and wall-clock values
  as advisory host-bounded measurements;
- issues the governed advantage-language `no_advantage_default` certificate;
- admits only committed public measurements, source-backed public replays, or
  explicit synthetic fixtures; unknown/private classifications and unsafe paths
  are refused;
- records challenge-oracle registration as fail-closed and descoped because the current
  oracle has no public classical-baseline registration API; and
- records multimodal forecasting as a design dependency, not completed wiring.

```python
from scpn_quantum_control.neural_operator_baseline_product import (
    build_neural_operator_baseline_product,
)

report = build_neural_operator_baseline_product(
    "docs/benchmarks/neural_operator_advantage.json"
)
assert report.artifact.valid
assert report.no_advantage.language_status == "no_advantage_default"
assert all(row.allowed for row in report.datasets)
```

## Claim boundary

This is a classical forecast-baseline composition. A held-out fidelity result
or arithmetic crossover does not authorise a quantum-advantage claim. It does
not accept private datasets, execute hardware forecasts, register a
challenge-oracle rank, or complete multimodal forecasting.

Serialized reports use `neural_operator_baseline_product.v2` and reject stale
schemas or altered claim boundaries.

See [Neural-Operator Advantage Study](neural_operator_advantage.md),
[Real-Data Synchronisation Forecasting](real_data_sync_forecasting.md), and the
[Advantage Language Protocol](advantage_language_protocol.md).
