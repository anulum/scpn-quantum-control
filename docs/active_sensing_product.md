# Active sensing and experimental design

`scpn_quantum_control.active_sensing_product` chooses the next synthetic scalar
observation by expected information gain, but only after the existing BL-47
policy accepts its complete shot plan. It then runs the existing S3 analytic
design protocol and maps the selection to a BL-33 observer record.

## Contract

| Surface | Behaviour |
|---|---|
| Information gain | Gaussian scalar posterior reduction in natural-log units |
| Shot budget | `hardware_safe_execution.dry_run_execution_plan` is authoritative |
| S3 evidence | Real ansatz and pulse proxy rows from `s3_design_protocol` |
| BL-33 | Immutable `ActiveSensingObserverRecord` |
| Hardware request | Refused before information/S3 evaluation |
| NV 20 T | Research-only, hardware-blocked inventory row |

```python
import numpy as np

from scpn_quantum_control.active_sensing_product import (
    demo_information_gain_candidates,
    plan_active_sensing,
)

k = np.array([[0.0, 0.4], [0.4, 0.0]])
omega = np.array([-0.1, 0.1])
plan = plan_active_sensing(
    demo_information_gain_candidates(),
    k,
    omega,
    policy_id="ci_dry_run_only",
    shots_per_observable=64,
)
assert plan.allowed
assert plan.observer is not None
assert plan.observer.hardware_execution is False
```

The score is synthetic design evidence, not a sensing-advantage measurement.
No path in this module submits a provider job. Adaptive hardware execution
requires a separate owner ticket and a surface that owns provider submission;
BL-68 deliberately does neither.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
