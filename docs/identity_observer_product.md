# Identity and robustness control observers

`scpn_quantum_control.identity_observer_product` maps the existing identity
robustness certificate, coherence budget, and optional CHSH witness into one
fail-closed co-design and control-stack safety decision.

## Safety contract

| Input | Existing authority | Trip behavior |
|---|---|---|
| Energy gap / transition probability | `identity.robustness` | Hold outside explicit thresholds |
| Planned-depth fidelity | `identity.coherence_budget` | Hold below fidelity or beyond budget |
| Optional CHSH witness | `identity.entanglement_witness` | Hold below threshold; abort if requested but unsupported |
| Cryptographic seal | Optional attested-result pointer only | No key or strength claim |

Thresholds are always supplied by the caller; this product does not invent a
universal identity criterion.

```python
import numpy as np

from scpn_quantum_control.identity_observer_product import (
    IdentityObserverThresholds,
    evaluate_identity_safety,
)

decision = evaluate_identity_safety(
    np.array([[0.0, 0.4], [0.4, 0.0]]),
    np.array([-0.1, 0.1]),
    thresholds=IdentityObserverThresholds(
        min_energy_gap=0.1,
        max_transition_probability=0.1,
        min_coherence_fidelity=0.5,
    ),
    planned_depth=2,
    n_qubits=2,
)
assert decision.action in {"continue", "hold", "abort"}
```

## Unsuitable interpretations

The public API lists the negative-space scenarios through
`identity_observer_unsuitable_scenarios()`. These observers are not evidence of
unbreakable identity, hardware robustness, cryptographic strength,
consciousness, personhood, or clinical state. Failure to violate CHSH also does
not establish the absence of every form of entanglement.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
