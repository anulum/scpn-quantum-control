# Key Equations

## SCPN Overview

The **Self-Consistent Phenomenological Network (SCPN)** is a theoretical
framework modeling hierarchical dynamics as 16 coupled oscillators. Each
layer n has a natural frequency omega_n and interacts with other layers
through a coupling matrix K_nm. The master equation governing all 16
layers is the **Unified Phase Dynamics Equation (UPDE)**, which takes
the form of a coupled Kuramoto system with layer-specific forcing terms.

**Reference**: M. Sotek, *Self-Consistent Phenomenological Network: Layer
Dynamics and Coupling Structure*, Working Paper 27 (2025). Manuscript in
preparation.

## Kuramoto -> XY Hamiltonian

Classical Kuramoto (the UPDE core):
```
d(theta_i)/dt = omega_i + sum_j K_ij sin(theta_j - theta_i)
```

Quantum XY Hamiltonian:
```
H = -sum_{i<j} K_ij (X_i X_j + Y_i Y_j) - sum_i omega_i Z_i
```

Time evolution via Lie-Trotter decomposition:
```
U(t) = exp(-iHt) ~ [exp(-iH_zz dt) exp(-iH_z dt)]^(t/dt)
```

Order parameter from qubit expectations:
```
R = (1/N) |sum_i (<X_i> + i <Y_i>)|
```

## Knm Canonical Parameters

Natural frequencies (16 layers, Paper 27 Table 1):
```
omega = [1.329, 1.255, 1.183, 1.114, 1.048, 0.985, 1.068, 1.148,
         1.095, 1.028, 0.974, 0.929, 1.012, 0.962, 1.042, 0.991]
```

Coupling matrix (Paper 27, Eq. 3):
```
K_nm = K_base * exp(-alpha * |n - m|)  # K_base=0.45, alpha=0.3
```

Calibration anchors: K[1,2]=0.302, K[2,3]=0.201, K[3,4]=0.252, K[4,5]=0.154.
Cross-hierarchy boosts: L1-L16=0.05, L5-L7=0.15.

## UPDE (Unified Phase Dynamics Equation)

The UPDE extends the Kuramoto model with layer-specific terms:
```
d(theta_n)/dt = omega_n + sum_m K_nm sin(theta_m - theta_n) + F_n(theta, t)
```

where F_n captures layer-specific forcing (external input, feedback loops).
In the quantum mapping, F_n corresponds to single-qubit Z rotations beyond
the natural frequency term.

## Quantum LIF Neuron

Membrane dynamics (classical):
```
v(t+1) = v(t) - (dt/tau)(v(t) - v_rest) + R*I*dt
```

Rotation angle encoding:
```
theta = pi * clip((v - v_rest) / (v_threshold - v_rest), 0, 1)
```

Spike probability:
```
P(spike) = sin^2(theta/2)
```

## Quantum Synapse (CRy)

Weight-to-angle:
```
theta_w = pi * (w - w_min) / (w_max - w_min)
```

Effective transmission probability:
```
P(post|pre=1) = sin^2(theta_w / 2)
```

## Parameter-Shift STDP

Gradient of expectation:
```
d<Z>/d(theta) = [<Z>(theta + pi/2) - <Z>(theta - pi/2)] / 2
```

Weight update:
```
delta_w = lr * pre_spike * d<Z_post>/d(theta_w)
```

## QAOA Cost Hamiltonian

MPC quadratic cost -> Ising:
```
C = sum_t ||B*u(t) - target||^2
  = sum_{i,j} J_ij Z_i Z_j + sum_i h_i Z_i + const
```

QAOA circuit:
```
|gamma, beta> = prod_p [exp(-i beta_p H_mixer) exp(-i gamma_p H_cost)] |+>^n
```

## VQLS Cost Function

For linear system Ax = b:
```
C = 1 - |<b|A|x>|^2 / <x|A'A|x>
```

where |x> = U(theta)|0> is a variational ansatz.
