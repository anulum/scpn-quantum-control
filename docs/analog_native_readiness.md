# Analog-Native Kuramoto Readiness

This is the S10 no-submit readiness surface for analog-native Kuramoto
backends. It records primitive accounting and provider export status
without hardware submission or analog-advantage promotion.

## Boundary

analog-native primitive accounting and provider export readiness only; no hardware submission and no analog-advantage claim

## Primitive Accounting

- Oscillators: `4`
- Native couplers: `5`
- Digital two-qubit gate baseline: `80`
- Native-to-digital primitive ratio: `0.0625`
- Fixed declared tolerance: `0.02`
- Hardware submission allowed: `False`
- analog advantage claim allowed: `False`

## Provider Readiness

| provider | platform | sdk available | can execute |
| --- | --- | ---: | ---: |
| pulser | neutral_atoms | `False` | `False` |
| bloqade | neutral_atoms | `False` | `False` |
| ibm_pulse | circuit_qed | `False` | `False` |

## Falsifier

digital Trotter compilation reaches a lower two-qubit-gate count at the same declared tolerance or provider validation fails to preserve the native coupling model

## Prerequisites
- provider SDK object construction validated in an approved emulator path
- calibrated units and coupling constraints fixed for each target platform
- digital baseline compiled with the same declared tolerance before any advantage claim
- raw provider execution records archived before hardware-performance statements

## Gate

Regenerate and compare this readiness artefact with:

```bash
scpn-bench s10-analog-native-readiness
```
