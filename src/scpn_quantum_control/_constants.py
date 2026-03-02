"""Shared numerical constants for scpn-quantum-control."""

COUPLING_SPARSITY_EPS = 1e-15  # coupling magnitudes below this treated as zero
CONCURRENCE_EPS = 1e-10  # concurrence below this treated as zero
QBER_SECURITY_THRESHOLD = 0.11  # Shor & Preskill, PRL 85 441 (2000)
VQLS_DENOMINATOR_EPS = 1e-15  # near-zero denominator guard in VQLS
WEIGHT_SPARSITY_EPS = 1e-15  # arc/weight magnitudes below this treated as zero
