# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Constants
"""Shared numerical constants for scpn-quantum-control."""

COUPLING_SPARSITY_EPS = 1e-15  # coupling magnitudes below this treated as zero
CONCURRENCE_EPS = 1e-10  # concurrence below this treated as zero
QBER_SECURITY_THRESHOLD = 0.11  # Shor & Preskill, PRL 85 441 (2000)
VQLS_DENOMINATOR_EPS = 1e-15  # near-zero denominator guard in VQLS
WEIGHT_SPARSITY_EPS = 1e-15  # arc/weight magnitudes below this treated as zero
