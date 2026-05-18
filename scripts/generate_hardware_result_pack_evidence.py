#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- hardware result-pack evidence-packet wrapper
"""Command-line wrapper for hardware result-pack evidence generation."""

from __future__ import annotations

from scpn_quantum_control.hardware_result_pack_evidence import main

if __name__ == "__main__":
    raise SystemExit(main())
