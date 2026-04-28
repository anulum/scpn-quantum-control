#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Frontier Campaign Tests (Batch 4)
import json

import numpy as np
from campaign_io import parameter_path, result_path

from scpn_quantum_control.analysis import dla_truncated_tn


def run_dla_tn_mapping():
    K_nm = np.load(parameter_path("tn_Knm_64x64.npy"))
    result = dla_truncated_tn(K_nm, max_bond_dim=32, dla_cutoff=1e-6)
    with open(result_path("dla_tensor_network.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    run_dla_tn_mapping()
