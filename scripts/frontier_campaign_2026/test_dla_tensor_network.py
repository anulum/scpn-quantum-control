#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
import json

import numpy as np

from scpn_quantum_control.analysis.dla_truncated_tn import dla_truncated_tn  # existing module


def run_dla_tn_mapping():
    K_nm = np.load("params/tn_Knm_64x64.npy")
    result = dla_truncated_tn.run(
        K_nm=K_nm, max_bond_dim=32, dla_cutoff=1e-6, observable="sync_order"
    )
    with open("results/dla_tensor_network.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    run_dla_tn_mapping()
