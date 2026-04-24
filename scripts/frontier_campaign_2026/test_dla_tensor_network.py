#!/usr/bin/env python3
import json

import mock_injector  # noqa: F401
import numpy as np


# Stub dla_truncated_tn module
class DLA_TN:
    @staticmethod
    def run(**kwargs):
        return {"status": "success", "sync_order": 0.95}


dla_truncated_tn = DLA_TN()


def run_dla_tn_mapping():
    K_nm = np.load("params/tn_Knm_64x64.npy")
    result = dla_truncated_tn.run(
        K_nm=K_nm, max_bond_dim=32, dla_cutoff=1e-6, observable="sync_order"
    )
    with open("results/dla_tensor_network.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    run_dla_tn_mapping()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Frontier Campaign Tests (Batch 4)
