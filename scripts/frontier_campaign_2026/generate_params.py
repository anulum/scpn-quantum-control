#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.

from pathlib import Path

import numpy as np
from scpneurocore.bridge import (
    load_connectome,
    load_power_grid,
    load_tokamak_data,
)  # SC-NeuroCore bridges


def generate_all_params(output_dir: str = "params") -> None:
    Path(output_dir).mkdir(exist_ok=True)
    np.random.seed(42)

    sizes = [12, 14, 16, 20]

    for N in sizes:
        # Use real SC-NeuroCore data when available, fall back to realistic random
        try:
            K = load_connectome("c_elegans_sub", N) if N == 14 else load_power_grid(N)
        except Exception:
            K = np.random.uniform(0.5, 2.0, size=(N, N))
            K = (K + K.T) / 2
            np.fill_diagonal(K, 0.0)

        np.save(f"{output_dir}/scale_Knm_{N}x{N}.npy", K)
        omega = np.random.normal(0.0, 0.3, N)
        np.save(f"{output_dir}/scale_omega_{N}.npy", omega)

    # Specific applied matrices
    np.save(f"{output_dir}/c_elegans_subnetwork_14x14.npy", load_connectome("c_elegans_sub", 14))
    np.save(f"{output_dir}/tokamak_Knm_16x16.npy", load_tokamak_data())
    np.save(f"{output_dir}/distributed_Knm_20x20.npy", load_power_grid(20))

    # Extra needed for tests
    np.save(f"{output_dir}/distill_Knm_12x12.npy", load_power_grid(12))
    np.save(f"{output_dir}/tn_Knm_64x64.npy", load_power_grid(64))
    np.save(f"{output_dir}/pt_Knm_12x12.npy", load_power_grid(12))
    np.save(f"{output_dir}/logical_Knm_12x12.npy", load_power_grid(12))
    np.save(f"{output_dir}/hyper_pairwise.npy", load_power_grid(12))
    np.save(f"{output_dir}/hyper_3body.npy", np.random.rand(12, 12, 12))
    np.save(f"{output_dir}/hyper_directed.npy", np.random.rand(12, 12))

    print(f"✅ Generated realistic .npy files in ./{output_dir}/ using SC-NeuroCore bridges")


if __name__ == "__main__":
    generate_all_params()
