# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hardware Campaign Tests

import os

import numpy as np

out_dir = "/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-QUANTUM-CONTROL/scripts/hardware_campaign_2026/params"
os.makedirs(out_dir, exist_ok=True)


def random_symmetric(n):
    A = np.random.rand(n, n)
    return (A + A.T) / 2


# Test 1
np.save(os.path.join(out_dir, "tokamak_Knm_12x12.npy"), random_symmetric(12))
np.save(os.path.join(out_dir, "tokamak_omega.npy"), np.random.rand(12))

# Test 2
np.save(os.path.join(out_dir, "power_grid_europe_16x16.npy"), random_symmetric(16))
np.save(os.path.join(out_dir, "power_grid_omega.npy"), np.random.rand(16))

# Test 3
np.save(os.path.join(out_dir, "c_elegans_connectome_14x14.npy"), random_symmetric(14))

# Test 4
np.save(os.path.join(out_dir, "clock_network_16x16.npy"), random_symmetric(16))
np.save(os.path.join(out_dir, "clock_omega.npy"), np.random.rand(16))

# Test 6
np.save(os.path.join(out_dir, "thermo_Knm_12x12.npy"), random_symmetric(12))

# Test 7
np.save(os.path.join(out_dir, "hyper_Knm_pairwise_12x12.npy"), random_symmetric(12))
np.save(os.path.join(out_dir, "hyper_Knm_3body.npy"), np.random.rand(12, 12, 12))

# Test 8
np.save(os.path.join(out_dir, "metrology_Knm_12x12.npy"), random_symmetric(12))

print("Parameters generated")
