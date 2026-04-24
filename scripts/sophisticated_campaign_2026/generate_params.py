import os

import numpy as np

out_dir = "params"
os.makedirs(out_dir, exist_ok=True)


def random_symmetric(n):
    A = np.random.rand(n, n)
    return (A + A.T) / 2


np.save(os.path.join(out_dir, "tokamak_Knm_16x16.npy"), random_symmetric(16))
np.save(os.path.join(out_dir, "tokamak_omega.npy"), np.random.rand(16))

np.save(os.path.join(out_dir, "c_elegans_subnetwork_14x14.npy"), random_symmetric(14))
np.save(os.path.join(out_dir, "resource_Knm_12x12.npy"), random_symmetric(12))
np.save(os.path.join(out_dir, "internet_timing_20x20.npy"), random_symmetric(20))
np.save(os.path.join(out_dir, "thermo_Knm_16x16.npy"), random_symmetric(16))

np.save(os.path.join(out_dir, "hyper_pairwise.npy"), random_symmetric(12))
np.save(os.path.join(out_dir, "hyper_3body.npy"), np.random.rand(12, 12, 12))
np.save(os.path.join(out_dir, "hyper_directed.npy"), np.random.rand(12, 12))

np.save(os.path.join(out_dir, "logical_Knm_12x12.npy"), random_symmetric(12))

print("Sophisticated parameters generated")
