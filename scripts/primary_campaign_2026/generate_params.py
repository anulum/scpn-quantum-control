import os

import numpy as np

out_dir = "params"
os.makedirs(out_dir, exist_ok=True)


def random_symmetric(n):
    A = np.random.rand(n, n)
    return (A + A.T) / 2


np.save(os.path.join(out_dir, "primary_Knm_12x12.npy"), random_symmetric(12))
np.save(os.path.join(out_dir, "primary_omega.npy"), np.random.rand(12))
np.save(os.path.join(out_dir, "c_elegans_connectome_14x14.npy"), random_symmetric(14))

for n in [8, 12, 16, 20, 24, 32]:
    np.save(os.path.join(out_dir, f"primary_Knm_{n}x{n}.npy"), random_symmetric(n))
    np.save(os.path.join(out_dir, f"primary_omega_{n}.npy"), np.random.rand(n))

print("Primary parameters generated")
