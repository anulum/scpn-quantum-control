import os

import numpy as np

out_dir = "params"
os.makedirs(out_dir, exist_ok=True)


def random_symmetric(n):
    A = np.random.rand(n, n)
    return (A + A.T) / 2


for n in [20, 40, 80, 160]:
    np.save(os.path.join(out_dir, f"scale_Knm_{n}x{n}.npy"), random_symmetric(n))
    np.save(os.path.join(out_dir, f"scale_omega_{n}.npy"), np.random.rand(n))

np.save(os.path.join(out_dir, "distill_Knm_12x12.npy"), random_symmetric(12))
np.save(os.path.join(out_dir, "distributed_Knm_20x20.npy"), random_symmetric(20))
np.save(os.path.join(out_dir, "tn_Knm_64x64.npy"), random_symmetric(64))
np.save(os.path.join(out_dir, "pt_Knm_12x12.npy"), random_symmetric(12))
np.save(os.path.join(out_dir, "logical_Knm_12x12.npy"), random_symmetric(12))

print("Frontier parameters generated")
