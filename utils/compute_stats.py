import h5py
import numpy as np
import glob
from tqdm import tqdm

file_stats = []

for path in tqdm(glob.glob("/junofs/users/ljones/py_reader/FC/**/*.h5")):
    with h5py.File(path, 'r') as f:
        if len(f['npe']) == 0 or len(f['fht']) == 0:
            continue
        npe = np.log1p(f['npe'][:]).astype(np.float32)
        fht = f['fht'][:].astype(np.float32)

        stats = {
            'npe_mean': npe.mean(),
            'npe_std': npe.std(),
            'npe_count': npe.size,
            'fht_mean': fht.mean(),
            'fht_std': fht.std(),
            'fht_count': fht.size,
        }
        file_stats.append(stats)

# Combine stats
def combine_stats(stats_list, key):
    total_count = sum(s[f'{key}_count'] for s in stats_list)
    total_mean = sum(s[f'{key}_mean'] * s[f'{key}_count'] for s in stats_list) / total_count
    total_var = sum(
        s[f'{key}_count'] * (
            s[f'{key}_std']**2 + (s[f'{key}_mean'] - total_mean)**2
        ) for s in stats_list
    ) / total_count
    total_std = np.sqrt(total_var)
    return total_mean, total_std

npe_mean, npe_std = combine_stats(file_stats, 'npe')
fht_mean, fht_std = combine_stats(file_stats, 'fht')

# Save
np.savez("utils/norm_stats.npz", npe_mean=npe_mean, npe_std=npe_std,
                             fht_mean=fht_mean, fht_std=fht_std)

print(f"npe: mean={npe_mean:.4f}, std={npe_std:.4f}")
print(f"fht: mean={fht_mean:.4f}, std={fht_std:.4f}")