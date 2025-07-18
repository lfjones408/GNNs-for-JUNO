import numpy as np

stat = np.load('utils/norm_stats.npz')

print(f'npe mean: {stat["npe_mean"]}, std: {stat["npe_std"]}')
print(f'npe mean: {stat["fht_mean"]}, std: {stat["fht_std"]}')