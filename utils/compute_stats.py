import h5py
import numpy as np
import glob
from tqdm import tqdm

file_stats = []

for path in tqdm(glob.glob("/hepstore/ljones/atm_nu/J24.1.2/FC/**/*.h5")):
    with h5py.File(path, 'r') as f:
        if "npe" not in f or "fht" not in f:
            print(f"[SKIP] {path}: missing npe or fht")
            continue
        if len(f['npe']) == 0 or len(f['fht']) == 0:
            continue
        npe           = np.log1p(f['npe'][:]).astype(np.float32)
        fht           = f['fht'][:].astype(np.float32)
        fpt           = f['fpt'][:].astype(np.float32)
        npemax        = f['npemax'][:].astype(np.float32)
        timemax       = f['timemax'][:].astype(np.float32)
        peaktime      = f['peaktime'][:].astype(np.float32)
        slope         = f['slope'][:].astype(np.float32)
        slope4        = f['slope4'][:].astype(np.float32)
        nperatio4     = f['nperatio4'][:].astype(np.float32)
        nperatio5     = f['nperatio5'][:].astype(np.float32)
        vTimeKurtosis = f['vTimeKurtosis'][:].astype(np.float32)
        vTimeSkewness = f['vTimeSkewness'][:].astype(np.float32)
        vTimeStd      = f['vTimeStd'][:].astype(np.float32)
        vTimeMean     = f['vTimeMean'][:].astype(np.float32)
        vTimeMedian   = f['vTimeMedian'][:].astype(np.float32)

        stats = {
            'npe_mean': npe.mean(),
            'npe_std': npe.std(),
            'npe_count': npe.size,
            'fht_mean': fht.mean(),
            'fht_std': fht.std(),
            'fht_count': fht.size,
            'fpt_mean': fpt.mean(),
            'fpt_std': fpt.std(),
            'fpt_count': fpt.size,
            'npemax_mean': npemax.mean(),
            'npemax_std': npemax.std(),
            'npemax_count': npemax.size,
            'timemax_mean': timemax.mean(),
            'timemax_std': timemax.std(),
            'timemax_count': timemax.size,
            'peaktime_mean': peaktime.mean(),
            'peaktime_std': peaktime.std(),
            'peaktime_count': peaktime.size,
            'slope_mean': slope.mean(),
            'slope_std': slope.std(),
            'slope_count': slope.size,
            'slope4_mean': slope4.mean(),
            'slope4_std': slope4.std(),
            'slope4_count': slope4.size,
            'nperatio4_mean': nperatio4.mean(),
            'nperatio4_std': nperatio4.std(),
            'nperatio4_count': nperatio4.size,
            'nperatio5_mean': nperatio5.mean(),
            'nperatio5_std': nperatio5.std(),
            'nperatio5_count': nperatio5.size,
            'vTimeKurtosis_mean': vTimeKurtosis.mean(),
            'vTimeKurtosis_std': vTimeKurtosis.std(),
            'vTimeKurtosis_count': vTimeKurtosis.size,
            'vTimeSkewness_mean': vTimeSkewness.mean(),
            'vTimeSkewness_std': vTimeSkewness.std(),
            'vTimeSkewness_count': vTimeSkewness.size,
            'vTimeStd_mean': vTimeStd.mean(),
            'vTimeStd_std': vTimeStd.std(),
            'vTimeStd_count': vTimeStd.size,
            'vTimeMean_mean': vTimeMean.mean(),
            'vTimeMean_std': vTimeMean.std(),
            'vTimeMean_count': vTimeMean.size,
            'vTimeMedian_mean': vTimeMedian.mean(),
            'vTimeMedian_std': vTimeMedian.std(),
            'vTimeMedian_count': vTimeMedian.size,
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

npe_mean, npe_std                     = combine_stats(file_stats, 'npe')
fht_mean, fht_std                     = combine_stats(file_stats, 'fht')
fpt_mean, fpt_std                     = combine_stats(file_stats, 'fpt')
npemax_mean, npemax_std               = combine_stats(file_stats, 'npemax')
timemax_mean, timemax_std             = combine_stats(file_stats, 'timemax')
peaktime_mean, peaktime_std           = combine_stats(file_stats, 'peaktime')
slope_mean, slope_std                 = combine_stats(file_stats, 'slope')
slope4_mean, slope4_std               = combine_stats(file_stats, 'slope4')
nperatio4_mean, nperatio4_std         = combine_stats(file_stats, 'nperatio4')
nperatio5_mean, nperatio5_std         = combine_stats(file_stats, 'nperatio5')
vTimeKurtosis_mean, vTimeKurtosis_std = combine_stats(file_stats, 'vTimeKurtosis')
vTimeSkewness_mean, vTimeSkewness_std = combine_stats(file_stats, 'vTimeSkewness')
vTimeStd_mean, vTimeStd_std           = combine_stats(file_stats, 'vTimeStd')
vTimeMean_mean, vTimeMean_std         = combine_stats(file_stats, 'vTimeMean')
vTimeMedian_mean, vTimeMedian_std     = combine_stats(file_stats, 'vTimeMedian')

# Save
np.savez("utils/norm_stats.npz",npe_mean=npe_mean, npe_std=npe_std,
                                fht_mean=fht_mean, fht_std=fht_std,
                                fpt_mean=fpt_mean, fpt_std=fpt_std,
                                npemax_mean=npemax_mean, npemax_std=npemax_std, 
                                timemax_mean=timemax_mean, timemax_std=timemax_std,
                                peaktime_mean=peaktime_mean, peaktime_std=peaktime_std,
                                slope_mean=slope_mean, slope_std=slope_std, 
                                slope4_mean=slope4_mean, slope4_std=slope4_std, 
                                nperatio4_mean=nperatio4_mean, nperatio4_std=nperatio4_std,
                                nperatio5_mean=nperatio5_mean, nperatio5_std=nperatio5_std, 
                                vTimeKurtosis_mean=vTimeKurtosis_mean, vTimeKurtosis_std=vTimeKurtosis_std,
                                vTimeSkewness_mean=vTimeSkewness_mean, vTimeSkewness_std=vTimeSkewness_std,
                                vTimeStd_mean=vTimeStd_mean, vTimeStd_std=vTimeStd_std,
                                vTimeMean_mean=vTimeMean_mean, vTimeMean_std=vTimeMean_std,
                                vTimeMedian_mean=vTimeMedian_mean, vTimeMedian_std=vTimeMedian_std
                                )

print(f"npe             : mean={npe_mean:.4f}, std={npe_std:.4f}")
print(f"fht             : mean={fht_mean:.4f}, std={fht_std:.4f}")
print(f"fpt             : mean={fpt_mean:.4f}, std={fpt_std:.4f}")
print(f"npemax          : mean={npemax_mean:.4f}, std={npemax_std:.4f}")
print(f"timemax         : mean={timemax_mean:.4f}, std={timemax_std:.4f}")
print(f"peaktime        : mean={peaktime_mean:.4f}, std={peaktime_std:.4f}")
print(f"slope           : mean={slope_mean:.4f}, std={slope_std:.4f}")
print(f"slope4          : mean={slope4_mean:.4f}, std={slope4_std:.4f}")
print(f"nperatio4       : mean={nperatio4_mean:.4f}, std={nperatio4_std:.4f}")
print(f"nperatio5       : mean={nperatio5_mean:.4f}, std={nperatio5_std:.4f}")
print(f"vTimeKurtosis   : mean={vTimeKurtosis_mean:.4f}, std={vTimeKurtosis_std:.4f}")
print(f"vTimeSkewness   : mean={vTimeSkewness_mean:.4f}, std={vTimeSkewness_std:.4f}")
print(f"vTimeStd        : mean={vTimeStd_mean:.4f}, std={vTimeStd_std:.4f}")
print(f"vTimeMean       : mean={vTimeMean_mean:.4f}, std={vTimeMean_std:.4f}")
print(f"vTimeMedian     : mean={vTimeMedian_mean:.4f}, std={vTimeMedian_std:.4f}")