import numpy as np
import pandas as pd
from scipy.special import lpmv
from math import factorial

def cartesian_to_spherical(xyz: np.ndarray):
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    r = np.linalg.norm(xyz, axis=1) + 1e-12
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))          # [0, π]
    phi = np.arctan2(y, x)                                 # (-π, π]
    phi = np.where(phi < 0, phi + 2*np.pi, phi)            # [0, 2π)
    return r, theta, phi

def real_sh(L: int, theta: np.ndarray, phi: np.ndarray):
    """
    Returns real spherical harmonics up to order L (inclusive).
    Shape: [N, (L+1)^2] with ordering:
      l=0: m=0
      l=1: m=-1,0,1   (we'll emit [Y_1^-1, Y_1^0, Y_1^1], etc.)
    Real basis:
      m=0:   Y_l^0
      m>0:   sqrt(2) * N_lm * P_l^m(cosθ) * cos(mφ)
      m<0:   sqrt(2) * N_l|m| * P_l^{|m|}(cosθ) * sin(|m|φ)
    where N_lm is the usual Condon–Shortley normalization.
    """
    N = theta.shape[0]
    out = np.zeros((N, (L+1)**2), dtype=np.float64)
    ct = np.cos(theta)
    idx = 0
    for l in range(0, L+1):
        # m = 0 term
        Pl0 = lpmv(0, l, ct)  # P_l^0
        Nl0 = np.sqrt((2*l + 1)/(4*np.pi))
        out[:, idx] = Nl0 * Pl0
        idx += 1

        # m = 1..l terms (paired sin/cos)
        for m in range(1, l+1):
            Plm = lpmv(m, l, ct)  # P_l^m
            # Condon–Shortley phase is handled by lpmv
            Nlm = np.sqrt(((2*l + 1)/(4*np.pi)) * (factorial(l - m)/factorial(l + m)))
            # real basis
            out[:, idx] = np.sqrt(2) * Nlm * Plm * np.cos(m * phi)  # m>0 (cos)
            idx += 1
            out[:, idx] = np.sqrt(2) * Nlm * Plm * np.sin(m * phi)  # m<0 (sin)
            idx += 1
    return out  # [N, (L+1)^2]

def compute_sh_basis(pmt_xyz: np.ndarray, L: int = 3):
    _, theta, phi = cartesian_to_spherical(pmt_xyz)
    sh = real_sh(L, theta, phi)  # [N, (L+1)^2]
    # column-wise standardization over PMT lattice (fix across runs)
    mean = sh.mean(axis=0, keepdims=True)
    std = sh.std(axis=0, keepdims=True) + 1e-12
    sh_z = (sh - mean) / std
    return sh_z, mean.squeeze(), std.squeeze()

# --- usage ---
# pmt_xyz: [N,3] array of fixed PMT positions
# Run this once offline and save to .npz
# sh_basis: [N, 16] for L=3
# mean,std let you reproduce the same normalization later if you recompute
# (or simply save the already-normalized matrix).
# np.savez("juno_sh_L3.npz", sh=sh_basis, mean=mean, std=std)

# Load PMT Geometry
pmt_pos_file = '/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.1.3/data/Detector/Geometry/PMTPos_CD_LPMT.csv'
pmt_csv = pd.read_csv(pmt_pos_file, comment='#', sep='\s+', header=None)
pmt_csv.columns = ['CopyNo', 'X', 'Y', 'Z', 'Theta', 'Phi']
points = np.column_stack((pmt_csv['X'] * 1e-3, pmt_csv['Y'] * 1e-3, pmt_csv['Z'] * 1e-3))

sphere_harms, mu, sig = compute_sh_basis(points, L=3)

print(sphere_harms)
print(f"shape spherical harmonics: {sphere_harms.shape}")
print(mu)
print(f"shape mean: {mu.shape}")
print(sig)
print(f"shape std: {sig.shape}")

np.savez("utils/spherical_harmonics.npz", sh_l3=sphere_harms
                                        , mean_l3=mu
                                        , std_l3=sig)