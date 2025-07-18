import os
import glob
import numpy as np
import torch

def get_all_h5_files(base_dir, recursive=True, sort=True):
    """
    Recursively finds all .h5 files under base_dir.

    Args:
        base_dir (str): Path to root directory.
        recursive (bool): Whether to search subdirectories.
        sort (bool): Whether to return a sorted list.

    Returns:
        List[str]: Full paths to all .h5 files.
    """
    if recursive:
        pattern = os.path.join(base_dir, '**', '*.h5')
        files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(base_dir, '*.h5')
        files = glob.glob(pattern)

    if sort:
        files.sort()
    return files

def load_stats(path):
    stats = np.load(path)
    return {k: stats[k].item() for k in stats}

def load_edge_index(path):
    return torch.load(path)