import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

class JUNODataset(Dataset):
    def __init__(self, h5_path, edge_index, limit=None, preload=False, stats=None, target=None):
        self.h5_path = h5_path
        self.edge_index = edge_index
        self.limit = int(limit) if limit is not None else None
        self.preload = preload
        self.target = target

        self.npe = None
        self.fht = None
        self._length = None

        self._load_data()

        if stats is not None:
            self.npe_mean = stats['npe_mean']
            self.npe_std = stats['npe_std']
            self.fht_mean = stats['fht_mean']
            self.fht_std = stats['fht_std']
        else:
            self._compute_normalization_stats()

    def _load_data(self):
        with h5py.File(self.h5_path, 'r') as f:
            npe_raw = np.log1p(f['npe'][()])
            fht_raw = f['fht'][()]
            labels_raw = f['labels'][()][1:4]

            if self.limit is not None:
                npe_raw = npe_raw[:self.limit]
                fht_raw = fht_raw[:self.limit]
                labels_raw = labels_raw[:self.limit] 

            if self.preload:
                self.npe = npe_raw
                self.fht = fht_raw
                self.labels = labels_raw 

            self._length = len(fht_raw)

    def _compute_normalization_stats(self):
        if self.npe is not None and self.fht is not None:
            npe_raw = self.npe
            fht_raw = self.fht
        else:
            with h5py.File(self.h5_path, 'r') as f:
                npe_raw = np.log1p(f['npe'][()])
                fht_raw = f['fht'][()]
                if self.limit is not None:
                    npe_raw = npe_raw[:self.limit]
                    fht_raw = fht_raw[:self.limit]

        self.npe_mean = npe_raw.mean()
        self.npe_std = npe_raw.std()
        self.fht_mean = fht_raw.mean()
        self.fht_std = fht_raw.std()

    def __len__(self):
        return self._length if self.limit is None else min(self._length, self.limit)

    def __getitem__(self, idx):
        if self.preload:
            npe = self.npe[idx]
            fht = self.fht[idx]
            label = self.labels[idx]
        else:
            with h5py.File(self.h5_path, 'r') as f:
                npe = np.log1p(f['npe'][idx])
                fht = f['fht'][idx]
                label = f['labels'][idx]

        npe = (npe - self.npe_mean) / self.npe_std
        fht = (fht - self.fht_mean) / self.fht_std
        features = np.stack((npe, fht), axis=1)
        x = torch.tensor(features, dtype=torch.float32)
        edge_index = self.edge_index

        label_tensor = torch.tensor(label[1:4], dtype=torch.float32)

        if self.target=='direction':
            label_tensor = label_tensor[0]
        elif self.target=='flavour':
            label_tensor = label_tensor[1]
        elif self.target=='energy':
            label_tensor = label_tensor[2]
        else:
            print('Invalid Target!')
 
        return Data(x=x, edge_index=edge_index, y=label_tensor)


class MultiJUNODataset(Dataset):
    def __init__(self, file_paths, edge_index, limit_per_file=None, preload=False, device='cpu', stats=None):
        self.file_paths = sorted(file_paths)
        self.edge_index = edge_index.to(device)
        self.limit = limit_per_file
        self.device = device
        self.preload = preload

        self.npe = []
        self.fht = []
        self.labels = []

        self.file_handles = [None] * len(file_paths)
        self.cumulative_lengths = [0]

        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                length = len(f['fht']) if self.limit is None else min(len(f['fht']), self.limit)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)

                if self.preload:
                    self.npe.append(np.log1p(f['npe'][:length]))
                    self.fht.append(f['fht'][:length])
                    self.labels.append(f['labels'][:length])

        if self.preload:
            self.npe = np.concatenate(self.npe, axis=0)
            self.fht = np.concatenate(self.fht, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)

        if stats:
            self.npe_mean = stats['npe_mean']
            self.npe_std = stats['npe_std']
            self.fht_mean = stats['fht_mean']
            self.fht_std = stats['fht_std']
        else:
            self._compute_global_stats()

    def _compute_global_stats(self):
        if self.preload:
            npe = self.npe
            fht = self.fht
        else:
            npe = []
            fht = []
            for path in tqdm(self.file_paths, desc="Calculating mean/std"):
                with h5py.File(path, 'r') as f:
                    length = len(f['fht']) if self.limit is None else min(len(f['fht']), self.limit)
                    npe.append(np.log1p(f['npe'][:length]))
                    fht.append(f['fht'][:length])
            npe = np.concatenate(npe)
            fht = np.concatenate(fht)

        self.npe_mean = npe.mean()
        self.npe_std = npe.std()
        self.fht_mean = fht.mean()
        self.fht_std = fht.std()

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _get_file_position(self, global_idx):
        file_idx = np.searchsorted(self.cumulative_lengths, global_idx, side='right') - 1
        local_idx = global_idx - self.cumulative_lengths[file_idx]
        return file_idx, local_idx

    def __getitem__(self, idx):
        if self.preload:
            npe = (self.npe[idx] - self.npe_mean) / self.npe_std
            fht = (self.fht[idx] - self.fht_mean) / self.fht_std
            label = self.labels[idx][1:4]
        else:
            file_idx, local_idx = self._get_file_position(idx)
            if self.file_handles[file_idx] is None:
                self.file_handles[file_idx] = h5py.File(self.file_paths[file_idx], 'r')
            f = self.file_handles[file_idx]
            npe = np.log1p(f['npe'][local_idx])
            fht = f['fht'][local_idx]
            label = f['labels'][local_idx][1:4]

            npe = (npe - self.npe_mean) / self.npe_std
            fht = (fht - self.fht_mean) / self.fht_std

        features = np.stack((npe, fht), axis=1)
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        y = torch.tensor(label, dtype=torch.float32, device=self.device)

        return Data(x=x, edge_index=self.edge_index, y=y)

    def __del__(self):
        for f in self.file_handles:
            if f is not None:
                f.close()

class TripletJUNODataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labels = []

        # Efficiently extract labels from base_dataset without triggering full __getitem__
        for i in range(len(base_dataset)):
            if hasattr(base_dataset, "labels"):
                self.labels.append(base_dataset.labels[i][1:4])  # Energy, Flavour, Dir
            else:
                self.labels.append(base_dataset[i].y.cpu().numpy())

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        anchor = self.base_dataset[idx]
        anchor_label = self.labels[idx]

        # --- Find positive sample: same flavour, similar energy ---
        pos_candidates = [
            i for i, label in enumerate(self.labels)
            if i != idx and label[0] == anchor_label[0] and abs(label[1] - anchor_label[1]) < 0.5
        ]
        if not pos_candidates:
            # fallback to same flavour
            pos_candidates = [
                i for i, label in enumerate(self.labels)
                if i != idx and label[0] == anchor_label[0]
            ]
        if not pos_candidates:
            pos_candidates = [i for i in range(len(self.base_dataset)) if i != idx]

        positive = self.base_dataset[random.choice(pos_candidates)]

        # --- Find negative sample: different flavour, energy difference >= 5 ---
        neg_candidates = [
            i for i, label in enumerate(self.labels)
            if label[0] != anchor_label[0] and abs(label[1] - anchor_label[1]) >= 5
        ]
        if not neg_candidates:
            # fallback to any different flavour
            neg_candidates = [
                i for i, label in enumerate(self.labels)
                if label[0] != anchor_label[0]
            ]
        if not neg_candidates:
            neg_candidates = [i for i in range(len(self.base_dataset)) if i != idx]

        negative = self.base_dataset[random.choice(neg_candidates)]

        return anchor, positive, negativ

class EGNNJUNODataset(Dataset):
    def __init__(self, h5_path, edge_index, pos, limit=None, preload=False, stats=None, target=None):
        self.h5_path = h5_path
        self.edge_index = edge_index
        self.pos = pos.detach().clone().float()
        self.limit = int(limit) if limit is not None else None
        self.preload = preload
        self.target = target

        self.npe = None
        self.fht = None
        self._length = None

        self._load_data()

        if stats is not None:
            self.npe_mean = stats['npe_mean']
            self.npe_std = stats['npe_std']
            self.fht_mean = stats['fht_mean']
            self.fht_std = stats['fht_std']
        else:
            self._compute_normalization_stats()

    def _load_data(self):
        with h5py.File(self.h5_path, 'r') as f:
            npe_raw = np.log1p(f['npe'][()])
            fht_raw = f['fht'][()]
            labels_raw = f['labels'][()][1:4]

            if self.limit is not None:
                npe_raw = npe_raw[:self.limit]
                fht_raw = fht_raw[:self.limit]
                labels_raw = labels_raw[:self.limit]

            if self.preload:
                self.npe = npe_raw
                self.fht = fht_raw
                self.labels = labels_raw

            self._length = len(fht_raw)

    def _compute_normalization_stats(self):
        if self.npe is not None and self.fht is not None:
            npe_raw = self.npe
            fht_raw = self.fht
        else:
            with h5py.File(self.h5_path, 'r') as f:
                npe_raw = np.log1p(f['npe'][()])
                fht_raw = f['fht'][()]
                if self.limit is not None:
                    npe_raw = npe_raw[:self.limit]
                    fht_raw = fht_raw[:self.limit]

        self.npe_mean = npe_raw.mean()
        self.npe_std = npe_raw.std()
        self.fht_mean = fht_raw.mean()
        self.fht_std = fht_raw.std()

    def __len__(self):
        return self._length if self.limit is None else min(self._length, self.limit)

    def __getitem__(self, idx):
        if self.preload:
            npe = self.npe[idx]
            fht = self.fht[idx]
            label = self.labels[idx]
        else:
            with h5py.File(self.h5_path, 'r') as f:
                npe = np.log1p(f['npe'][idx])
                fht = f['fht'][idx]
                label = f['labels'][idx]

        npe = (npe - self.npe_mean) / self.npe_std
        fht = (fht - self.fht_mean) / self.fht_std
        features = np.stack((npe, fht), axis=1)

        x = torch.tensor(features, dtype=torch.float32)
        edge_index = self.edge_index
        pos = self.pos  # shared across all samples

        label_tensor = torch.tensor(label[1:4], dtype=torch.float32)

        if self.target == 'direction':
            label_tensor = label_tensor[0]
        elif self.target == 'flavour':
            label_tensor = label_tensor[1]
        elif self.target == 'energy':
            label_tensor = label_tensor[2]
        else:
            raise ValueError('Invalid target label.')

        return Data(x=x, pos=pos, edge_index=edge_index, y=label_tensor)