import os
import glob
import psutil
import h5py
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, random_split
from sklearn.neighbors import NearestNeighbors
from argparse import ArgumentParser
from tqdm import tqdm
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append('/junofs/users/ljones/lem_triplet/models/')
from gat_encoder import DeepGATEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 ** 2
    logger.info(f"[MEM] {stage} - RSS: {mem_mb:.2f} MB")

# Argument Parser
parser = ArgumentParser(description='Graph Autoencoder for JUNO event data reduction')
parser.add_argument('--input', type=str, required=True, help='Input .h5 file containing events')
parser.add_argument('--output', type=str, required=True, help='Output directory for embeddings')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--latent_dim', type=int, default=16, help='Latent space dimension')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--limit', type=int, default=100, help='Limit on events from h5 file')
args = parser.parse_args()

# Device setup (must be defined before model instantiation)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
log_memory_usage("After device setup")

# Utility to ensure directory exists
def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

# Load PMT Geometry
pmt_pos_file = '/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.1.3/data/Detector/Geometry/PMTPos_CD_LPMT.csv'
pmt_csv = pd.read_csv(pmt_pos_file, comment='#', sep='\s+', header=None)
pmt_csv.columns = ['CopyNo', 'X', 'Y', 'Z', 'Theta', 'Phi']
points = np.column_stack((pmt_csv['X'] * 1e-3, pmt_csv['Y'] * 1e-3, pmt_csv['Z'] * 1e-3))

logger.info('-> geometry read')
log_memory_usage("After geometry read")

# Build KNN Graph
knn = NearestNeighbors(n_neighbors=8)
knn.fit(points)
edges = knn.kneighbors_graph(mode='connectivity').tocoo()
edge_index = torch.from_numpy(np.vstack((edges.row, edges.col))).long()
edge_index = torch.cat([
    edge_index,
    edge_index[[1, 0]],  # bidirectional
    torch.arange(points.shape[0]).repeat(2, 1)  # self-loops
], dim=1)

logger.info('-> KNN Graph built')
log_memory_usage("After graph build")

# Custom Dataset Class
class JUNODataset(Dataset):
    def __init__(self, h5_path, edge_index, limit=None, preload=False):
        self.h5_path = h5_path
        self.edge_index = edge_index
        self.limit = int(limit) if limit is not None else None
        self.preload = preload

        self.npe = None
        self.fht = None
        self._length = None

        self._load_data()
        self._compute_normalization_stats()

    def _load_data(self):
        with h5py.File(self.h5_path, 'r') as f:
            npe_raw = np.log1p(f['npe'][()])
            fht_raw = f['fht'][()]

            if self.limit is not None:
                npe_raw = npe_raw[:self.limit]
                fht_raw = fht_raw[:self.limit]

            if self.preload:
                self.npe = npe_raw
                self.fht = fht_raw

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
        else:
            with h5py.File(self.h5_path, 'r') as f:
                npe = np.log1p(f['npe'][idx])
                fht = f['fht'][idx]

        npe = (npe - self.npe_mean) / self.npe_std
        fht = (fht - self.fht_mean) / self.fht_std
        features = np.stack((npe, fht), axis=1)
        x = torch.tensor(features, dtype=torch.float32)

        return Data(x=x, edge_index=self.edge_index)

# Multiple Files
class MultiJUNODataset(Dataset):
    def __init__(self, file_paths, edge_index, limit_per_file=None):
        self.file_paths = sorted(file_paths)  # List of your .h5 files
        self.edge_index = edge_index.to(device)
        self.limit = limit_per_file
        self.file_handles = [None] * len(file_paths)  # Will store open files
        
        # Build index mapping
        self.cumulative_lengths = [0]  # Start with 0
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                file_length = min(len(f['fht']), self.limit) if self.limit else len(f['fht'])
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + file_length)
        
        # Pre-compute global stats
        self._compute_global_stats()

    def _compute_global_stats(self):
        """Calculate mean/std across ALL files without loading everything"""
        npe_sum = fht_sum = count = 0
        
        for path in tqdm(self.file_paths, desc="Calculating stats"):
            with h5py.File(path, 'r') as f:
                chunk_size = min(len(f['fht']), self.limit) if self.limit else len(f['fht'])
                npe_sum += np.log1p(f['npe'][:chunk_size]).sum()
                fht_sum += f['fht'][:chunk_size].sum()
                count += chunk_size
                
        self.npe_mean = npe_sum / count
        self.fht_mean = fht_sum / count
        npe_var_sum = fht_var_sum = 0
        for path in tqdm(self.file_paths, desc="Calculating std"):
            with h5py.File(path, 'r') as f:
                chunk_size = min(len(f['fht']), self.limit) if self.limit else len(f['fht'])
                npe_chunk = np.log1p(f['npe'][:chunk_size])
                fht_chunk = f['fht'][:chunk_size]
                
                npe_var_sum += ((npe_chunk - self.npe_mean) ** 2).sum()
                fht_var_sum += ((fht_chunk - self.fht_mean) ** 2).sum()
        
        self.npe_std = np.sqrt(npe_var_sum / count)
        self.fht_std = np.sqrt(fht_var_sum / count)

    def __len__(self):
        return self.cumulative_lengths[-1]  # Total events across all files

    def _get_file_position(self, global_idx):
        """Convert global index to (file_num, local_idx)"""
        file_num = np.searchsorted(self.cumulative_lengths, global_idx, side='right') - 1
        local_idx = global_idx - self.cumulative_lengths[file_num]
        return file_num, local_idx

    def __getitem__(self, idx):
        file_num, local_idx = self._get_file_position(idx)
        
        # Lazy file opening
        if self.file_handles[file_num] is None:
            self.file_handles[file_num] = h5py.File(self.file_paths[file_num], 'r')
        
        f = self.file_handles[file_num]
        npe = (np.log1p(f['npe'][local_idx]) - self.npe_mean) / self.npe_std
        fht = (f['fht'][local_idx] - self.fht_mean) / self.fht_std
        
        x = torch.as_tensor(np.column_stack([npe, fht]), 
                          dtype=torch.float32,
                          device=device)
        return Data(x=x, edge_index=self.edge_index)

    def __del__(self):
        """Clean up file handles"""
        for f in self.file_handles:
            if f is not None: f.close()


# GAE Model
# class GraphAutoencoder(nn.Module):
#     def __init__(self, input_dim=2, latent_dim=16, hidden_dim=64, heads=4):
#         super().__init__()
#         self.encoder = DeepGATEncoder(
#             input_dim=input_dim,
#             hidden_dim=hidden_dim,
#             latent_dim=latent_dim,
#             heads=heads
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, input_dim)
#         )
#     def forward(self, x, edge_index, batch):
#         x = F.relu(self.encoder1(x, edge_index))
#         z = self.encoder2(x, edge_index)
#         z_graph = global_mean_pool(z, batch)
#         x_recon = self.decoder(z)
#         return x_recon, z_graph

class GATAutoencoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=16, heads=4):
        super().__init__()
        self.encoder = DeepGATEncoder(input_dim, hidden_dim, latent_dim, heads)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index, batch)  # returns latent graph embedding
        x_recon = self.decoder(z)
        return x_recon, z

logger.info('-> model built')
log_memory_usage("After model built")

file_paths = sorted(glob.glob("/junofs/users/ljones/py_reader/FC/nu_e/*.h5")) 

print(file_paths[:5])

# Load Dataset and Split
logger.info("-> loading dataset")
start_data = time.time()
dataset = JUNODataset(args.input, 
                      edge_index, 
                      limit=args.limit, 
                      preload=True
                      )
# dataset = MultiJUNODataset(
#     file_paths=file_paths[:5],
#     edge_index=edge_index,
# )
end_data = time.time()
logger.info(f"[TIME] Dataset load: {end_data - start_data:.2f}s")

logger.info("-> preparing data loaders")
start_loader = time.time()
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_dataset, 
                          batch_size=args.batch_size, 
                          shuffle=True,
                          num_workers=4, 
                          persistent_workers=False, 
                          pin_memory=torch.cuda.is_available()
                          )
val_loader = DataLoader(val_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False,
                        num_workers=4, 
                        persistent_workers=False, 
                        pin_memory=torch.cuda.is_available()
                        )
end_loader = time.time()
logger.info(f"[TIME] DataLoader init: {end_loader - start_loader:.2f}s")
logger.info('-> data read')
log_memory_usage('After data read')

# Model, Optimizer, Scheduler
model = GATAutoencoder(input_dim=2, latent_dim=args.latent_dim, hidden_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience//2)
loss_fn = nn.MSELoss()

logger.info('-> model, optimiser and scheduler defined')
log_memory_usage('After model, optimiser and scheduler defined')

# Training Loop
train_losses, val_losses = [], []
best_val_loss = float('inf')
counter = 0
for epoch in tqdm(range(args.epochs), desc="Training"):
    start = time.time()
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        x_recon, _ = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))
        graph_target = global_mean_pool(batch.x.to(device), batch.batch.to(device))
        loss = loss_fn(x_recon, graph_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    avg_train = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x_recon, _ = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))
            graph_target = global_mean_pool(batch.x.to(device), batch.batch.to(device))
            val_loss += loss_fn(x_recon, graph_target).item()
    avg_val = val_loss / len(val_loader)

    scheduler.step(avg_val)
    train_losses.append(avg_train)
    val_losses.append(avg_val)
    logger.info(f"Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}, Time={time.time()-start:.1f}s")
    log_memory_usage(f'After Epoch {epoch+1}')
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        counter = 0
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output, 'best_model.pt'))
    else:
        counter += 1
        if counter >= args.patience:
            logger.info("Early stopping")
            break

# Plot Training Curve
ensure_dir('plots')
plt.figure(figsize=(10,6))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/training_curve.png')

# Save Embeddings
model.load_state_dict(torch.load(os.path.join(args.output, 'best_model.pt'))['model_state_dict'])
model.eval()
embeds = []
with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=args.batch_size):
        embeds.append(model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))[1].cpu())
np.save(os.path.join(args.output, 'latent_vectors.npy'), torch.cat(embeds).numpy())
logger.info(f"Saved embeddings to {args.output}")
log_memory_usage("End of script")