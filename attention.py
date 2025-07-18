import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch.nn import TripletMarginLoss
from tqdm import tqdm
import random
import logging

# ---------------------- Logging Setup ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------- GCN Encoder ----------------------
class GCNEncoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=32):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, 64)
        self.gcn2 = GCNConv(64, latent_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.gcn1(x, edge_index))
        z = self.gcn2(x, edge_index)
        return global_mean_pool(z, batch)

# ---------------------- Triplet Dataset ----------------------
class TripletDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.label_to_indices = self._build_label_index()

    def _build_label_index(self):
        from collections import defaultdict
        label_map = defaultdict(list)
        for idx, data in enumerate(self.base_dataset):
            label = int(data.y.item())
            label_map[label].append(idx)
        return label_map

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        anchor = self.base_dataset[idx]
        anchor_label = int(anchor.y.item())

        positive_idx = random.choice(self.label_to_indices[anchor_label])
        while positive_idx == idx:
            positive_idx = random.choice(self.label_to_indices[anchor_label])

        negative_label = random.choice([l for l in self.label_to_indices if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])

        positive = self.base_dataset[positive_idx]
        negative = self.base_dataset[negative_idx]

        return anchor, positive, negative

# ---------------------- Training Function ----------------------
def train_triplet_model(dataset, input_dim=2, latent_dim=32, batch_size=32, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = GCNEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    loss_fn = TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    triplet_dataset = TripletDataset(dataset)
    triplet_loader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        encoder.train()
        total_loss = 0
        for anchor, positive, negative in tqdm(triplet_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()

            def embed(data):
                return encoder(data.x.to(device), data.edge_index.to(device), data.batch.to(device))

            anchor_embed = embed(anchor)
            positive_embed = embed(positive)
            negative_embed = embed(negative)

            loss = loss_fn(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}: Triplet Loss = {total_loss / len(triplet_loader):.4f}")

    return encoder

# ---------------------- Usage Example ----------------------
# from your_data_module import JUNODataset
# dataset = JUNODataset('your_file.h5', edge_index, limit=5000)
# trained_encoder = train_triplet_model(dataset)