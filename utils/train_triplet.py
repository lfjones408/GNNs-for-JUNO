import os
import time
import torch
import argparse
import yaml
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from model import GATAutoencoder
from dataset import JUNODataset, TripletJUNODataset
from loss import Losses

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_stats(stats_path):
    stats = np.load(stats_path)
    return {k: stats[k].item() for k in stats}


def triplet_collate_fn(batch):
    """
    Custom collate function for batching triplets (anchor, positive, negative).
    Returns batched Data objects for each role.
    """
    anchors, positives, negatives = zip(*batch)

    anchor_batch = Batch.from_data_list(anchors)
    positive_batch = Batch.from_data_list(positives)
    negative_batch = Batch.from_data_list(negatives)

    return anchor_batch, positive_batch, negative_batch

def get_dataloaders(h5_path, edge_index, stats, batch_size, limit=None):
    base_dataset = JUNODataset(h5_path, edge_index=edge_index, stats=stats, limit=limit)
    triplet_dataset = TripletJUNODataset(base_dataset)
    loader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True, collate_fn=triplet_collate_fn)
    return loader

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for anchor_batch, positive_batch, negative_batch in loader:
        optimizer.zero_grad()

        # Move batches to device
        anchor_batch = anchor_batch.to(device)
        positive_batch = positive_batch.to(device)
        negative_batch = negative_batch.to(device)

        # Get embeddings
        z_a = model.encode(anchor_batch.x, anchor_batch.edge_index, anchor_batch.batch)
        z_p = model.encode(positive_batch.x, positive_batch.edge_index, positive_batch.batch)
        z_n = model.encode(negative_batch.x, negative_batch.edge_index, negative_batch.batch)

        loss = loss_fn.triplet_loss((z_a, z_p, z_n))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    h5_path = cfg['input']
    graph = torch.load(cfg['graph'])
    stats = load_stats(cfg['stats'])
    edge_index = graph['edge_index']

    batch_size = cfg['training']['batch_size']
    latent_dim = cfg['training']['latent_dim']
    hidden_dim = cfg['training']['hidden_dim']
    lr = cfg['training']['lr']
    patience = cfg['training']['patience']
    epochs = cfg['training']['epochs']
    limit = cfg['training']['limit']
    output_dir = cfg['output']
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[Device] {device}")

    loader = get_dataloaders(h5_path, edge_index, stats, batch_size, limit)
    logger.info(f"[Data] batches: {len(loader)}")

    model = GATAutoencoder(input_dim=2, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience // 2)

    loss_fn = Losses(loss_type='triplet')

    best_loss = float('inf')
    counter = 0
    losses = []

    for epoch in range(epochs):
        start = time.time()
        train_loss = train_epoch(model, loader, loss_fn, optimizer, device)
        scheduler.step(train_loss)

        losses.append(train_loss)
        logger.info(f"[Epoch {epoch+1}] Loss: {train_loss:.4f} | Time: {time.time()-start:.1f}s")

        if train_loss < best_loss:
            best_loss = train_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # Plot
    plt.figure()
    plt.plot(losses, label='Train Triplet Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Triplet Loss Training Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'triplet_loss.png'))

if __name__ == "__main__":
    main()