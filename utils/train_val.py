import os
import time
import torch
import argparse
import yaml
import numpy as np
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from model import GATAutoencoder
from dataset import JUNODataset
from loss import Losses

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_stats(stats_path):
    stats = np.load(stats_path)
    return {k: stats[k].item() for k in stats}

def get_dataloaders(h5_path, edge_index, stats, batch_size, val_split=0.2, limit=None, num_workers=0):
    dataset = JUNODataset(h5_path, edge_index=edge_index, stats=stats, limit=limit)
    train_len = int((1 - val_split) * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        x_recon, _ = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(x_recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_recon, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(x_recon, batch)
            total_loss += loss.item()
    return total_loss / len(loader)

def plot_losses(train_losses, val_losses, save_path="plots/loss_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig(save_path)

def main():
    # --- Config ---
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
    loss_fn = Losses(loss_type = cfg['training']['loss'])
    output_dir = cfg['output']
    os.makedirs(output_dir, exist_ok=True)

    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[Device] {device}")

    # --- Data ---
    train_loader, val_loader = get_dataloaders(h5_path, edge_index, stats, batch_size=batch_size, limit=limit)
    logger.info(f"[Data] train batch: {len(train_loader)} | validation batch: {len(val_loader)}")
    logger.info(f"[Data] train evt size: {len(train_loader.dataset)} | validation evt size: {len(val_loader.dataset)}")

    # --- Model, Loss, Optimizer ---
    model = GATAutoencoder(input_dim=2, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience // 2)

    best_val_loss = float('inf')
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"[Epoch {epoch+1}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {time.time()-start:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered.")
                break

    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()