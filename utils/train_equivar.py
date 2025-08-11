import os
import time
import torch
import argparse
import yaml
import glob
import psutil
import csv
import numpy as np
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import random
from torch.amp import autocast, GradScaler

from model import EGNNEnergyRegressor, EGNNFlavourClassifier
from dataset import EGNNJUNODataset, EGNNMultiJUNODataset
from loss import Losses

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_stats(stats_path):
    stats = np.load(stats_path)
    return {k: stats[k].item() for k in stats}

def get_dataloaders(h5_path, edge_index, pos, stats, batch_size, val_split=0.2, limit=None, num_workers=2, target=None, class_type=None, preload=False, device='cpu'):
    if isinstance(h5_path, list):
        file_list = h5_path
    elif os.path.isdir(h5_path):
        file_list = sorted(glob.glob(os.path.join(h5_path, '*.h5')))
    elif isinstance(h5_path, str) and h5_path.endswith('.h5'):
        file_list = [h5_path]
    else:
        raise ValueError("Invalid h5_path format.")

    if len(file_list) > 1:
        dataset = EGNNMultiJUNODataset(
            file_paths=file_list,
            edge_index=edge_index,
            pos=pos,
            stats=stats,
            limit_per_file=limit,
            preload=preload,
            target=target,
            class_type=class_type,
            device=device
        )
    else:
        dataset = EGNNJUNODataset(
            h5_path=file_list[0],
            edge_index=edge_index,
            pos=pos,
            stats=stats,
            limit=limit,
            preload=preload,
            target=target
        )


    train_len = int((1 - val_split) * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader

def train_epoch(model, loader, loss_fn, optimizer, device, device_type, scaler, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        with autocast(device_type=device_type):
            preds = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            loss = loss_fn(preds, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device, device_type):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with autocast(device_type=device_type):
                preds = model(batch.x, batch.pos, batch.edge_index, batch.batch)
                loss = loss_fn(preds, batch)
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

def log_summary_to_csv(cfg, avg_time, train_size, test_size, peak_gpu_mem, peak_cpu_mem, csv_path="utils/job_logs/training_summary.csv"):

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Date/Time", "Input", "Batch Size", "Learning Rate", "Latent Dim", "Hidden Dim",
                "Num Epochs", "Num Workers", "Loss Type", "Target",
                "Train Size", "Test Size",
                "Avg Epoch Time (s)", "Peak GPU Mem (MB)", "Peak CPU Mem (MB)"
            ])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            cfg['input'],
            cfg['training']['batch_size'],
            cfg['training']['lr'],
            cfg['training']['latent_dim'],
            cfg['training']['hidden_dim'],
            cfg['training']['epochs'],
            cfg['training']['num_workers'],
            cfg['training']['loss'],
            cfg['training']['target'],
            train_size,
            test_size,
            round(avg_time, 2),
            round(peak_gpu_mem, 2),
            round(peak_cpu_mem, 2)
        ])

def plot_input_data(train_loader, test_loader, save_path='plots/target_hist.png'):
    train_e, train_phi, train_flav = [], [], []
    test_e, test_phi, test_flav = [], [], []
    flavour_map = {
                    "antinu_e": 0, "nu_e": 1,
                    "antinu_mu": 2, "nu_mu": 3,
                    "nc": 4
                }

    for batch in train_loader:
        train_e.append(batch.energy.numpy())
        train_phi.append(batch.direction.numpy())
        train_flav.append(batch.flavour.numpy())

    for batch in test_loader:
        test_e.append(batch.energy.numpy())
        test_phi.append(batch.direction.numpy())
        test_flav.append(batch.flavour.numpy())

    train_e = np.array(train_e).flatten()
    train_phi = np.array(train_phi).flatten()
    train_flav = np.array(train_flav).flatten()

    test_e = np.array(test_e).flatten()
    test_phi = np.array(test_phi).flatten()
    test_flav = np.array(test_flav).flatten()

    for flavour, label in flavour_map.items():
        globals()[f"mask_{flavour}_training"] = train_flav == label
        globals()[f"mask_{flavour}_test"] = test_flav == label

    fig = plt.figure(figsize=(12,12))
    ax_train_e   = fig.add_subplot(211)
    ax_test_e    = fig.add_subplot(212)

    latex_labels = [r"$\bar{\nu}_e$", r"$\nu_e$", r"$\bar{\nu}_{\mu}$", r"$\nu_{\mu}$", r"$nc$"]

    bin_edges = np.linspace(0, 20, 21)

    for flavour, i in flavour_map.items():
        mask_train = globals()['mask_'+flavour+'_training']
        ax_train_e.hist(train_e[mask_train], bins=bin_edges, histtype='step', linewidth=2.5, label=latex_labels[i])
    ax_train_e.set_xlabel('Energy (GeV)')
    ax_train_e.set_ylabel('Freq')
    ax_train_e.legend()

    for flavour, i in flavour_map.items():
        mask_test = globals()['mask_'+flavour+'_test']
        ax_test_e.hist(test_e[mask_test], bins=bin_edges, histtype='step', linewidth=2.5, label=latex_labels[i])
    ax_test_e.set_xlabel('Energy (GeV)')
    ax_test_e.set_ylabel('Freq')
    ax_test_e.legend()

    plt.savefig(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    with open(cfg['input'], 'r') as path:
        h5_path = [line.strip() for line in path]

    graph = torch.load(cfg['graph'])
    stats = load_stats(cfg['stats'])

    edge_index = graph['edge_index']
    pos = graph['pmt_positions']
    batch_size = cfg['training']['batch_size']
    latent_dim = cfg['training']['latent_dim']
    hidden_dim = cfg['training']['hidden_dim']
    lr = cfg['training']['lr']
    patience = cfg['training']['patience']
    epochs = cfg['training']['epochs']
    limit = cfg['training']['limit']
    num_workers = cfg['training']['num_workers']
    loss_fn = Losses(loss_type=cfg['training']['loss'])
    class_type = cfg['training']['class_type']
    target = cfg['training']['target']
    output_dir = cfg['output']
    os.makedirs(output_dir, exist_ok=True)

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    logger.info(f"[Device] {device} | Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 2):.2f} MB | Memory Allocated {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB")

    train_loader, val_loader = get_dataloaders(h5_path=h5_path, edge_index=edge_index, stats=stats, pos=pos, batch_size=batch_size, num_workers=num_workers, limit=limit, target=target, class_type=class_type)
    logger.info(f"[Data] train batch: {len(train_loader)} | validation batch: {len(val_loader)}")
    logger.info(f"[Data] train evt size: {len(train_loader.dataset)} | validation evt size: {len(val_loader.dataset)}")

    plot_input_data(train_loader, val_loader, save_path=os.path.join(output_dir, 'plots/target_hist.png'))

    model = EGNNFlavourClassifier(
        in_features=2,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_classes= 3 if class_type=='3-label' else 5
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience // 2)
    scaler = GradScaler(device=device_type)

    best_val_loss = float('inf')
    counter = 0
    train_losses, val_losses = [], []

    time_log = []
    mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)

    logger.info(f"[Memory] GPU Allocated: {mem_alloc:.2f} MB | GPU Reserved: {mem_reserved:.2f} MB | CPU RSS: {cpu_mem:.2f} MB")
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        start = time.time()
        # torch.cuda.empty_cache()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, device_type, scaler, max_grad_norm=1.0)
        val_loss = validate(model, val_loader, loss_fn, device, device_type)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time()-start
        logger.info(f"[Epoch {epoch+1}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        time_log.append(epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "egnn.pt"))
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"[Device] {device} | Total GPU Memory Used: {torch.cuda.device_memory_used(device) / (1024 ** 2):.2f} MB")

    peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)
    avg_epoch = np.mean(time_log)

    log_summary_to_csv(cfg, avg_epoch, len(train_loader.dataset), len(val_loader.dataset), peak_gpu_mem, peak_cpu_mem)
    plot_losses(train_losses, val_losses, save_path=os.path.join(output_dir, "loss_curve_egnn.png"))

if __name__ == "__main__":
    main()