import os
import time
import torch
import argparse
import datetime
import yaml
import glob
import psutil
import csv
import numpy as np
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import random
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

from model import EGNNEnergyRegressor, EGNNFlavourClassifier, GATClassifier
from dataset import EGNNJUNODataset, EGNNMultiJUNODataset
from loss import Losses

# --- DDP ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import get_rank
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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

def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    
    init_process_group(
        backend="nccl",
        rank=global_rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=600),
        init_method="env://"
    )

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
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
        shuffle=True, 
        drop_last=True,  
        seed=42
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
        shuffle=False,
        drop_last=False,
        seed=42
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader

def vertex_loss_huber(vhat, vtrue, delta=0.5):
    # vhat: [B,3] tensor (model output)
    # vtrue: could be list/np/tensor -> make it a tensor matching vhat
    if not isinstance(vtrue, torch.Tensor):
        vtrue = torch.as_tensor(vtrue, dtype=vhat.dtype, device=vhat.device)
    else:
        vtrue = vtrue.to(dtype=vhat.dtype, device=vhat.device)

    if vtrue.dim() == 1:               # e.g. [3] -> [1,3]
        vtrue = vtrue.unsqueeze(0)
    return torch.nn.functional.huber_loss(vhat, vtrue, delta=delta)

def energy_loss(preds, batch):
    # allow model to optionally return (preds, …)
    if isinstance(preds, (tuple, list)):
        preds = preds[0]

    # get ground truth from the batch
    if hasattr(batch, "energy"):
        true_E = batch.energy
    elif hasattr(batch, "y"):
        true_E = batch.y
    else:
        raise AttributeError("Batch has no 'energy' or 'y' field for ground-truth energy.")

    # ensure shapes/devices/dtypes line up
    pred_E = preds.view_as(true_E).to(dtype=true_E.dtype)
    true_E = true_E.to(device=preds.device, dtype=preds.dtype)

    # print(pred_E)
    # print(true_E)

    # relative MSE (change to your preferred loss if needed)
    rel = (pred_E - true_E) / true_E.clamp_min(1e-3)
    return (rel ** 2).mean()

def train_epoch(model, loader, loss_fn, optimizer, device, device_type, scaler,
                max_grad_norm=1.0):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device)

        with autocast(device_type=device_type):
            # (mu, log_var) = model(batch.x, batch.pos, batch.vertex, batch.raw_npe, batch.raw_fht, batch.edge_index, batch.batch)
            # loss = loss_fn((mu, log_var), batch)
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            pred = pred.squeeze(-1)
            loss = loss_fn(pred, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    n = len(loader)
    return total_loss / n

def validate(model, loader, loss_fn, device, device_type):
    model.eval()
    tot_loss, meds, rmses = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with autocast(device_type=device_type):
                # (mu, log_var) = model(batch.x, batch.pos, batch.vertex, batch.raw_npe, batch.raw_fht, batch.edge_index, batch.batch)
                # loss  = loss_fn((mu, log_var), batch)
                pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
                pred = pred.squeeze(-1)
                loss = loss_fn(pred, batch)

            tot_loss += loss.item()
            med, rmse = relerr_metrics(pred.detach(), batch.energy.detach())
            meds.append(med); rmses.append(rmse)
    return tot_loss/len(loader), float(np.mean(meds)), float(np.mean(rmses))

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
        train_e.append(batch.energy.cpu().numpy())
        train_phi.append(batch.direction.cpu().numpy())
        train_flav.append(batch.flavour.cpu().numpy())

    for batch in test_loader:
        test_e.append(batch.energy.cpu().numpy())
        test_phi.append(batch.direction.cpu().numpy())
        test_flav.append(batch.flavour.cpu().numpy())

    # Safe flattening using concatenate
    train_e = np.concatenate([x.flatten() for x in train_e])
    train_phi = np.concatenate([x.flatten() for x in train_phi])
    train_flav = np.concatenate([x.flatten() for x in train_flav])

    test_e = np.concatenate([x.flatten() for x in test_e])
    test_phi = np.concatenate([x.flatten() for x in test_phi])
    test_flav = np.concatenate([x.flatten() for x in test_flav])

    fig = plt.figure(figsize=(12, 12))
    ax_train_e = fig.add_subplot(211)
    ax_test_e = fig.add_subplot(212)

    latex_labels = [r"$\bar{\nu}_e$", r"$\nu_e$", r"$\bar{\nu}_{\mu}$", r"$\nu_{\mu}$", r"$nc$"]
    bin_edges = np.linspace(0, 20, 21)

    for flavour, i in flavour_map.items():
        mask_train = train_flav == i
        ax_train_e.hist(train_e[mask_train], bins=bin_edges, histtype='step', linewidth=2.5, label=latex_labels[i])

    ax_train_e.set_xlabel('Energy (GeV)')
    ax_train_e.set_ylabel('Freq')
    ax_train_e.legend()

    for flavour, i in flavour_map.items():
        mask_test = test_flav == i
        ax_test_e.hist(test_e[mask_test], bins=bin_edges, histtype='step', linewidth=2.5, label=latex_labels[i])

    ax_test_e.set_xlabel('Energy (GeV)')
    ax_test_e.set_ylabel('Freq')
    ax_test_e.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def relerr_metrics(preds, target, eps=1e-6):
    r = (preds - target) / (target + eps)
    r_abs = r.abs()
    med = torch.median(r_abs).item()
    rmse = torch.sqrt(torch.mean(r**2)).item()
    return med, rmse

def rel_huber(pred_E: torch.Tensor, true_E: torch.Tensor, eps: float = 1e-3, delta: float = 1):
    """
    Relative Huber on (pred - true)/true.
    pred_E, true_E: shape (B,) or (B,1)
    """
    if true_E.dim() > 1:
        true_E = true_E.squeeze(-1)
    if pred_E.dim() > 1:
        pred_E = pred_E.squeeze(-1)

    rel = (pred_E - true_E) / true_E.clamp_min(eps)     # (B,)
    abs_rel = rel.abs()
    quad = torch.clamp(delta - abs_rel, min=0.0)        # (B,)
    # Huber: 0.5*(x^2) for |x|<δ, else δ*(|x| - 0.5δ)
    loss = 0.5 * (rel**2) * (abs_rel <= delta) + (delta * (abs_rel - 0.5 * delta)) * (abs_rel > delta)
    return loss.mean()

def main():
    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    device_type = 'cuda'

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
    pool_levels = torch.load("utils/pmt_pooled_graph_multilevel.pt", map_location="cpu")

    logger.info(f"[Device] {device} | Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / (1024 ** 2):.2f} MB | Memory Allocated {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB")

    train_loader, val_loader = get_dataloaders(h5_path=h5_path, edge_index=edge_index, stats=stats, pos=pos, batch_size=batch_size, num_workers=num_workers, limit=limit, target=target, class_type=class_type)
    logger.info(f"[Data] train batch: {len(train_loader)} | validation batch: {len(val_loader)}")
    logger.info(f"[Data] train evt size: {len(train_loader.dataset)} | validation evt size: {len(val_loader.dataset)}")

    # logger.info(f"[Rank {get_rank()}] Starting model instantiation...")

    # raw_model = EGNNFlavourClassifier(
    raw_model = EGNNEnergyRegressor(
        in_features=2,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
        # pooled_levels=pool_levels
    ).to(device)

    # param_count = sum(p.numel() for p in raw_model.parameters())
    # logger.info(f"[Rank {get_rank()}] Finished model init with param count: {param_count}")

    # logger.info(f"[Rank {get_rank()}] Model param count: {sum(p.numel() for p in raw_model.parameters())}")

    # logger.info(f"[Rank {get_rank()}] Model architecture:\n{raw_model}")
    # logger.info(f"[Rank {get_rank()}] Model parameters:")
    # for name, param in raw_model.named_parameters():
    #     logger.info(f"  {name:40} | shape: {tuple(param.shape)} | requires_grad: {param.requires_grad} | numel: {param.numel()}")

    torch.distributed.barrier()
    model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank) #, find_unused_parameters=True)
    logger.info("[Model] Wrapped in DDP")
    
    # # If using DDP, unwrap to the real module
    # core = model.module if hasattr(model, "module") else model

    # # Collect distinct parameter lists
    # token_params = list(core.encoder.token_layer.parameters())
    # token_ids    = {id(p) for p in token_params}

    # # All encoder params EXCEPT token_layer
    # enc_params = [p for p in core.encoder.parameters() if id(p) not in token_ids]

    # head_params = list(core.head.parameters())

    # # (Optional) sanity checks
    # assert not any(id(p) in token_ids for p in enc_params), "Overlap enc/token"
    # assert len({id(p) for p in enc_params}.intersection({id(p) for p in head_params})) == 0, "Overlap enc/head"
    # assert len({id(p) for p in token_params}.intersection({id(p) for p in head_params})) == 0, "Overlap token/head"

    # # Different learning rates per group
    # lr_backbone = 1e-3
    # lr_token    = 3e-3
    # lr_head     = 3e-3

    # param_groups = [
    #     {"params": enc_params,   "lr": lr_backbone},
    #     {"params": token_params, "lr": lr_token},
    #     {"params": head_params,  "lr": lr_head},
    # ]

    # optimizer = AdamW(param_groups, weight_decay=1e-4)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience // 2)
    scaler = GradScaler(device=device_type)

    best_val_loss = float('inf')
    counter = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    time_log = []
    mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)

    logger.info(f"[Memory] GPU Allocated: {mem_alloc:.2f} MB | GPU Reserved: {mem_reserved:.2f} MB | CPU RSS: {cpu_mem:.2f} MB")
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        start = time.time()
        train_loader.sampler.set_epoch(epoch)
        # if epoch > 10:
        #     loss_fn = Losses(loss_type='huber', huber_delta=1.0)
        train_total = train_epoch(model, train_loader, loss_fn, optimizer, device, device_type, scaler)
        val_total, val_med, val_rmse = validate(model, val_loader, loss_fn, device, device_type)    
        scheduler.step(val_total)

        train_losses.append(train_total)
        val_losses.append(val_total)

        epoch_time = time.time()-start

        if get_rank() == 0:
            logger.info(f"[Epoch {epoch + 1}] "
            f"Train: {train_total:.4f} | "
            f"Val: loss {val_total:.4f} | med|ΔE|/E {val_med:.4f} | RMSE(ΔE/E) {val_rmse:.4f} | " 
            f"Time = {epoch_time:.1f}s")
        
        time_log.append(epoch_time)

        torch.save({
          'epoch': epoch,
        #   'fold' : fold,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'train_losses': train_losses,
          'test_losses': val_losses},
          f'{output_dir}/snapshots/regress/nu_mu_like/energy/epoch_{epoch}_snapshot.pth')

        if val_total < best_val_loss:
            best_val_loss = val_total
            counter = 0
            if get_rank() == 0:
                torch.save(model.state_dict(), os.path.join(output_dir, "egnn.pt"))
        else:
            counter += 1
            if counter >= patience:
                if get_rank() == 0:
                    logger.info("Early stopping triggered.")
                break

    logger.info(f"[Device] {device} | Total GPU Memory Used: {torch.cuda.device_memory_used(device) / (1024 ** 2):.2f} MB")

    if get_rank() == 0:
        peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)
        avg_epoch = np.mean(time_log)

        logger.info(f"[Post] Plotting Input Data")
        plot_input_data(train_loader, val_loader, save_path=os.path.join(output_dir, 'plots/target_hist.png'))
        logger.info(f"[Post] Saving to CSV")
        log_summary_to_csv(cfg, avg_epoch, len(train_loader.dataset), len(val_loader.dataset), peak_gpu_mem, peak_cpu_mem)
        logger.info(f"[Post] Plotting Losses")
        plot_losses(train_losses, val_losses, save_path=os.path.join(output_dir, "loss_curve_egnn.png"))

    print(f"[rank {dist.get_rank()}] reached postprocessing barrier")
    dist.barrier()
    print(f"[rank {dist.get_rank()}] tearing down DDP")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()