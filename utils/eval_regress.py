import os
import torch
import argparse
import yaml
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from model import EGNNEnergyRegressor
from dataset import EGNNJUNODataset, EGNNMultiJUNODataset
from loss import Losses

from torch_geometric.loader import DataLoader

# --- Logging Setup ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utilities ---
def load_stats(stats_path):
    stats = np.load(stats_path)
    return {k: stats[k].item() for k in stats}

def get_eval_loader(h5_path, edge_index, stats, pos, batch_size, limit=None, num_workers=2, target=None, preload=False, device='cpu'):
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

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

def evaluate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            preds_all.append(preds.cpu().numpy())
            targets_all.append(batch.y.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    mse = mean_squared_error(targets_all, preds_all)
    r2 = r2_score(targets_all, preds_all)

    return preds_all, targets_all, mse, r2

def plot_predictions(preds, targets, save_path="plots/pred_vs_true.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.scatter(targets, preds, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. True")
    plt.savefig(save_path)

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    with open('evaluation_files.txt', 'r') as path:
        h5_path = [line.strip() for line in path]

    graph = torch.load(cfg['graph'])
    stats = load_stats(cfg['stats'])
    output_dir = cfg['output']
    model_path = os.path.join(output_dir, "egnn.pt")

    batch_size = cfg['training']['batch_size']
    latent_dim = cfg['training']['latent_dim']
    hidden_dim = cfg['training']['hidden_dim']
    target = cfg['training']['target']
    limit = cfg['training']['limit']

    edge_index = graph['edge_index']
    pos = graph['pmt_positions']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[Device] {device}")

    model = EGNNEnergyRegressor(
        in_features=2,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Loaded model from {model_path}")

    eval_loader = get_eval_loader(h5_path, edge_index, stats, pos,batch_size=batch_size, limit=limit, target=target)
    preds, targets, mse, r2 = evaluate(model, eval_loader, device)

    logger.info(f"Evaluation MSE: {mse:.4f}, R2: {r2:.4f}")
    plot_predictions(preds, targets, save_path=os.path.join(output_dir, "plots/pred_vs_true_egnn_energy_full.png"))

if __name__ == "__main__":
    main()