import os
import torch
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from model import GATRegressor
from dataset import JUNODataset
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

def get_eval_loader(h5_path, edge_index, stats, batch_size, limit=None, target=None):
    dataset = JUNODataset(h5_path, edge_index=edge_index, stats=stats, limit=limit, target=target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def evaluate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.batch)
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

    h5_path = '/junofs/users/ljones/py_reader/FC/nu_e/pmt_data_2.h5'
    graph = torch.load(cfg['graph'])
    stats = load_stats(cfg['stats'])
    output_dir = cfg['output']
    model_path = os.path.join(output_dir, "best_model.pt")

    batch_size = cfg['training']['batch_size']
    latent_dim = cfg['training']['latent_dim']
    hidden_dim = cfg['training']['hidden_dim']
    target = cfg['training']['target']
    limit = cfg['training']['limit']

    edge_index = graph['edge_index']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[Device] {device}")

    model = GATRegressor(
        input_dim=2,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=1,
        heads=4
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Loaded model from {model_path}")

    eval_loader = get_eval_loader(h5_path, edge_index, stats, batch_size=batch_size, limit=limit, target=target)
    preds, targets, mse, r2 = evaluate(model, eval_loader, device)

    logger.info(f"Evaluation MSE: {mse:.4f}, R2: {r2:.4f}")
    plot_predictions(preds, targets, save_path=os.path.join(output_dir, "pred_vs_true.png"))

if __name__ == "__main__":
    main()