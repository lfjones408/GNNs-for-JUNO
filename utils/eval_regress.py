import os
import torch
import argparse
import yaml
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm

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
    energy = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            preds_all.append(preds.cpu().numpy())
            targets_all.append(batch.y.cpu().numpy())
            energy.append(batch.energy.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    mse = mean_squared_error(targets_all, preds_all)
    r2 = r2_score(targets_all, preds_all)

    return preds_all, targets_all, mse, r2, energy

def plot_predictions(y_pred, y_true, name="Energy", units="GeV", plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)

    # 2D Histogram: True vs Predicted
    plt.figure(figsize=(16, 10))
    plt.hist2d(y_true, y_pred, bins=20, cmap='viridis')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel(f"True {name} ({units})")
    plt.ylabel(f"Predicted {name} ({units})")
    plt.title(f"2D Histogram: Predicted vs. True {name}")
    plt.colorbar(label='Count')
    plt.grid(True)
    plt.savefig(f"{plot_dir}/pred_vs_true_{name}.pdf")
    plt.clf()

    # Differences PDF
    residuals = y_true - y_pred
    mu, sigma = norm.fit(residuals)

    x = np.linspace(np.min(residuals), np.max(residuals), 1000)

    plt.figure(figsize=(16, 10))
    plt.hist(residuals, bins=100, density=True, color='red', alpha=0.7)
    plt.plot(x, norm.pdf(x, mu, sigma), color='blue')
    plt.xlabel(f"{name} True - Predicted ({units})")
    plt.ylabel("Probability Density")
    plt.title(f"Residual Distribution ({name})")
    plt.text(
        0.95, 0.95,
        f"μ = {mu:.2f}\nσ = {sigma:.2f}",
        transform=plt.gca().transAxes,
        ha='right', va='top',
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    plt.grid(True)
    plt.savefig(f"{plot_dir}/residuals_{name}.pdf")
    plt.clf()

def plot_predictions_by_energy_bin(y_pred, y_true, energy_true, bins, name="Energy", units="MeV", plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)

    bin_indices = np.digitize(energy_true, bins)

    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if not np.any(bin_mask):
            continue  # Skip empty bins

        bin_pred = y_pred[bin_mask]
        bin_true = y_true[bin_mask]

        bin_label = f"{bins[i-1]:.1f}-{bins[i]:.1f} {units}"

        # === Residuals ===
        residuals = bin_true - bin_pred
        mu, sigma = norm.fit(residuals)

        x = np.linspace((mu - sigma*5), (mu + sigma*5), 1000)

        plt.figure(figsize=(16, 10))
        plt.hist(residuals, bins=100, density=True, color='red', alpha=0.7)
        plt.plot(x, norm.pdf(x, mu, sigma), color='blue')
        plt.xlabel(f"{name} True - Predicted ({units})")
        plt.ylabel("Probability Density")
        plt.title(f"{name} Residuals ({bin_label})")
        plt.text(
            0.95, 0.95,
            f"μ = {mu:.2f}\nσ = {sigma:.2f}",
            transform=plt.gca().transAxes,
            ha='right', va='top',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        plt.grid(True)
        plt.savefig(f"{plot_dir}/residuals_{i}.pdf")
        plt.clf()

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    with open('evaluation_files_nu_e.txt', 'r') as path:
        h5_path = [line.strip() for line in path]

    graph = torch.load(cfg['graph'])
    stats = load_stats(cfg['stats'])
    output_dir = cfg['output']
    model_path = os.path.join(output_dir, "snapshots/regress/nu_e_like/energy/epoch_53_snapshot.pth")

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

    snapshot = torch.load(model_path, map_location=device)

    state_dict = snapshot['model_state_dict']

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    logger.info(f"Loaded model from {model_path}")

    eval_loader = get_eval_loader(h5_path, edge_index, stats, pos,batch_size=batch_size, limit=limit, target=target)
    preds, targets, mse, r2, energy = evaluate(model, eval_loader, device)

    energy = np.concatenate([x.flatten() for x in energy])
    energy_bins = np.linspace(1, 20, 20)

    logger.info(f"Evaluation MSE: {mse:.4f}, R2: {r2:.4f}")
    plot_predictions(preds, targets, plot_dir=os.path.join(output_dir, "plots"))
    plot_predictions_by_energy_bin(preds, targets, energy, bins=energy_bins, units='GeV', plot_dir=os.path.join(output_dir, "plots/nu_e/energy"))

if __name__ == "__main__":
    main()