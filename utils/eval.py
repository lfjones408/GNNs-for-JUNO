# evaluate.py
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import numpy as np

from model import GATAutoencoder
from dataset import JUNODataset
from utils import load_edge_index, load_stats  # Optional: utility functions

def evaluate_model(
    model_path,
    h5_path,
    edge_index_path,
    stats_path,
    batch_size=1,
    limit=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Load edge index and stats
    edge_index = torch.load(edge_index_path)
    stats = load_stats(stats_path)

    # Load dataset and model
    dataset = JUNODataset(h5_path, edge_index=edge_index, stats=stats, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = GATAutoencoder(input_dim=2, latent_dim=16, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    loss_fn = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_recon, _ = model(batch.x, batch.edge_index, batch.batch)
            target = global_mean_pool(batch.x, batch.batch)
            total_loss += loss_fn(x_recon, target).item()

    avg_loss = total_loss / len(loader)
    print(f"Average Evaluation Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    evaluate_model(
        model_path="checkpoints/best_model.pt",
        h5_path="your_data.h5",
        edge_index_path="edge_index.pt",
        stats_path="norm_stats.npz"
    )