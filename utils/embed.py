import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from model import GATAutoencoder
from dataset import JUNODataset

def load_stats(stats_path):
    stats = np.load(stats_path)
    return {k: stats[k].item() for k in stats}

def extract_embeddings(
    model_path,
    h5_path,
    edge_index_path,
    stats_path,
    output_path="embeddings/latent_with_labels.npz",
    batch_size=1,
    limit=10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    graph = torch.load(edge_index_path)
    edge_index = graph['edge_index']
    stats = load_stats(stats_path)
    dataset = JUNODataset(h5_path, edge_index=edge_index, stats=stats, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = GATAutoencoder(input_dim=2, latent_dim=16, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, z = model(batch.x, batch.edge_index, batch.batch)
            embeddings.append(z.cpu())
            labels.append(batch.y.cpu())

    print(embeddings)
    print(labels)
    latent_vectors = torch.cat(embeddings).numpy()
    true_labels = torch.stack(labels).numpy()

    print(latent_vectors)
    print(true_labels)
    np.savez(output_path, embeddings=latent_vectors, labels=true_labels)
    print(f"Saved embeddings and labels to {output_path}")

if __name__ == "__main__":
    extract_embeddings(
        model_path="output/best_model.pt",
        h5_path="/junofs/users/ljones/py_reader/FC/nu_e/pmt_data_2.h5",
        edge_index_path="utils/pmt_graph.pt",
        stats_path="utils/norm_stats.npz"
    )