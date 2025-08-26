# utils/build_pooled_graph_multilevel.py
import torch
from torch_geometric.nn import fps, knn, knn_graph
from torch_scatter import scatter_add

def build_level(pos_prev, keep_ratio=0.25, k_knn=24):
    """
    pos_prev: (M_prev, 3) positions for previous level
    Returns a dict with mapping to the next level and its geometry.
    """
    device = pos_prev.device
    M_prev = pos_prev.size(0)
    batch0 = pos_prev.new_zeros(M_prev, dtype=torch.long)  # single graph
    
    print(f"pos initial = {pos_prev}")
    print(f"pos initialshape = {pos_prev.shape}")

    # FPS centers on previous level
    centers_idx = fps(pos_prev, batch0, ratio=keep_ratio, random_start=False)  # (M_next,)
    M_next = centers_idx.numel()

    # Assign each prev-node to nearest center
    # knn with x=centers, y=all prev nodes → (row, col) with len=row=col=M_prev
    row, col = knn(
        x=pos_prev[centers_idx], y=pos_prev, k=1,
        batch_x=batch0[centers_idx], batch_y=batch0
    )
    # row is 0..M_prev-1, col is center index in [0..M_next-1]
    cluster_id = col  # shape (M_prev,)

    # Compute super-node positions as mean of assigned nodes
    ones = torch.ones(M_prev, device=device)
    counts = scatter_add(ones, cluster_id, dim=0, dim_size=M_next).clamp_min(1)
    pos_sum = torch.zeros(M_next, 3, device=device)
    pos_sum.index_add_(0, cluster_id, pos_prev)
    pos_next = pos_sum / counts.unsqueeze(1)
    print(f"pos = {pos_next}")
    print(f"pos shape = {pos_next.shape}")

    # Build edges among super-nodes
    edge_index_next = knn_graph(pos_next, k=k_knn, loop=False)

    return {
        "cluster_id": cluster_id.cpu(),      # maps prev → next
        "centers_idx": centers_idx.cpu(),    # optional
        "pos_pool": pos_next.cpu(),          # geometry for next level
        "edge_index_pool": edge_index_next.cpu(),
        "meta": {
            "M_prev": int(M_prev),
            "M_next": int(M_next),
            "keep_ratio": float(keep_ratio),
            "k_knn": int(k_knn),
        }
    }

if __name__ == "__main__":
    # Load base detector geometry
    g = torch.load("utils/pmt_graph.pt", map_location="cpu")
    pos0 = (g.get("pmt_positions")).float()  # (N0, 3)

    levels = []
    pos_prev = pos0
    num_levels = 4
    keep_ratio = 0.25
    k_knn = 24  # pick a k you’re comfortable with

    for L in range(num_levels):
        L_dict = build_level(pos_prev, keep_ratio=keep_ratio, k_knn=k_knn)
        levels.append(L_dict)
        pos_prev = L_dict["pos_pool"]  # feed next level

    torch.save(levels, "utils/pmt_pooled_graph_multilevel.pt")
    print("Saved", len(levels), "levels to utils/pmt_pooled_graph_multilevel.pt")