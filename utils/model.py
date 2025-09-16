import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_min, scatter_max
from torch_cluster import fps, knn, knn_graph, radius
from torch_geometric.nn import knn_graph, graclus
from torch_geometric.nn import GATv2Conv, global_mean_pool, TopKPooling, GlobalAttention
from torch_geometric.utils import softmax, degree, to_dense_adj, dense_to_sparse


# --- GAT ---
class DeepGATEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32, heads=4):
        super().__init__()

        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.pool1 = TopKPooling(hidden_dim * heads, ratio=0.5)

        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)

        self.gat3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.bn3 = nn.BatchNorm1d(hidden_dim * heads)

        self.gat4 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.bn4 = nn.BatchNorm1d(hidden_dim * heads)

        self.gat5 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=2, concat=True)
        self.bn5 = nn.BatchNorm1d(hidden_dim * 2)
        self.pool5 = TopKPooling(hidden_dim * 2, ratio=0.2)

        self.gat6 = GATv2Conv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.bn6 = nn.BatchNorm1d(hidden_dim * 2)

        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)

        x = self.gat4(x, edge_index)
        x = self.bn4(x)
        x = F.elu(x)

        x = self.gat5(x, edge_index)
        x = self.bn5(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool5(x, edge_index, None, batch)

        x = self.gat6(x, edge_index)
        x = self.bn6(x)
        x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
 
        return x


class GATAutoencoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32, heads=4):
        super().__init__()
        self.encoder = DeepGATEncoder(input_dim, hidden_dim, latent_dim, heads)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index, batch)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x, edge_index, batch):
        """
        Return latent embeddings only, useful for contrastive or triplet training.
        """
        _, z = self.forward(x, edge_index, batch)
        return z

class GATClassifier(nn.Module):
    def __init__(self, in_features=2, hidden_dim=64, latent_dim=32, num_classes=2, heads=4, dropout=0.2):
        super().__init__()
        self.encoder = DeepGATEncoder(in_features, hidden_dim, latent_dim, heads)
        self.classifier_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index, batch)
        return self.classifier_head(z)

class GATRegressor(nn.Module):
    def __init__(self, in_features=2, hidden_dim=64, latent_dim=32, output_dim=1, heads=4, dropout=0.2):
        super().__init__()
        self.encoder = DeepGATEncoder(in_features, hidden_dim, latent_dim, heads)
        self.regressor_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index, batch)
        return self.regressor_head(z).squeeze(-1)

# --- EGNN ---
class EGNNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.1):
        super().__init__()
        # -- pre/post normalisation --
        self.pre  = nn.LayerNorm(in_features)
        self.norm = nn.LayerNorm(out_features)

        # -- MLPs --
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_features + 5, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )

        # -- residual connection --
        if in_features != out_features:
            self.res_connection = nn.Linear(in_features, out_features)
        else:
            self.res_connection = nn.Identity()

    def forward(self, x, pos, edge_index):
        x = self.pre(x)

        row, col = edge_index
        x_i, x_j = x[row], x[col]
        pos_i, pos_j = pos[row], pos[col]
        time_i, time_j = x[row][:,1], x[col][:,1]
        charge_i, charge_j = x[row][:,0], x[col][:,0]

        dist = (pos_i - pos_j)
        time_diff = (time_i - time_j).unsqueeze(1)
        charge_diff = (charge_i - charge_j).unsqueeze(1)

        edge_input = torch.cat([x_i, x_j, dist, time_diff, charge_diff], dim=1)
        edge_feat = self.edge_mlp(edge_input)

        agg = torch.zeros(x.size(0), edge_feat.size(1), device=x.device)
        agg = agg.to(edge_feat.dtype)
        agg.index_add_(0, row, edge_feat)

        node_input = torch.cat([x, agg], dim=1)
        out = self.node_mlp(node_input)

        out = self.norm(out + self.res_connection(x))
        return out, pos

class EGNNLayerWithCoord(nn.Module):
    def __init__(self, in_features, out_features,
                 dropout_prob=0.1,
                 coord_clip=0.2, init_alpha=0.1,
                 update_coords=True):
        super().__init__()
        self.update_coords = update_coords
        self.coord_clip = coord_clip

        # --- your usual EGNN MLPs (keep your versions)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_features + 3, out_features),
            nn.SiLU(), nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.SiLU(), nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.SiLU()
        )

        self.pre  = nn.LayerNorm(in_features)
        self.post = nn.LayerNorm(out_features)

        self.res_connection = (
            nn.Linear(in_features, out_features)
            if in_features != out_features else nn.Identity()
        )

        # --- coordinate path (ONLY if enabled)
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(out_features, 64), nn.SiLU(),
                nn.Linear(64, 1)
            )
            self.coord_alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        else:
            # Do not register coord params if unused:
            self.coord_mlp = None
            self.register_parameter('coord_alpha', None)

    def forward(self, x, pos, edge_index):
        row, col = edge_index

        # build your real edge features here (x_i, x_j, dist, time_diff, charge_diff, ...)
        x_i, x_j = x[row], x[col]
        pos_i, pos_j = pos[row], pos[col]
        dist_vec = pos_i - pos_j
        e = self.edge_mlp(torch.cat([x_i, x_j, dist_vec], dim=1))  # <- keep your original inputs

        # aggregate → node update
        agg = torch.zeros(x.size(0), e.size(1), device=x.device, dtype=e.dtype)
        agg.index_add_(0, row, e)
        node_in = torch.cat([x, agg], dim=1)

        x_new = self.node_mlp(node_in)
        x_out = self.post(x_new + self.res_connection(self.pre(x)))

        # coordinate update (only if enabled)
        if self.update_coords and row.numel() > 0:
            den = (dist_vec.square().sum(-1, keepdim=True).sqrt() + 1e-8)
            w   = self.coord_mlp(e)         # [E,1]
            m   = w * dist_vec / den        # [E,3]

            pos_delta = torch.zeros_like(pos)
            pos_delta.index_add_(0, row,  m)
            pos_delta.index_add_(0, col, -m)
            pos = pos + self.coord_alpha * pos_delta.clamp(-self.coord_clip, self.coord_clip)

        return x_out, pos
    
class EGNNLayerWithAttention(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_features + 5, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )
    
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * in_features + 5, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * in_features + 5, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )
        
        # self.norm = nn.BatchNorm1d(out_features)
        self.norm = nn.LayerNorm(out_features) # <- might fix nans
        
        if in_features != out_features:
            self.res_connection = nn.Linear(in_features, out_features)
        else:
            self.res_connection = nn.Identity()

    def forward(self, x, pos, edge_index):
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        pos_i, pos_j = pos[row], pos[col]
        time_i, time_j = x[row][:,1], x[col][:,1]
        charge_i, charge_j = x[row][:,0], x[col][:,0]

        dist = (pos_i - pos_j)
        time_diff = (time_i - time_j).unsqueeze(1)
        charge_diff = (charge_i - charge_j).unsqueeze(1)

        edge_input = torch.cat([x_i, x_j, dist, time_diff, charge_diff], dim=1)

        edge_feat = self.edge_mlp(edge_input)

        alpha = self.att_mlp(edge_input).squeeze(-1)   
        alpha = softmax(alpha, index=row, dim=0)  

        gate = self.gate_mlp(edge_input)

        edge_feat = edge_feat * alpha.unsqueeze(1) * gate

        agg = torch.zeros(x.size(0), edge_feat.size(1), device=x.device, dtype=edge_feat.dtype)
        agg.index_add_(0, row, edge_feat)

        node_input = torch.cat([x, agg], dim=1)
        out = self.node_mlp(node_input)

        out = self.norm(out + self.res_connection(x))
        return out, pos
    
class EGNNLayerVtx(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.1, use_vec=True):
        super().__init__()
        self.use_vec = use_vec

        self.pre  = nn.LayerNorm(in_features)
        self.norm = nn.LayerNorm(out_features)

        add_dims = 3 if use_vec else 1   # geometry r_ij or d_ij
        add_dims += 1                    # dt (from TOF)
        add_dims += 2                    # projections from vtx (proj_i, proj_j)
        add_dims += 1                    # cos_ij opening angle

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_features + add_dims, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )

        self.res_connection = (
            nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        )

    def forward(self, x, pos, vtx, tof, edge_index, batch):
        # x: (N,C), pos: (N,3), vtx: (G,3), tof: (N,1) or (N,), batch: (N,)
        x = self.pre(x)

        r_i = (pos - vtx[batch])                     # (N,3)
        r_norm = r_i.norm(dim=1, keepdim=True)       # (N,1)
        u_i = r_i / (r_norm + 1e-6)                  # (N,3)

        row, col = edge_index
        x_i, x_j       = x[row], x[col]
        pos_i, pos_j   = pos[row], pos[col]
        u_i_row, u_i_col = u_i[row], u_i[col]

        r_ij = pos_i - pos_j                         # (E,3)
        d_ij = r_ij.norm(dim=1, keepdim=True).clamp_min(1e-6)  # (E,1)

        # use TOF-corrected time ONLY
        if tof.dim() == 1:
            tof = tof.unsqueeze(1)                   # (N,1)
        dt = (tof[row] - tof[col])                   # (E,1)

        proj_i = (u_i_row * r_ij).sum(dim=1, keepdim=True)  # (E,1)
        proj_j = (u_i_col * r_ij).sum(dim=1, keepdim=True)  # (E,1)
        cos_ij = (u_i_row * u_i_col).sum(dim=1, keepdim=True).clamp(-1., 1.)  # (E,1)

        geom = r_ij if self.use_vec else d_ij

        edge_input = torch.cat([x_i, x_j, geom, dt, proj_i, proj_j, cos_ij], dim=1)  # (E, 2C + add_dims)
        edge_feat  = self.edge_mlp(edge_input)                                        # (E, out)

        agg = torch.zeros(x.size(0), edge_feat.size(1), device=x.device, dtype=edge_feat.dtype)
        agg.index_add_(0, row, edge_feat)

        out = self.node_mlp(torch.cat([x, agg], dim=1))
        out = self.norm(out + self.res_connection(x))
        return out, pos
    

class EGNNLayerAttentionVtx(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.1, use_vec=True):
        super().__init__()
        self.use_vec = use_vec

        self.pre  = nn.LayerNorm(in_features)
        self.norm = nn.LayerNorm(out_features)

        # extra edge features:
        # geometry: r_ij (3) if use_vec else d_ij (1)
        add_dims = 3 if use_vec else 1
        add_dims += 1  # dt (from TOF)
        add_dims += 2  # projections from vtx (proj_i, proj_j)
        add_dims += 1  # cos_ij opening angle

        edge_in = 2 * in_features + add_dims

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )

        # --- attention + gate (ported from your attention layer) ---
        self.att_mlp = nn.Sequential(
            nn.Linear(edge_in, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(edge_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # -----------------------------------------------------------

        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )

        self.res_connection = (
            nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        )

    def forward(self, x, pos, vtx, tof, edge_index, batch):
        # x: (N,C), pos: (N,3), vtx: (G,3), tof: (N,1) or (N,), batch: (N,)
        x = self.pre(x)

        r_i = (pos - vtx[batch])                       # (N,3)
        r_norm = r_i.norm(dim=1, keepdim=True)         # (N,1)
        u_i = r_i / (r_norm + 1e-6)                    # (N,3)

        row, col = edge_index
        x_i, x_j           = x[row], x[col]
        pos_i, pos_j       = pos[row], pos[col]
        u_i_row, u_i_col   = u_i[row], u_i[col]

        r_ij = pos_i - pos_j                           # (E,3)
        d_ij = r_ij.norm(dim=1, keepdim=True).clamp_min(1e-6)  # (E,1)

        # TOF-corrected time ONLY
        if tof.dim() == 1:
            tof = tof.unsqueeze(1)                     # (N,1)
        dt = (tof[row] - tof[col])                     # (E,1)

        proj_i = (u_i_row * r_ij).sum(dim=1, keepdim=True)  # (E,1)
        proj_j = (u_i_col * r_ij).sum(dim=1, keepdim=True)  # (E,1)
        cos_ij = (u_i_row * u_i_col).sum(dim=1, keepdim=True).clamp(-1., 1.)  # (E,1)

        geom = r_ij if self.use_vec else d_ij

        # Edge features + attention/gating
        edge_input = torch.cat([x_i, x_j, geom, dt, proj_i, proj_j, cos_ij], dim=1)  # (E, 2C + add_dims)

        edge_feat = self.edge_mlp(edge_input)                 # (E, out)

        # --- attention & gate ---
        alpha = self.att_mlp(edge_input).squeeze(-1)          # (E,)
        alpha = softmax(alpha, index=row, dim=0)              # softmax over incoming edges per target node i
        gate  = self.gate_mlp(edge_input)                     # (E,1)

        edge_feat = edge_feat * alpha.unsqueeze(1) * gate     # (E, out)
        # -----------------------

        # Aggregate to nodes i (targets of edges)
        agg = torch.zeros(x.size(0), edge_feat.size(1), device=x.device, dtype=edge_feat.dtype)
        agg.index_add_(0, row, edge_feat)                     # sum over neighbors j -> i

        out = self.node_mlp(torch.cat([x, agg], dim=1))
        out = self.norm(out + self.res_connection(x))
        return out, pos

class EGNNEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim, latent_dim):
        super().__init__()
        self.egnn1 = EGNNLayer(in_features, hidden_dim)
        self.egnn2 = EGNNLayer(hidden_dim, hidden_dim)
        self.egnn3 = EGNNLayer(hidden_dim, hidden_dim)
        self.egnn4 = EGNNLayer(hidden_dim, hidden_dim)
        self.egnn5 = EGNNLayer(hidden_dim, hidden_dim)
        self.egnn6 = EGNNLayer(hidden_dim, hidden_dim)


        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x, pos, edge_index, batch):
        x, pos = self.egnn1(x, pos, edge_index)
        x, pos = self.egnn2(x, pos, edge_index)
        x, pos = self.egnn3(x, pos, edge_index)
        x, pos = self.egnn4(x, pos, edge_index)
        x, pos = self.egnn5(x, pos, edge_index)
        x, pos = self.egnn6(x, pos, edge_index)

        # num_nodes = batch.size(0)
        # num_graphs = batch.max().item() +1
        # global_node_indices = torch.arange(num_graphs, device=batch.device) + (num_nodes - num_graphs)
        # mask = torch.ones(num_nodes, dtype=torch.bool, device=batch.device)
        # mask[global_node_indices] = False
        # x_masked = x[mask]
        # batch_masked = batch[mask]

        # pooled = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        # pooled.index_add_(0, batch_masked, x_masked)

        # counts = torch.bincount(batch_masked, minlength=num_graphs).float().unsqueeze(1)
        # x_mean = pooled / (counts + 1e-8)

        x_mean = global_mean_pool(x, batch)
        return self.lin(x_mean)

class EGNNAttentionEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim, latent_dim):
        super().__init__()
        self.egnn1 = EGNNLayerWithAttention(in_features, hidden_dim)
        self.egnn2 = EGNNLayerWithAttention(hidden_dim, hidden_dim)
        self.egnn3 = EGNNLayerWithAttention(hidden_dim, hidden_dim)
        self.egnn4 = EGNNLayerWithAttention(hidden_dim, hidden_dim)
        self.egnn5 = EGNNLayerWithAttention(hidden_dim, hidden_dim)
        self.egnn6 = EGNNLayerWithAttention(hidden_dim, hidden_dim)


        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x, pos, edge_index, batch):
        x, pos = self.egnn1(x, pos, edge_index)
        x, pos = self.egnn2(x, pos, edge_index)
        x, pos = self.egnn3(x, pos, edge_index)
        x, pos = self.egnn4(x, pos, edge_index)
        x, pos = self.egnn5(x, pos, edge_index)
        x, pos = self.egnn6(x, pos, edge_index)

        x_mean = global_mean_pool(x, batch)
        return self.lin(x_mean)

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__(); self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.0: return x
        keep = 1.0 - self.p
        mask = torch.rand(x.shape[0], 1, device=x.device) < keep  # per-node mask
        return x * mask / keep

def pool_once(x, pos, batch, keep_ratio=0.25, k=8, charge_col=0, time_col=1, eps=1e-8):
    device = x.device
    N, C = x.size()
    G = int(batch.max().item()) + 1

    # ---- (1) FPS: get indices (NOT a mask)
    centers_idx = fps(pos, batch, ratio=keep_ratio, random_start=False)  # LongTensor [M0]
    assert centers_idx.dtype == torch.long, "fps must return Long indices"

    # ---- (2) Ensure ≥1 center per graph
    batch_centers = batch[centers_idx]                    # [M0]
    present = torch.zeros(G, dtype=torch.bool, device=device)
    present.scatter_(0, batch_centers, True)
    if (~present).any():
        # grab 1 fallback node per missing graph
        missing_g = (~present).nonzero(as_tuple=False).view(-1)          # [G_missing]
        # first node of each missing graph:
        first_idx = torch.stack([ (batch == g).nonzero(as_tuple=False)[0] for g in missing_g ]).view(-1)
        centers_idx = torch.cat([centers_idx, first_idx], dim=0)
        batch_centers = batch[centers_idx]

    # ---- (3) Build a map from (global centers_idx) -> [0..M-1]
    M = centers_idx.numel()
    inv = -torch.ones(N, dtype=torch.long, device=device)  # size N, -1 default
    inv[centers_idx] = torch.arange(M, device=device)      # inv[orig_node_id] = center_id

    # 4) Assign each node to nearest center within the same graph
    col, row = knn(  # row: "center index", col: "node index"
        x=pos[centers_idx], y=pos, k=1,
        batch_x=batch_centers, batch_y=batch
    )

    # print(row)
    # print(col)

    # if row.numel():
    #     print("N,M =", pos.size(0), centers_idx.numel())
    #     print("row min/max:", int(row.min().item()), int(row.max().item()))
    #     print("Are row indices already local? ->", row.max().item() < centers_idx.numel())

    # --- Robust remap of 'row' to [0..M-1]
    # Some builds return 'row' already 0..M-1; others return GLOBAL ids.
    if row.numel() > 0 and row.max().item() >= M:
        # 'row' holds GLOBAL node ids -> map with 'inv'c
        cluster = inv[row]                # [N], now 0..M-1
    else:
        # 'row' is already local 0..M-1
        cluster = row

    # Sanity checks BEFORE scatter:
    assert cluster.numel() == pos.size(0), f"Expected one assignment per node, got {cluster.numel()} vs N={pos.size(0)}"
    bad_min = int(cluster.min().item())
    bad_max = int(cluster.max().item())
    assert bad_min >= 0, f"cluster has negative ids: {bad_min}"
    assert bad_max < M,  f"cluster max {bad_max} >= M {M}"

    # ---- (6) Counts per cluster
    ones = torch.ones(N, dtype=torch.float32, device=device)
    counts = scatter_add(ones, cluster, dim=0, dim_size=M).clamp_min(1.0)  # [M]

    # ---- (7) Charge-weight for pos mean
    w = x[:, charge_col].clamp_min(0).to(torch.float32) + eps              # [N]
    w_sum = scatter_add(w, cluster, dim=0, dim_size=M).clamp_min(eps)      # [M]

    # ---- (8) Pooled features
    x_pool = torch.zeros(M, C, device=device, dtype=x.dtype)

    # charge -> sum
    charge_sum = scatter_add(x[:, charge_col], cluster, dim=0, dim_size=M)
    x_pool[:, charge_col] = charge_sum

    # time -> min
    t_min, _ = scatter_min(x[:, time_col], cluster, dim=0, dim_size=M)
    t_min = torch.where(torch.isinf(t_min), torch.zeros_like(t_min), t_min)
    x_pool[:, time_col] = t_min

    # others -> mean
    rest = [i for i in range(C) if i not in (charge_col, time_col)]
    if rest:
        rest_sum = scatter_add(x[:, rest], cluster.unsqueeze(-1).expand(-1, len(rest)), dim=0, dim_size=M)
        x_pool[:, rest] = rest_sum / counts.unsqueeze(1).to(x.dtype)

    # ---- (9) Pooled positions: charge-weighted mean
    pos_pool = (scatter_add(w.unsqueeze(1) * pos.to(torch.float32), cluster, dim=0, dim_size=M)
                / w_sum.unsqueeze(1)).to(pos.dtype)

    batch_pool = batch[centers_idx]  # per-center batch
    # ---- (10) Rebuild edges on pooled graph
    edge_index_pool = knn_graph(pos_pool, k=k, batch=batch_pool, loop=False)

    return x_pool, pos_pool, batch_pool, edge_index_pool, centers_idx
    
def simple_graclus_pool(x, pos, edge_index, batch, keep_ratio=0.25, k=8, 
                       charge_col=0, time_col=1, eps=1e-8):
    """
    Simplified Graclus pooling without complex merging/splitting
    """
    device = x.device
    N, C = x.size()
    
    # 1. Perform Graclus clustering
    cluster = graclus(edge_index, num_nodes=N)
    
    # Handle unassigned nodes
    if (cluster == -1).any():
        unassigned = (cluster == -1).nonzero().flatten()
        max_cluster = cluster.max() + 1 if cluster.numel() > 0 else 0
        cluster[unassigned] = torch.arange(max_cluster, max_cluster + len(unassigned), device=device)
    
    # 2. Select centers and pool
    unique_clusters = cluster.unique()
    M = unique_clusters.numel()
    
    # Select centers based on charge
    centers_idx = []
    for cluster_id in unique_clusters:
        cluster_mask = (cluster == cluster_id)
        cluster_nodes = cluster_mask.nonzero().flatten()
        if len(cluster_nodes) > 0:
            cluster_charges = x[cluster_nodes, charge_col]
            center_idx = cluster_nodes[torch.argmax(cluster_charges)]
            centers_idx.append(center_idx)
    
    centers_idx = torch.tensor(centers_idx, device=device)
    
    # 3. Pool features
    cluster_mapping = torch.zeros(cluster.max() + 1, dtype=torch.long, device=device)
    cluster_mapping[unique_clusters] = torch.arange(M, device=device)
    cluster_indices = cluster_mapping[cluster]
    
    x_pool = torch.zeros(M, C, device=device, dtype=x.dtype)
    ones = torch.ones(N, dtype=torch.float32, device=device)
    counts = scatter_add(ones, cluster_indices, dim=0, dim_size=M).clamp_min(1.0)
    
    w = x[:, charge_col].clamp_min(0).to(torch.float32) + eps
    w_sum = scatter_add(w, cluster_indices, dim=0, dim_size=M).clamp_min(eps)
    
    # Charge -> sum
    x_pool[:, charge_col] = scatter_add(x[:, charge_col], cluster_indices, dim=0, dim_size=M)
    
    # Time -> min
    t_min, _ = scatter_min(x[:, time_col], cluster_indices, dim=0, dim_size=M)
    t_min = torch.nan_to_num(t_min, nan=0.0, posinf=0.0, neginf=0.0)
    x_pool[:, time_col] = t_min
    
    # Others -> mean
    rest = [i for i in range(C) if i not in (charge_col, time_col)]
    if rest:
        rest_sum = scatter_add(x[:, rest], cluster_indices.unsqueeze(-1).expand(-1, len(rest)), dim=0, dim_size=M)
        x_pool[:, rest] = rest_sum / counts.unsqueeze(1).to(x.dtype)
    
    # Pooled positions
    pos_float = pos.to(torch.float32)
    weighted_pos_sum = scatter_add(w.unsqueeze(1) * pos_float, cluster_indices, dim=0, dim_size=M)
    pos_pool = (weighted_pos_sum / w_sum.unsqueeze(1)).to(pos.dtype)
    
    batch_pool = batch[centers_idx]
    edge_index_pool = knn_graph(pos_pool, k=min(k, M-1), batch=batch_pool, loop=False)
    
    return x_pool, pos_pool, batch_pool, edge_index_pool, centers_idx

class EGNNEncoderNew(nn.Module):
    def __init__(self, in_features, hidden_dim, latent_dim, n_layers=8, drop_path_max=0.2, edge_drop_prob=0.05, jk="mean"):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = EGNNLayerWithCoord(
                in_features if i==0 else hidden_dim, hidden_dim,
                dropout_prob=0.1, coord_clip=0.2, init_alpha=0.1
            )
            # set DropPath prob per depth
            layer.drop_path = DropPath(p=(i/max(1, n_layers-1))*drop_path_max)
            layer.edge_drop_prob = edge_drop_prob
            self.layers.append(layer)

        self.jk = jk
        if jk == "concat":
            self.proj = nn.Linear(n_layers*hidden_dim, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x, pos, edge_index, batch):
        if self.jk == "concat":
            xs = []
        else:
            x_jk = 0.0

        for i, layer in enumerate(self.layers, 1):
            # (inside layer.forward you apply DropEdge when self.training)
            x, pos = layer(x, pos, edge_index)

            if self.jk == "concat":
                xs.append(x)
            elif self.jk == "mean":
                x_jk = x_jk + (x - x_jk) / i
            elif self.jk == "max":
                x_jk = x if i==1 else torch.maximum(x_jk, x)
            else:
                raise ValueError("jk must be 'mean' | 'max' | 'concat'")

        x = torch.cat(xs, dim=-1) if self.jk == "concat" else x_jk
        if self.jk == "concat":
            x = self.proj(x)

        x = self.final_norm(x)
        g = global_mean_pool(x, batch)
        return self.lin(g)

def make_vertex_seeds(x, pos, batch, early_frac=0.3, eps=1e-8):
    # x[:,0]=charge, x[:,1]=time
    charge = x[:, 0].clamp_min(0)
    time   = x[:, 1]

    seeds, timing = [], []
    G = int(batch.max().item()) + 1
    for g in range(G):
        m = (batch == g)

        pos_b = pos[m]                    # (Nb, 3)
        q_b   = charge[m].unsqueeze(1)    # (Nb, 1)  <-- critical change
        t_b   = time[m]                   # (Nb,)

        qsum = q_b.sum().clamp_min(eps)   # scalar (1,)
        cen_q = (q_b * pos_b).sum(dim=0) / qsum  # (3,) charge‑weighted centroid

        mean_t = t_b.mean(dim=0)
        std_t = t_b.std(dim=0)

        seeds.append(cen_q)
        timing.append(mean_t)
        timing.append(std_t)

    return torch.stack(seeds, dim=0)   
    
class VertexHead(nn.Module):
    def __init__(self, in_dim, head_hidden=256):
        super().__init__()
        self.seed_proj = nn.Linear(3, head_hidden // 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + head_hidden // 2, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Linear(head_hidden // 2, 3)
        )
    def forward(self, g_v, seed):
        s = torch.relu(self.seed_proj(seed))
        delta = self.mlp(torch.cat([g_v, s], dim=-1))
        return seed + delta
    
def tof_correct_times(raw_fht, pos, vhat, batch, v_ls=0.205, center='mean'):
    """
    x[:,1] is the raw time (ns), pos (m), vhat (m), v_ls (m/ns).
    """
    # print(pos)
    # print(vhat)
    # print(v_ls)

    d = (pos - vhat[batch]).norm(dim=1, keepdim=True)  # (N,1)
    t_raw = raw_fht
    # print(t_raw)
    t_corr = t_raw - d / v_ls
    if center == 'mean':
        t0 = scatter_mean(t_corr, batch, dim=0)[batch]  # (N,1)
        t_corr = t_corr - t0
    return t_corr  # (N,1)

class TokenLayer(nn.Module):
    def __init__(self, K=8, iters=3, temp=1.0, return_descriptors=True):
        super().__init__()
        self.K = K
        self.iters = iters
        self.temp = temp
        self.return_descriptors = return_descriptors
        self.log_w_dist = nn.Parameter(torch.tensor(0.0))
        self.log_w_tof  = nn.Parameter(torch.tensor(0.0))
        self.log_w_q    = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.last_debug = {}

    @torch.no_grad()
    def _init_tokens(self, vtx, batch):
        G = vtx.size(0)
        tok_pos = vtx.repeat_interleave(self.K, dim=0)
        tok_batch = torch.arange(G, device=vtx.device).repeat_interleave(self.K)
        return tok_pos, tok_batch

    def forward(self, x, pos, raw_fht, vtx, batch, raw_npe=None):
        device = pos.device
        raw_fht = raw_fht.view(-1, 1)  # (N,1)

        # ---- TOF-corrected time in real units ----
        v_ls = 0.205  # m/ns
        d = (pos - vtx[batch]).norm(dim=1, keepdim=True)   # (N,1)
        t_corr = raw_fht - d / v_ls                        # (N,1)
        t0 = scatter_mean(t_corr, batch, dim=0)[batch]
        t_corr = t_corr - t0                               # (N,1)

        # ---- Charge weight: prefer raw_npe if provided ----
        if raw_npe is not None:
            q = raw_npe.view(-1, 1).clamp_min(0.0)        # (N,1)
        else:
            # If x[:,0] is z-scored log1p(npe), un-zscore if you have stats:
            # log1p_npe = x[:,0:1]*npe_std + npe_mean
            # q = torch.expm1(log1p_npe).clamp_min(0.0)
            # If stats aren’t available here, fall back (less ideal):
            q = torch.expm1(x[:, 0:1]).clamp_min(0.0)

        # ---- Init tokens at vertex ----
        tok_pos, tok_batch = self._init_tokens(vtx, batch)
        G = vtx.size(0)

        w_dist = self.log_w_dist.exp()
        w_tof  = self.log_w_tof.exp()
        w_q    = self.log_w_q.exp()

        for _ in range(self.iters):
            new_chunks = []
            for g in range(G):
                m = (batch == g)
                if not m.any():
                    new_chunks.append(tok_pos[tok_batch == g])
                    continue

                P   = pos[m]                    # (Ng,3)
                Q   = q[m]                      # (Ng,1)
                T   = t_corr[m]                 # (Ng,1)
                Tok = tok_pos[tok_batch == g]   # (K,3)

                diff  = P.unsqueeze(1) - Tok.unsqueeze(0)  # (Ng,K,3)
                dist2 = (diff * diff).sum(-1)              # (Ng,K)

                logits = -(w_dist * dist2) \
                         - (w_tof  * T.abs()) \
                         + (w_q    * torch.log1p(Q))        # (Ng,K)

                a_ik = (logits / max(self.temp, 1e-6)).softmax(dim=1)   # (Ng,K)
                denom = a_ik.sum(dim=0, keepdim=True).clamp_min(1e-8)   # (1,K)
                alpha = a_ik / denom                                    # (Ng,K)

                Tok_new = (alpha.unsqueeze(-1) * P.unsqueeze(1)).sum(dim=0)  # (K,3)
                step = torch.clamp(self.alpha, 0.0, 1.0)
                Tok = Tok + step * (Tok_new - Tok)
                new_chunks.append(Tok)

            tok_pos = torch.cat(new_chunks, dim=0)

        desc = None
        if self.return_descriptors:
            desc_list = []
            for g in range(G):
                T = tok_pos[tok_batch == g]
                if T.size(0) < 2:
                    desc_list.append(torch.zeros(3, device=device))
                    continue
                used = torch.zeros(T.size(0), dtype=torch.bool, device=device)
                order = [0]; used[0] = True
                for _ in range(1, T.size(0)):
                    last = T[order[-1]].unsqueeze(0)
                    d = torch.cdist(last, T[~used], p=2).squeeze(0)
                    nxt = torch.arange(T.size(0), device=device)[~used][torch.argmin(d)]
                    order.append(int(nxt)); used[nxt] = True
                T_ord = T[torch.tensor(order, device=device)]
                seg = T_ord[1:] - T_ord[:-1]
                L_poly = torch.linalg.norm(seg, dim=1).sum()
                L_end  = torch.linalg.norm(T_ord[-1] - T_ord[0])
                straight = (L_end / (L_poly + 1e-8)).clamp_max(1.0)
                desc_list.append(torch.stack([L_end, L_poly, straight], dim=0))
            desc = torch.stack(desc_list, dim=0)

        with torch.no_grad():
            self.last_debug = {
                'w_dist': float(w_dist.item()),
                'w_tof' : float(w_tof.item()),
                'w_q'   : float(w_q.item()),
                'alpha_step': float(torch.clamp(self.alpha,0,1).item()),
            }

        return tok_pos, tok_batch, desc

class EGNNEncoderToF(nn.Module):
    def __init__(self, in_features, hidden_dim, latent_dim,
                 pre_layers=3, mid_layers=2, post_layers=2,
                 pooled_levels=None,
                 charge_col=0, time_col=1, prop_speed_ns_per_m=0.205,
                 token_K=24, token_iters=3):
        super().__init__()
        self.charge_col = charge_col
        self.time_col = time_col
        self.prop_speed = prop_speed_ns_per_m

        self.pre = nn.ModuleList([EGNNLayerAttentionVtx(in_features if i==0 else hidden_dim, hidden_dim)
                                  for i in range(pre_layers)])

        self.token_layer = TokenLayer(K=token_K, iters=token_iters, return_descriptors=True)

        self.level_blocks = nn.ModuleList([
            nn.ModuleList([EGNNLayerAttentionVtx(hidden_dim, hidden_dim) for _ in range(mid_layers)]),
            nn.ModuleList([EGNNLayerAttentionVtx(hidden_dim, hidden_dim) for _ in range(mid_layers)]),
            nn.ModuleList([EGNNLayerAttentionVtx(hidden_dim, hidden_dim) for _ in range(mid_layers)]),
            nn.ModuleList([EGNNLayerAttentionVtx(hidden_dim, hidden_dim) for _ in range(post_layers)]),
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.desc_proj = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))

        self.readout_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.att_pool = GlobalAttention(gate_nn=self.readout_gate)

        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.has_fixed_pool = pooled_levels is not None
        if self.has_fixed_pool:
            self.num_levels = len(pooled_levels)
            for L, Ldict in enumerate(pooled_levels):
                self.register_buffer(f"cluster_id_{L}",      Ldict["cluster_id"].long())
                self.register_buffer(f"pos_pool_{L}",        Ldict["pos_pool"].float())
                self.register_buffer(f"edge_pool_{L}",       Ldict["edge_index_pool"].long())

    # helpers
    def _t_corr(self, pos, vtx, batch, raw_fht):
        if raw_fht is None:
            return None
        if raw_fht.dim() == 1:
            raw_fht = raw_fht.unsqueeze(1)
        d = (pos - vtx[batch]).norm(dim=1, keepdim=True)
        t_corr = raw_fht - d / self.prop_speed
        t0 = scatter_mean(t_corr, batch, dim=0)[batch]
        return t_corr - t0
    
    def _apply_fixed_pool_level(
        self, x, pos, batch, L,
        raw_npe=None, raw_fht=None,
        charge_col=None, time_col=None, eps=1e-8
    ):
        """
        Fixed pooling for level L over a *batched* detector graph.

        Returns:
            x_pool, pos_pool, batch_pool, edge_index_pool, pooled_raw_npe, pooled_raw_fht
        """
        charge_col = self.charge_col if charge_col is None else charge_col
        time_col   = self.time_col   if time_col   is None else time_col

        cluster_id_single = getattr(self, f"cluster_id_{L}")      # (N_prev,)
        pos_pool_single   = getattr(self, f"pos_pool_{L}")        # (M_L, 3)
        edge_pool_single  = getattr(self, f"edge_pool_{L}")       # (2, E_L)

        device = x.device
        N = x.size(0)
        G = int(batch.max().item()) + 1
        M = pos_pool_single.size(0)

        # --- local index per event (assumes fixed node count & ordering) ---
        # If node counts may vary, replace this with a per-event offset method.
        N0 = (batch == 0).sum().item()
        if not torch.all(batch.bincount() == N0):
            raise RuntimeError("Fixed pooling requires equal nodes per event (and consistent ordering).")
        i_local = torch.arange(N, device=device) % N0  # 0..N0-1 within each event

        # --- flat cluster ids across batch (g*M + cluster_id[i_local]) ---
        cluster_flat = cluster_id_single[i_local].to(device) + batch.to(device) * M
        dim_size = G * M

        # --- counts per super-node (avoid divide-by-zero) ---
        ones = torch.ones(N, dtype=torch.float32, device=device)
        counts = scatter_add(ones, cluster_flat, dim=0, dim_size=dim_size).clamp_min(1.0)

        # ======================= Feature pooling (x) =======================
        C = x.size(1)
        x_pool = torch.zeros(G * M, C, device=device, dtype=x.dtype)

        # charge -> sum
        x_pool[:, charge_col] = scatter_add(x[:, charge_col], cluster_flat, dim=0, dim_size=dim_size)

        # time -> min
        t_min, _ = scatter_min(x[:, time_col], cluster_flat, dim=0, dim_size=dim_size)
        t_min = torch.where(torch.isinf(t_min), torch.zeros_like(t_min), t_min)
        x_pool[:, time_col] = t_min

        # rest -> mean
        rest = [i for i in range(C) if i not in (charge_col, time_col)]
        if rest:
            rest_sum = scatter_add(
                x[:, rest],
                cluster_flat.unsqueeze(-1).expand(-1, len(rest)),
                dim=0, dim_size=dim_size
            )
            x_pool[:, rest] = rest_sum / counts.unsqueeze(1).to(x.dtype)

        # ======================= Position pooling =========================
        # charge-weighted mean of positions
        w = x[:, charge_col].clamp_min(0).to(torch.float32) + eps
        w_sum = scatter_add(w, cluster_flat, dim=0, dim_size=dim_size).clamp_min(eps)

        pos_pool = torch.zeros(G * M, 3, device=device, dtype=pos.dtype)
        pos_pool.index_add_(0, cluster_flat, (w.unsqueeze(1) * pos.to(torch.float32)))
        pos_pool = (pos_pool / w_sum.unsqueeze(1)).to(pos.dtype)

        # =================== Optional: raw_npe / raw_fht ===================
        pooled_raw_npe = None
        if raw_npe is not None:
            if raw_npe.dim() == 2 and raw_npe.size(-1) == 1:
                raw_npe = raw_npe.squeeze(-1)
            pooled_raw_npe = scatter_add(raw_npe.to(torch.float32), cluster_flat, dim=0, dim_size=dim_size)
            pooled_raw_npe = pooled_raw_npe.view(-1, 1)

        pooled_raw_fht = None
        if raw_fht is not None:
            if raw_fht.dim() == 2 and raw_fht.size(-1) == 1:
                raw_fht = raw_fht.squeeze(-1)
            pooled_raw_fht, _ = scatter_min(raw_fht.to(torch.float32), cluster_flat, dim=0, dim_size=dim_size)
            pooled_raw_fht = torch.where(torch.isinf(pooled_raw_fht), torch.zeros_like(pooled_raw_fht), pooled_raw_fht)
            pooled_raw_fht = pooled_raw_fht.view(-1, 1)

        # ===================== Batched pooled edges =======================
        e = edge_pool_single.to(device)
        offsets = (torch.arange(G, device=device) * M).view(-1, 1, 1)  # (G,1,1)
        edge_index_pool = (e.unsqueeze(0) + offsets).reshape(2, -1).contiguous()

        batch_pool = torch.arange(G, device=device).repeat_interleave(M)

        return x_pool, pos_pool, batch_pool, edge_index_pool, pooled_raw_npe, pooled_raw_fht

    def _pool_and_retime(self, x, pos, batch, edge_index, raw_npe, raw_fht, L):
        x, pos, batch, edge_index, raw_npe, raw_fht = self._apply_fixed_pool_level(
            x, pos, batch, L=L, raw_npe=raw_npe, raw_fht=raw_fht,
            charge_col=self.charge_col, time_col=self.time_col
        )
        return x, pos, batch, edge_index, raw_npe, raw_fht

    def forward(self, x, pos, vtx, raw_npe, raw_fht, edge_index, batch):
        if not self.has_fixed_pool:
            raise RuntimeError("Fixed pooling required but pooled_levels=None.")

        # pre
        t_corr = self._t_corr(pos, vtx, batch, raw_fht)
        for layer in self.pre:
            x, pos = layer(x, pos, vtx, t_corr, edge_index, batch)

        _, _, desc = self.token_layer(x, pos, raw_fht, vtx, batch, raw_npe=raw_npe)  # (G,3)

        # multilevel
        levels_to_use = min(self.num_levels, len(self.level_blocks))
        for L in range(levels_to_use):
            x, pos, batch, edge_index, raw_npe, raw_fht = self._pool_and_retime(
                x, pos, batch, edge_index, raw_npe, raw_fht, L=L
            )
            t_corr = self._t_corr(pos, vtx, batch, raw_fht)
            for layer in self.level_blocks[L]:
                x, pos = layer(x, pos, vtx, t_corr, edge_index, batch)

        # readout
        x = self.final_norm(x)
        g = self.att_pool(x, batch)
        desc = F.layer_norm(desc, desc.shape[1:])
        g = torch.cat([g, self.desc_proj(desc)], dim=1)
        return self.lin(g)
    
class EGNNHierEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim=64, latent_dim=32,
                 k=16, keep1=0.25, keep2=0.25,
                 pre_layers=3, mid_layers=2, post_layers=2):
        super().__init__()
        
        self.pre = nn.ModuleList([
            EGNNLayerWithCoord(
                in_features if i==0 else hidden_dim,
                hidden_dim,
                dropout_prob=0.1,
                coord_clip=0.2, init_alpha=0.1,
                update_coords=True
            ) for i in range(pre_layers)
        ])

        self.mid = nn.ModuleList([
            EGNNLayerWithCoord(
                hidden_dim, hidden_dim,
                dropout_prob=0.1,
                update_coords=True
            ) for _ in range(mid_layers)
        ])

        # all but the last post layer update coords; the last one does not
        self.post = nn.ModuleList(
            [EGNNLayerWithCoord(hidden_dim, hidden_dim, dropout_prob=0.1, update_coords=True)
             for _ in range(max(0, post_layers - 1))]
            + [EGNNLayerWithCoord(hidden_dim, hidden_dim, dropout_prob=0.1, update_coords=False)]
        )

        self.keep1, self.keep2, self.k = keep1, keep2, k
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x, pos, edge_index, batch):
        # pre
        for layer in self.pre:
            x, pos = layer(x, pos, edge_index)

        # pool → mid
        x, pos, batch, edge_index, _ = pool_once(x, pos, batch, keep_ratio=self.keep1, k=self.k)
        for layer in self.mid:
            x, pos = layer(x, pos, edge_index)

        # pool → post
        x, pos, batch, edge_index, _ = pool_once(x, pos, batch, keep_ratio=self.keep2, k=self.k)
        for layer in self.post:
            x, pos = layer(x, pos, edge_index)

        # readout
        x = self.final_norm(x)
        g = global_mean_pool(x, batch)
        return self.lin(g)


class EGNNEnergyRegressor(nn.Module):
    def __init__(self, in_features, hidden_dim=64, latent_dim=32, pooled_levels=None):
        super().__init__()
        # self.encoder = EGNNAttentionEncoder(in_features, hidden_dim, latent_dim)
        self.encoder = EGNNEncoderToF(
            in_features=in_features, hidden_dim=hidden_dim, latent_dim=latent_dim,
            pre_layers=3, mid_layers=4, post_layers=3, 
            pooled_levels=pooled_levels
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    # def forward(self, x, pos, edge_index, batch):
    def forward(self, x, pos, vtx, raw_npe, raw_fht, edge_index, batch):
        # z = self.encoder(x, pos, edge_index, batch)
        z = self.encoder(x, pos, vtx, raw_npe, raw_fht, edge_index, batch)
        # out = self.head(z)
        # mu, log_var = out[...,0], out[...,1]
        return self.head(z)

class EGNNFlavourClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=64, latent_dim=32, num_classes=3):
        super().__init__()
        self.encoder = EGNNEncoder(in_features, hidden_dim, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, pos, edge_index, batch):
        z = self.encoder(x, pos, edge_index, batch)
        return self.head(z)

class EGNNZenithRegressor(nn.Module):
    def __init__(self, in_features, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = EGNNEncoder(in_features, hidden_dim, latent_dim)
        self.direction_head = nn.Linear(latent_dim, 3)

    def forward(self, x, pos, edge_index, batch):
        z = self.encoder(x, pos, edge_index, batch)
        direction = F.normalize(self.direction_head(z), dim=-1)
        zenith = torch.acos(direction[:, 2].clamp(-1.0, 1.0))
        return zenith
    
## --- GRU + EGNN ---
class GRU2EGNN(nn.Module):
    """
    Pattern B (interleaved) with your existing EGNNLayer.

    Expect x to be [num_nodes_total, K, 2] = per-node sequences of (charge, time) already
    batched together by PyG (i.e., 'batch' maps nodes to graphs).

    At each step s:
      - GRUCell updates a per-node hidden state h from the current [charge,time]
      - Build x_step = [charge, time, h]
      - Run ONE EGNNLayer call (uses charge/time for diffs)
      - Overwrite h with EGNN's output (ultra-minimal; no gating)

    After the K steps:
      - Mean pool per graph and pass through a small MLP head
    """
    def __init__(self, h_dim: int, egnn_width: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRUCell(input_size=2, hidden_size=h_dim)  # per-node temporal memory
        # EGNN sees [charge, time, h] each step
        self.egnn = EGNNLayer(in_features=2 + h_dim, out_features=egnn_width, dropout_prob=dropout)
        # Make EGNN output dimension == h_dim so we can overwrite h directly
        if egnn_width != h_dim:
            self.post = nn.Linear(egnn_width, h_dim)
        else:
            self.post = nn.Identity()

        # same pooling + head style as your encoder
        self.head = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, latent_dim)
        )

    def forward(self, x, pos, edge_index, batch):
        """
        x:    [num_nodes_total, K, 2]  (charge,time) per node
        pos:  [num_nodes_total, 3]
        edge_index: [2, E]   (already over the batched node index space)
        batch: [num_nodes_total]  (PyG graph ids)
        """
        assert x.dim() == 3 and x.size(-1) == 2, "x must be [num_nodes_total, K, 2] for this module."
        num_nodes, K, _ = x.shape
        H = self.gru.hidden_size

        # per-node hidden state
        h = x.new_zeros(num_nodes, H)   # [N_total, H]

        # time loop (no mask; simplest)
        for s in range(K):
            # 1) GRU update (per node) using raw step features [charge,time]
            h = self.gru(x[:, s, :], h)   # [N_total, H]

            # 2) EGNN step on [charge,time,h] for this time index
            x_step = torch.cat([x[:, s, :], h], dim=-1)  # [N_total, 2+H]
            egnn_out, _ = self.egnn(x_step, pos, edge_index)  # [N_total, egnn_width]

            # 3) overwrite h with EGNN output (ultra-minimal fusion)
            h = self.post(egnn_out)  # [N_total, H]

        # graph-level pooling + head
        x_mean = global_mean_pool(h, batch)   # [num_graphs, H]
        return self.head(x_mean)              # [num_graphs, latent_dim]