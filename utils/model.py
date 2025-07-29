import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, TopKPooling
from torch_geometric.utils import softmax

# --- GAT ---

class DeepGATEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32, heads=4):
        super().__init__()

        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.pool1 = TopKPooling(hidden_dim * heads, ratio=0.5)

        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=2, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.pool2 = TopKPooling(hidden_dim * 2, ratio=0.2)

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
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

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
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32, num_classes=2, heads=4, dropout=0.2):
        super().__init__()
        self.encoder = DeepGATEncoder(input_dim, hidden_dim, latent_dim, heads)
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
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32, output_dim=1, heads=4, dropout=0.2):
        super().__init__()
        self.encoder = DeepGATEncoder(input_dim, hidden_dim, latent_dim, heads)
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
        self.norm = nn.LayerNorm(out_features)

        # For residual connection
        if in_features != out_features:
            self.res_connection = nn.Linear(in_features, out_features)
        else:
            self.res_connection = nn.Identity()

    def forward(self, x, pos, edge_index):
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        pos_i, pos_j = pos[row], pos[col]
        time_i, time_j = x[row][:,1], x[col][:,1]
        charge_i, charge_j = x[row][:,0], x[col][:,1]

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
        
        self.norm = nn.BatchNorm1d(out_features)
        
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

class EGNNEncoder(nn.Module):
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

        num_nodes = batch.size(0)
        num_graphs = batch.max().item() +1
        global_node_indices = torch.arange(num_graphs, device=batch.device) + (num_nodes - num_graphs)
        mask = torch.ones(num_nodes, dtype=torch.bool, device=batch.device)
        mask[global_node_indices] = False
        x_masked = x[mask]
        batch_masked = batch[mask]

        pooled = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        pooled.index_add_(0, batch_masked, x_masked)

        counts = torch.bincount(batch_masked, minlength=num_graphs).float().unsqueeze(1)
        x_mean = pooled / (counts + 1e-8)

        return self.lin(x_mean)

class EGNNEnergyRegressor(nn.Module):
    def __init__(self, in_features, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = EGNNEncoder(in_features, hidden_dim, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, pos, edge_index, batch):
        z = self.encoder(x, pos, edge_index, batch)
        return self.head(z).squeeze(-1)

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