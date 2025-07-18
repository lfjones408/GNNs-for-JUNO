import torch
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss, TripletMarginLoss
from torch_geometric.nn import global_mean_pool

class Losses:
    def __init__(self, loss_type='graph', margin=1.0, alpha=1.0):
        self.loss_type = loss_type
        self.margin = margin
        self.alpha = alpha
        self.mse = MSELoss()
        self.cross_entropy = CrossEntropyLoss()
        self.triplet = TripletMarginLoss(margin=margin)

    def __call__(self, model_output, batch):
        if self.loss_type == 'graph':
            return self.graph_level_loss(model_output, batch)
        elif self.loss_type == 'node':
            return self.node_level_loss(model_output, batch)
        elif self.loss_type == 'hybrid':
            return self.hybrid_loss(model_output, batch)
        elif self.loss_type == 'triplet':
            return self.triplet_loss(model_output)
        elif self.loss_type == 'classification':
            return self.classification_loss(model_output, batch.y)
        elif self.loss_type == 'regression':
            return self.regression_loss(model_output, batch.y)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def graph_level_loss(self, x_recon, batch):
        target = global_mean_pool(batch.x, batch.batch)
        return self.mse(x_recon, target)

    def node_level_loss(self, x_recon, batch):
        return self.mse(x_recon, batch.x)

    def hybrid_loss(self, x_recon_tuple, batch):
        x_recon_graph, x_recon_nodes = x_recon_tuple
        graph_target = global_mean_pool(batch.x, batch.batch)
        loss_graph = self.mse(x_recon_graph, graph_target)
        loss_node = self.mse(x_recon_nodes, batch.x)
        return loss_graph + self.alpha * loss_node

    def triplet_loss(self, triplets):
        anchor, positive, negative = triplets
        return self.triplet(anchor, positive, negative)

    def classification_loss(self, logits, labels):
        return self.cross_entropy(logits, labels)

    def regression_loss(self, preds, targets):
        return self.mse(preds, targets)