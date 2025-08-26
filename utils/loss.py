import torch
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss, TripletMarginLoss
from torch_geometric.nn import global_mean_pool

class Losses:
    def __init__(self, loss_type='graph', margin=1.0, alpha=1.0, huber_delta=1.0,
                 clamp_logvar=10.0, reduction='mean'):
        self.loss_type = loss_type
        self.margin = margin
        self.alpha = alpha
        self.huber_delta = huber_delta
        self.clamp_logvar = clamp_logvar
        self.reduction = reduction

        self.mse = MSELoss(reduction=reduction if reduction in ('mean','sum') else 'mean')
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
        elif self.loss_type == 'huber':
            return self.huber_loss(model_output, batch.y)
        elif self.loss_type in ('heteroscedastic', 'hetero'):
            return self.heteroscedastic_loss(model_output, batch.y)
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

    def huber_loss(self, preds, targets):
        if preds.dim() == 2 and preds.size(-1) == 1:
            preds = preds.squeeze(-1)
        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        return F.huber_loss(preds, targets, delta=self.huber_delta, reduction=self.reduction)

    # ---- New: Heteroscedastic Gaussian NLL ----
    def heteroscedastic_loss(self, preds, targets, eps: float = 1e-8):
        """
        preds: either (mu, log_var) tuple/list OR Tensor with last dim==2 ([...,0]=mu, [...,1]=log_var)
        targets: same shape as mu (broadcasted if needed)
        """
        # Unpack predictions
        if isinstance(preds, (tuple, list)):
            if len(preds) != 2:
                raise ValueError("heteroscedastic_loss expects (mu, log_var) when passing a tuple/list.")
            mu, log_var = preds
        else:
            if preds.size(-1) != 2:
                raise ValueError(f"heteroscedastic_loss expected last dim=2, got {preds.size(-1)}")
            mu, log_var = preds[..., 0], preds[..., 1]

        # Ensure target shape
        if targets.shape != mu.shape:
            targets = targets.view_as(mu)

        # Clamp for numerical stability
        log_var = torch.clamp(log_var, min=-self.clamp_logvar, max=self.clamp_logvar)
        var = torch.exp(log_var) + eps

        # Per-sample NLL
        nll = 0.5 * ((targets - mu) ** 2 / var + log_var)

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")