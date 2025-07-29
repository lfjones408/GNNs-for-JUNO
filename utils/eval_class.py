import os
import torch
import argparse
import yaml
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score

from model import EGNNFlavourClassifier
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

def get_eval_loader(h5_path, edge_index, stats, pos, batch_size, limit=None, num_workers=2, target=None, class_type=None, preload=False, device='cpu'):
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
            class_type=class_type,
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
            logits = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            preds = logits.argmax(dim=1)
            preds_all.append(preds.cpu().numpy())
            targets_all.append(batch.y.cpu().numpy())

    logger.info(f' preds_all[0] datatype = {preds_all[0].dtype}')
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    cnf_mtrx = confusion_matrix(targets_all, preds_all)

    logger.info(f'Confusion Matrix: \n {cnf_mtrx}')
    logger.info(f'Accuracy Score {accuracy_score(targets_all, preds_all)} | Precision Score {precision_score(targets_all, preds_all, average=None)} | Recall Score {recall_score(targets_all, preds_all, average=None)}')

    return preds_all, targets_all, cnf_mtrx

def plot_predictions(preds, targets, conf_matrix, class_type='3-label', save_path="output/plots/classification"):
    os.makedirs(save_path, exist_ok=True)

    if class_type=='3-label':
        class_names = [r'$\nu_{e}$-like-CC', r'$\nu_{\mu}$-like-CC', 'NC']
    elif class_type=='5-label':
        class_names = [r'$\bar{\nu}_{e}$-CC', r'\nu_{e}$-CC', r'$\bar{\nu}_{\mu}$-CC', r'$\nu_{\mu}$-CC', 'NC']
    else:
        logger.info("Give a valid classification type")

    plt.figure(figsize=(6,6))
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    display.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.close()

    eff_cnf_mtrx = confusion_matrix(targets, preds, normalize='true')

    plt.figure(figsize=(6,6))
    display = ConfusionMatrixDisplay(confusion_matrix=eff_cnf_mtrx, display_labels=class_names)
    display.plot(cmap='Blues')
    plt.title("Confusion Matrix Efficiencies")
    plt.savefig(f"{save_path}/confusion_matrix_eff.png")
    plt.close()


    # fpr, tpr, _ = roc_curve(targets, preds)
    # auc = roc_auc_score(targets, preds)

    # plt.figure(figsize=(6,6))
    # plt.plot(fpr, tpr, label=f'AUC Score = {auc}')
    # plt.plot([0,1], [0,1], linestyle='--', label='Random Classifier')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.savefig(f"{save_path}/roc_curve.png")

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
    class_type = cfg['training']['class_type']
    limit = cfg['training']['limit']

    edge_index = graph['edge_index']
    pos = graph['pmt_positions']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[Device] {device}")

    model = EGNNFlavourClassifier(
        in_features=2,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Loaded model from {model_path}")

    eval_loader = get_eval_loader(h5_path, edge_index, stats, pos,batch_size=batch_size, limit=limit, target=target, class_type=class_type)
    preds, targets, cnf_mtrx = evaluate(model, eval_loader, device)

    plot_predictions(preds, targets, cnf_mtrx, class_type=class_type)

if __name__ == "__main__":
    main()