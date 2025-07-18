import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from sklearn.manifold import TSNE

# Arguement Parser
parser = ArgumentParser(description='Graph Auto Encoder to reduce JUNO event for matching')
parser.add_argument('--input', type=str, required=True, help='input file containing events')
args = parser.parse_args()

latent_vec = np.load(args.input)

z_embedded = TSNE(n_components=2, perplexity=2).fit_transform(latent_vec)
plt.scatter(z_embedded[:, 0], z_embedded[:, 1])  # if you have labels
plt.title("Event Embeddings (t-SNE)")
plt.savefig('plots/embeddings_vis.png')