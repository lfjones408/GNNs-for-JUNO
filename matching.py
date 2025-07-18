import faiss
import numpy as np
from numpy.linalg import norm

data = np.load('embeddings/latent_with_labels.npz')
z_graph = data["embeddings"]
y = data["labels"]

index = faiss.IndexFlatL2(z_graph.shape[1])
index.add(z_graph)

query_idx = 2
query = z_graph[query_idx].reshape(1, -1)
D, I = index.search(query, k=10)

query_label = y[query_idx]

print(f"Query #{query_idx} | Flavour: {query_label[1]} | Energy: {query_label[2]} | Dir: {query_label[0]}")
print("Matches:")
for i, idx in enumerate(I[0]):
    match_label = y[idx]
    print(f"  Match {i}: idx={idx} | Flavour: {match_label[1]} | Energy: {match_label[2]} | Dir: {match_label[0]}")