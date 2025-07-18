from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch

# Load graph data
data = torch.load("utils/pmt_graph.pt")
edge_index = data['edge_index']
positions = data['pmt_positions'].numpy()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot edges
for i, j in edge_index.T.numpy():
    x = [positions[i, 0], positions[j, 0]]
    y = [positions[i, 1], positions[j, 1]]
    z = [positions[i, 2], positions[j, 2]]
    ax.plot(x, y, z, color='gray', linewidth=0.2, alpha=0.25)

# Plot nodes
# ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=2, c='blue')

ax.set_title("PMT KNN Graph (3D)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
plt.tight_layout()
plt.savefig('plots/graph_plot.pdf')