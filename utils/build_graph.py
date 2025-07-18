import logging
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Optional logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load PMT Geometry
pmt_pos_file = '/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.1.3/data/Detector/Geometry/PMTPos_CD_LPMT.csv'
pmt_csv = pd.read_csv(pmt_pos_file, comment='#', sep='\s+', header=None)
pmt_csv.columns = ['CopyNo', 'X', 'Y', 'Z', 'Theta', 'Phi']
points = np.column_stack((pmt_csv['X'] * 1e-3, pmt_csv['Y'] * 1e-3, pmt_csv['Z'] * 1e-3))

logger.info('-> Geometry loaded')

# Build KNN Graph
knn = NearestNeighbors(n_neighbors=8)
knn.fit(points)
edges = knn.kneighbors_graph(mode='connectivity').tocoo()

rows = edges.row.astype(np.int64)
cols = edges.col.astype(np.int64)

edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)

# Add bidirectional edges
edge_index_rev = edge_index[[1, 0]]
edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

# Add self-loops
self_loops = torch.arange(points.shape[0], dtype=torch.long)
self_loops = self_loops.unsqueeze(0).repeat(2, 1)
edge_index = torch.cat([edge_index, self_loops], dim=1)

logger.info('-> Graph built')

# Save to file
torch.save({
    'edge_index': edge_index,
    'pmt_positions': torch.tensor(points, dtype=torch.float32),
}, 'utils/pmt_graph.pt')

logger.info("Graph saved to 'pmt_graph.pt'")