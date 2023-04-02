import torch
import numpy as np
from torch_geometric.data import Data


def fix_datatypes(data: Data) -> Data:
    edge_features = torch.from_numpy(np.mean(data.x[data.edge_index], 1)).type(torch.float32)
    x = torch.from_numpy(data.x).type(torch.float32)
    edge_index = torch.from_numpy(data.edge_index.T).type(torch.int64)
    y = torch.tensor(1.0, dtype=torch.float32) if data.y > 0.8 else torch.tensor(0.0, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_features)