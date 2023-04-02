import numpy as np
from torch_geometric.data import Data

def filter_empty_values(data: Data) -> bool:
    return data.x.size != 0 and data.edge_index.size != 0