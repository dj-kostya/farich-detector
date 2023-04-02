from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter

SMOOTHING_LAYER_NAME = 'smoothing'


class SmoothingLayer(MessagePassing):

    def __init__(self, idx: int = 0):
        super(SmoothingLayer, self).__init__()
        self.name = SMOOTHING_LAYER_NAME
        self.idx = idx

    def forward(self, x, edge_index, edge_attr):
        out_nodes, out_edges = self.propagate(
            edge_index, x=x, edge_attr=edge_attr)
        return out_nodes, out_edges

    def message(self, x_i, x_j):
        updated_edges = (x_i + x_j) / 2
        return updated_edges

    def aggregate(self, updated_edges, edge_index):
        node_dim = 0
        out = scatter(
            updated_edges, edge_index[0, :], dim=node_dim, reduce='mean')
        return out, updated_edges
