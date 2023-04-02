from torch import nn, cat, div, abs
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter

PROCESSOR_LAYER_NAME = 'processor'


class ProcessorLayer(MessagePassing):

    def __init__(self, edge_feats, node_feats, hidden_state, idx=0):
        super(ProcessorLayer, self).__init__()

        self.name = PROCESSOR_LAYER_NAME
        self.idx = idx

        self.edge_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      nn.ReLU(),
                                      nn.LazyLinear(edge_feats)
                                      )
        self.node_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      nn.ReLU(),
                                      nn.LazyLinear(node_feats)
                                      )

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        out, updated_edges = self.propagate(
            edge_index, x=x, edge_attr=edge_attr)

        updated_nodes = cat([x, out], dim=1)
        updated_nodes = self.node_mlp(updated_nodes)

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        updated_edges = cat(
            [div(x_i + x_j, 2), abs(x_i - x_j) / 2, edge_attr], 1)
        
        updated_edges = self.edge_mlp(updated_edges)
        return updated_edges

    def aggregate(self, updated_edges, edge_index):
        node_dim = 0
        out = scatter(
            updated_edges, edge_index[0, :], dim=node_dim, reduce='sum')
        return out, updated_edges
