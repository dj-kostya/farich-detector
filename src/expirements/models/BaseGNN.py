import torch
import numpy as np
from torch import nn
from torch_geometric.data import Data
from src.expirements.layers import ProcessorLayer, SmoothingLayer, SMOOTHING_LAYER_NAME

class BaseGNN(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, geom_in_dim=3, out_dim=3, hidden_nodes=128):
        super().__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.processor = nn.ModuleList()

        self.decoder = nn.Linear(self.num_filters[-1], out_dim)

        for ef, nf in zip(self.edge_feat_dims, self.num_filters):
            self.processor.append(ProcessorLayer(ef, nf, hidden_nodes))
            self.processor.append(SmoothingLayer())

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # skip_info = x[:, :self.geom_in_dim]
        # print(skip_info)
        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)
            # if layer.name == SMOOTHING_LAYER_NAME:
            #     x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred

    def loss(self, pred, inp):
        true_flow = inp
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)