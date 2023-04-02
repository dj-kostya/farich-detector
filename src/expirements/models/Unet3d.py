import sys
import time
import torch
import torch.nn as nn
from src.expirements.layers.unet_components import EncoderBlock, DecoderBlock


class Unet3D(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="sigmoid"):
        super(Unet3D, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        if x.is_sparse:
            x=x.to_dense()
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.final_activation(x)
        # print("Final output shape: ", x.shape)
        return x