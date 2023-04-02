import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn, optim, sparse_coo_tensor
from src.expirements.models import Unet3D


class Unet3DModule(pl.LightningModule):
    def __init__(self, model, loss, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.model = model
        self.loss = loss
    
    def training_step(self, batch, batch_index):

        x, y = batch
        x = sparse_coo_tensor(*x).to_dense()
        z = self.model(x)
        x = x.to_sparse()
        y = sparse_coo_tensor(*y).to_dense()
        loss = self.loss(z, y)
        assert loss

        self.log("loss/train", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_index):

        x, y = batch
        x = sparse_coo_tensor(*x).to_dense()
        z = self.model(x)
        x = x.to_sparse()
        y = sparse_coo_tensor(*y).to_dense()
        loss = self.loss(z, y)
        assert loss
        self.log("loss/val", loss, batch_size=self.batch_size)
        return z, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=3e-4, weight_decay=3e-5)