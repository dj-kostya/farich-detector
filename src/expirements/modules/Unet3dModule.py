import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn, optim
from src.expirements.models import Unet3D


class Unet3DModule(pl.LightningModule):
    def __init__(self, model:Unet3D, loss, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.model = model
        self.loss = loss
    
    def training_step(self, batch, batch_index):
        x, y = batch
        z = self.model(x)
        loss = self.loss(z, y.to_dense())

        self.log("loss/train", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        z = self.model(x)
        loss = self.loss(z, y.to_dense())
        self.log("loss/val", loss, batch_size=self.batch_size)
        return z, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=3e-4, weight_decay=3e-5)