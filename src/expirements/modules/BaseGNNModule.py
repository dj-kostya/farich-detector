import pytorch_lightning as pl
from torch import nn, optim
from src.expirements.models import BaseGNN


class BaseGNNModule(pl.LightningModule):
    def __init__(self, model: BaseGNN):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch, batch.y 
        batch_index = batch.batch
        self.f
        z = self.model(x, )
        loss = self.model.loss(z, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
