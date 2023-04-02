import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F


class LightningGCN(pl.LightningModule):

    def __init__(self, model, batch_size, num_features=3, num_classes=2):
        super(LightningGCN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        # hidden layer node features
        self.hidden = 256
        self.batch_size = batch_size
        self.model = model

    def forward(self, x, edge_index, batch_index):

        x_out = self.model(x, edge_index, batch_index)

        return x_out

    def training_step(self, batch, batch_index):

        x, edge_index = batch.x, batch.edge_index
        batch_index = batch.batch

        x_out = self.forward(x, edge_index, batch_index)

        loss = F.binary_cross_entropy(x_out.view(-1), batch.y)

        # metrics here
        pred = x_out.argmax(-1)
        label = batch.y
        accuracy = (pred == label).sum() / pred.shape[0]

        self.log("loss/train", loss, batch_size=self.batch_size)
        self.log("accuracy/train", accuracy, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_index):
        x, edge_index = batch.x, batch.edge_index
        batch_index = batch.batch

        x_out = self.forward(x, edge_index, batch_index)

        loss = F.binary_cross_entropy(x_out.view(-1), batch.y)

        pred = x_out.argmax(-1)
        label = batch.y
        accuracy = (pred == label).sum() / pred.shape[0]
        self.log("loss/val", loss, batch_size=self.batch_size)
        self.log("accuracy/val", accuracy, batch_size=self.batch_size)
        return x_out, pred, batch.y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=3e-4, weight_decay=3e-5)
