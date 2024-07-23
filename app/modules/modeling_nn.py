# app/modules/modeling_nn.py

import lightning as L
import torch
import torch.nn.functional as F
from torcheval.metrics import BinaryAUROC

from app.modules.layers import Layers


class NeuralNet(L.LightningModule):

    def __init__(
        self,
        input_dim: int,
        hiden_dim: int,
        output_dim: int,
        dropout: float,
        lr: float,
        weight_decay: float,
    ):
        super().__init__()
        # Training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        # Validation parameters
        self.metric = BinaryAUROC()
        # Model parameters
        # TODO
        self.model = Layers(input_dim, hiden_dim, output_dim, dropout)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def forward(self, batch: torch.Tensor):
        # TODO
        pred = self.model(batch)
        return pred

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        # TODO
        loss = F.binary_cross_entropy_with_logits(out, y)
        roc_auc = self.metric.compute()
        self.log_dict(
            {"test_loss": float(loss), "test_roc_auc": float(roc_auc)},
            # on_epoch=True,
            prog_bar=True,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # TODO
        loss = F.binary_cross_entropy_with_logits(out, y)
        self.log("train_loss", float(loss), on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # TODO
        self.metric.update(out, y)
        loss = F.binary_cross_entropy_with_logits(out, y)
        roc_auc = self.metric.compute()
        self.log_dict(
            {"val_loss": float(loss), "val_roc_auc": float(roc_auc)},
            on_epoch=True,
            prog_bar=True,
        )
