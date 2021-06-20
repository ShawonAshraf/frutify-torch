import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import torchvision
import os
import copy
import pytorch_lightning as pl


class FrutifyClassifier(pl.LightningModule):
    def __init__(self, num_labels, learning_rate):
        super(FrutifyClassifier, self).__init__()

        self.num_labels = num_labels
        self.learning_rate = learning_rate

        self.inceptionV3 = torchvision.models.inception_v3(pretrained=True)
        num_features = self.inceptionV3.fc.in_features

        self.inceptionV3.fc = nn.Linear(num_features, self.num_labels)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.inceptionV3(x)
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_index):
        image, label = batch["image"], batch["label"]

        out = self(image)
        logits, _ = torch.max(out, dim=1)
        loss = self.loss_fn(logits, label)

        return {
            "loss": loss
        }

    def validation_step(self, batch, batch_index):
        image, label = batch["image"], batch["label"]

        out = self(image)
        logits, _ = torch.max(out, dim=1)
        loss = self.loss_fn(logits, label)

        # log validation loss to progress bar
        self.log('validation_loss', loss, prog_bar=True)
