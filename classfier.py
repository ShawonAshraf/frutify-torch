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

        self.model = torchvision.models.inception_v3(pretrained=True,
                                                     progress=True,
                                                     transform_input=True)
        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(num_features, self.num_labels)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out, _ = self.model(x)
        out = self.softmax(out)
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_index):
        image, label = batch["image"], batch["label"]

        out = self.forward(image)
        loss = F.cross_entropy(out, label)

        return {
            "loss": loss
        }

    def validation_step(self, batch, batch_index):
        image, label = batch["image"], batch["label"]

        out = self.forward(image)
        loss = F.cross_entropy(out, label)

        # log validation loss to progress bar
        self.log('validation_loss', loss, prog_bar=True)
