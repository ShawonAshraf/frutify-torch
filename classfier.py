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

        self.save_hyperparameters()

        self.num_labels = num_labels
        self.learning_rate = learning_rate

        self.feature_model = torchvision.models.resnet18(pretrained=True, progress=True)
        self.feature_model.eval()
        # freeze params
        for param in self.feature_model.parameters():
            param.requires_grad = False

        n_features = self.__find_n_features()

        self.classifier = nn.Linear(n_features, num_labels)

    def __find_n_features(self):
        inp = torch.autograd.Variable(
            torch.rand(1, 3, 299, 299)
        )
        out_f = self.feature_model(inp)
        n_size = out_f.data.view(1, -1).size(1)
        return n_size

    def forward(self, x):
        out = self.feature_model(x)
        out = out.view(out.size(0), -1)
        out = F.log_softmax(self.classifier(out), dim=1)
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_index):
        image, label = batch["image"], batch["label"]

        out = self.forward(image)
        loss = F.nll_loss(out, label)

        return {
            "loss": loss
        }

    def validation_step(self, batch, batch_index):
        image, label = batch["image"], batch["label"]

        out = self.forward(image)
        loss = F.nll_loss(out, label)

        # log validation loss to progress bar
        self.log('validation_loss', loss, prog_bar=True)
