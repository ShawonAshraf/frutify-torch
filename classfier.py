import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import pytorch_lightning as pl

"""
FrutifyInceptionV3 Model
Fine tuned from pretrained inception-v3
"""


class FrutifyInceptionV3(pl.LightningModule):
    def __init__(self, num_labels, learning_rate, model_name="inception_v3"):
        super(FrutifyInceptionV3, self).__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate

        self.save_hyperparameters()

        # disable aux logits
        self.feature_model = torchvision.models.inception_v3(pretrained=True, progress=True, aux_logits=False)

        # freeze params
        for param in self.feature_model.parameters():
            param.requires_grad = False

        # fc layer of inceptionv3 gives 1000 dim output
        n_features = 1000

        self.classifier = nn.Linear(n_features, num_labels)

    def forward(self, x):
        out = self.feature_model(x)
        out = F.log_softmax(self.classifier(out), dim=1)
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_index):
        image, label = batch["image"], batch["label"]

        out = self.forward(image)
        loss = F.cross_entropy(out, label)

        return {
            "loss": loss,
            "log": {
                "train_loss" : loss
            }
        }

    def validation_step(self, batch, batch_index):
        image, label = batch["image"], batch["label"]

        out = self(image)
        loss = F.cross_entropy(out, label)

        # log validation loss to progress bar
        self.log('validation_loss', loss, prog_bar=True)

        return {
            "val_loss": loss,
            "log": {
                "val_loss": loss
            }
        }


"""
FrutifyResnet101
Fine tuned from resnet101
"""


class FrutifyResnet101(pl.LightningModule):
    def __init__(self, num_labels, learning_rate, model_name="resnet101"):
        super(FrutifyResnet101, self).__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate

        self.save_hyperparameters()

        self.feature_model = torchvision.models.resnet18(pretrained=True, progress=True)
        self.feature_model.eval()
        # freeze params
        for param in self.feature_model.parameters():
            param.requires_grad = False

        n_features = self.__find_n_features()

        self.classifier = nn.Linear(n_features, num_labels)

    # finds the number of dimensions of the last layer of resnet
    def __find_n_features(self):
        inp = torch.autograd.Variable(
            torch.rand(1, 3, 299, 299)
        )
        out_f = self.feature_model(inp)
        n_size = out_f.data.view(1, -1).size(1)
        return n_size

    def forward(self, x):
        out = self.feature_model(x)

        # flatten for resnet inference
        out = out.view(out.size(0), -1)
        
        out = F.log_softmax(self.classifier(out), dim=1)
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

        out = self(image)
        loss = F.cross_entropy(out, label)

        # log validation loss to progress bar
        self.log('validation_loss', loss, prog_bar=True)
