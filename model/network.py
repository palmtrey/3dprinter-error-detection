# Based on https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py

import warnings

warnings.filterwarnings('ignore')

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam


class ResNetClassifier(pl.LightningModule):
    def __init__(self, resnet_version, num_classes=2, batch_size = 10, epochs = 50,
                optimizer='adam', lr=1e-3,
                transfer=True, tune_fc_only=True, weight_decay=0):
        super().__init__()

       

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        self.lr = lr
        self.weight_decay = weight_decay
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        #instantiate loss criterion
        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if tune_fc_only: # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

        self.save_hyperparameters()

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        # print(len(x.shape))
        if len(x.shape) == 3:
            # print(x.unsqueeze(0).shape)
            x = x.unsqueeze(0)
        return int(torch.argmax(self(x), 1)[0])




class ResNeXtClassifier(pl.LightningModule):
    def __init__(self, resnet_version, num_classes=2, batch_size = 10, epochs = 50,
                optimizer='adam', lr=1e-3,
                transfer=True, tune_fc_only=True, weight_decay=0):
        super().__init__()

       

        self.__dict__.update(locals())
        resnets = {
            50: models.resnext50_32x4d, 
            101_32: models.resnext101_32x8d,
            101_64: models.resnext101_64x4d
        }
        self.lr = lr
        self.weight_decay = weight_decay
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        #instantiate loss criterion
        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if tune_fc_only: # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

        self.save_hyperparameters()

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        # print(len(x.shape))
        if len(x.shape) == 3:
            # print(x.unsqueeze(0).shape)
            x = x.unsqueeze(0)
        return int(torch.argmax(self(x), 1)[0])