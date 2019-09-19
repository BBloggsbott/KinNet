import torch.nn as nn
import torch.optim as optim
import logging

from models import KinNet

class KinNetLoss(nn.Module):
    def __init__(self):
        super(KinNetLoss, self).__init__()
        # TODO Define elements necessary for loss calculation

    def forward(self, input, target):
        # TODO Implement KinNet Loss
        return input - target


class KinNetTrainer:
    def __init__(self):
        self.model = KinNet()
        self.criterion = KinNetLoss()
        self. optimizer = optim.Adam(self.model.parameters())

    def train_model(self, epochs, data):
        logging.info("Starting training")
        for _ in range(epochs):
            batch = data.get_batch()
            target = batch[1]
            parent = batch[0][0]
            child = batch[0][1]
            self.optimizer.zero_grad()
            preds = self.model(parent, child)
            loss = self.criterion(preds, target)
            loss.backward()
            self.optimizer.step()