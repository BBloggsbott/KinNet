import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

from .data_utils import KinNetDataset
from .models import KinNet

class KinNetLoss(nn.Module):
    k = 20
    def __init__(self):
        super(KinNetLoss, self).__init__()
    
    def forward(self, xp, xc, xpi, xci):
        k = self.k
        li = (1/2)*(self.similarity(xp, xc) - 1)**2
        lik = 0
        for i in range(k):
            lik = lik + (1/2)*(self.similarity(xp, xci[i])+ 1)**2
        lki = 0
        for i in range(k):
            lki = lki + (1/2)*(self.similarity(xpi[i], xc) + 1)**2
        return li + (1/k)*lik + (1/k)*lki
    
    def similarity(self, xp, xc):
        return F.cosine_similarity(xp, xc)


class KinNetTrainer:
    def __init__(self, dataset = 1, data_dir="data", batch_size = 8):
        self.data = KinNetDataset(dataset, data_dir, batch_size)
        self.model = KinNet()
        self.criterion = KinNetLoss()
        self. optimizer = optim.Adam(self.model.parameters())

    def train_model(self, epochs):
        logging.info("Starting training")
        for _ in range(epochs):
            loss = 0
            for i in range(self.data.bs):
                parent, child = self.data.get_random_pair(kin=True)
                self.optimizer.zero_grad()
                parent, child = self.model(parent, child)
                xpi = []
                xci = []
                for j in range(self.criterion.k):
                    xp, xc = self.data.get_random_pair()
                    xp, xc = self.model(xp, xc)
                    xpi.append(xp)
                    xci.append(xc)
                loss += self.criterion(parent, child)
            loss = loss/self.data.bs
            loss.backward()
            self.optimizer.step()
            print(" Epoch {} - Done".format(_+1))