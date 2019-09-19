import torch.nn as nn
import torchvision.models as models

class BranchModel(nn.Module):
    def __init__(self):
        super(BranchModel, self).__init__()
        # currently using vgg13 other available architectures are vgg11, vgg19. Diff combinations will by tried during training for best results
        self.vgg = models.vgg13(pretrained=True)

    def forward(self, x):
        x = self.vgg(x)
        return x


class KinNet(nn.Module):
    def __init__(self):
        super(KinNet, self).__init__()
        self.parent_branch = BranchModel()
        self.child_branch = BranchModel()

    def forward(self, parent, child):
        parent = self.parent_branch(parent)
        child = self.child_branch(child)
        return parent, child