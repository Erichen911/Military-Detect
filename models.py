import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        pretrained_model = models.resnet152(pretrained=True)
        modules = list(pretrained_model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        for p in self.extractor.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.extractor(x).view(1, -1)

class DQN(nn.Module):
    def __init__(self, n_inp, n_out):
        super(DQN, self).__init__()
        self.n_inp = n_inp
        self.n_out = n_out

        self.fc1 = nn.Linear(n_inp, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_vals = self.fc3(x)
        return q_vals
