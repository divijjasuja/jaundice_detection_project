from torch import nn
from torchvision import models
import torch

def resnet(device :str) -> models:
    model = models.resnet34(pretrained= True)
    model.eval()
    with torch.no_grad():
        nr_filters = model.fc.in_features  #number of input features of last layer
        model.fc = nn.Linear(nr_filters, 1)
        model = model.to(device)
    return model