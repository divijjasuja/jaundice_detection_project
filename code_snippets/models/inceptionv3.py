from torch import nn
from torchvision import models
import torch

def inceptionv3(device :str) -> models:
    model = models.inception_v3(pretrained= True)
    model.eval()
    with torch.no_grad():
        nr_filters = model.fc.in_features  #number of input features of last layer
        model.fc = nn.Linear(nr_filters, 1)
        model = model.to(device)
    return model