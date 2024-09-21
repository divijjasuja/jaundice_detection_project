from torch import nn
from torchvision import models
import torch

def googlenet(device :str) -> models:
    model = models.googlenet(pretrained= True)
    model.eval()
    with torch.no_grad():
        nn_filters = model.fc.in_features
        model.fc = nn.Linear(nn_filters,1)
        model = model.to(device)
    return model