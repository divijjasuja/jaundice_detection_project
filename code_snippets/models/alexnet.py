from torch import nn
from torchvision import models
import torch

def alexnet(device :str) -> models:
    model = models.alexnet(pretrained= True)
    model.eval()
    with torch.no_grad():
        nn_filters = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(nn_filters,1)
        model = model.to(device)
    return model