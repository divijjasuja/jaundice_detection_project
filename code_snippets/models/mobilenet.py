from torch import nn
from torchvision import models
import torch

def mobilenet(device :str) -> models:
    model = models.mobilenet_v2(pretrained= True)
    model.eval()
    with torch.no_grad():
        nn_filters = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(nn_filters,1)
        model = model.to(device)
    return model