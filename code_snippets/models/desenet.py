from torch import nn
from torchvision import models
import torch

def densenet(device :str) -> models:
    model = models.densenet121(pretrained= True)
    model.eval()
    with torch.no_grad():
        nn_filters = model.classifier.in_features
        model.classifier = nn.Linear(nn_filters,1)
        model = model.to(device)
    return model