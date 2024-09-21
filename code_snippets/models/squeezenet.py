from torch import nn
from torchvision import models
import torch


def squeezenet(device: str) -> models:
    model = models.squeezenet(pretrained=True)
    model.eval()
    with torch.no_grad():
        model.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1))
        model.num_classes = 1
        model = model.to(device)
    return model
