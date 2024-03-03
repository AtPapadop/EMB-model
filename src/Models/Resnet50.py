import torch
import torchvision

model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Linear(2048, 23)