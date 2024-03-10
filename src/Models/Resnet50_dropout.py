# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

import torch
import torchvision

model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(2048,23))