# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

import torch
import torchvision

model = torchvision.models.efficientnet_b0(weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[0] = torch.nn.Dropout(p=0.5,inplace=True)
model.classifier[1] = torch.nn.Linear(1280, 23)