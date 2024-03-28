# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:01:15 2024

@author: Thanasis
"""

import torch
import torchvision

model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(1280, 2)