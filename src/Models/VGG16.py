# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:46:52 2024

@author: Thanasis
"""

import torch
import torchvision

model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = torch.nn.Linear(4096, 23)