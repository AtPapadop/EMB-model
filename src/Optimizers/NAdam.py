# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

from torch.optim import NAdam

def get_optimizer(model, learning_rate):
    return NAdam(model.parameters(), lr=learning_rate)
