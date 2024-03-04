# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

from torch.optim import Adam

def get_optimizer(model, learning_rate):
    return Adam(model.parameters(), lr=learning_rate)
