# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=224),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.3),
        A.CenterCrop(height=224, width=224),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean = (0.5392, 0.4123, 0.3805), std = (0.2566, 0.2130, 0.2068)),
        A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.05,0.05), contrast_limit=(-0.05, 0.05), p=0.3),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=224),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean = (0.5392, 0.4123, 0.3805), std = (0.2566, 0.2130, 0.2068)),
        ToTensorV2(),
    ]
)

# Only for mean and std calculation
data_transforms = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=0, std=1),
    ToTensorV2(),
])