# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

import cv2
import image_paths
import transforms
from torch.utils.data import Dataset

class LoadDataset(Dataset):
    def __init__(self, image_paths, transform = False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('/')[-2]
        label = image_paths.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label
    
    
train_dataset = LoadDataset(image_paths.train_image_paths, transforms.train_transforms)
valid_dataset = LoadDataset(image_paths.valid_image_paths, transforms.test_transforms)
test_dataset = LoadDataset(image_paths.test_image_paths, transforms.test_transforms)

