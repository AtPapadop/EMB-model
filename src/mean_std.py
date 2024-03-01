# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:00:44 2024

@author: Thanasis
"""
import torch
from Dataset import full_dataset, train_dataset
from torch.utils.data import DataLoader


def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = len(full_dataset) * 224 * 224
    mean = 0.0
    std = 0.0
    psum  = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        psum += images.sum(axis = [0, 2, 3])
        psum_sq += (images ** 2).sum(axis = [0, 2, 3])
        
    mean = psum / num_pixels
    var = (psum_sq / num_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    return mean, std

# To properly compute the mean and standard deviation of the dataset, we need apply the data_transforms to the dataset
# Currently the train_transforms are applied to the train_dataset and the test_transforms are applied to the test_dataset
def main():
    batch_size = 256
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # We avoid using the full dataset to calculate the mean and standard deviation, as it is will provide future information
    # from the validation and test datasets which will scue the results of the model
    
    mean, std = get_mean_std(loader)
    print(mean, std)
    print(type(mean), type(std))

if __name__ == "__main__":
    main()

# The mean and standard deviation of the dataset are printed in the console
# This script is used to calculate the mean and standard deviation of the dataset and should be run standalone before training the model
# if there is a change in the dataset or the shuffle seed