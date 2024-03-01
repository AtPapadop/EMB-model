# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

import glob
import random
from pandas.core.common import flatten
from sys import platform

train_data_path = '/home/a/atpapadop/EMB/Dermnet/train'
test_data_path = '/home/a/atpapadop/EMB/Dermnet/test'

train_image_paths = []
classes = []

for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/')[-1])
    train_image_paths.append(glob.glob(data_path + '/*'))
    
classes_num = []
for class_ in range(len(classes)):
    classes_num.append(len(train_image_paths[class_]))
    
train_image_paths = list(flatten(train_image_paths))

random.seed(2024)
# Added random seed for reproducibility and for mean and std calculation to be consistent with the train dataset
random.shuffle(train_image_paths)

train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 


test_image_paths = []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

test_image_paths = list(flatten(test_image_paths))


print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

