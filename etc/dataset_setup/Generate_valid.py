# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:21:39 2024

@author: Thanasis
"""

import glob
import random
from pandas.core.common import flatten
from pathlib import Path
import os


train_data_path = '/home/a/atpapadop/EMB/Dermnet/train'
valid_data_path = '/home/a/atpapadop/EMB/Dermnet/valid'


classes = ['Acne and Rosacea Photos',
 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
 'Atopic Dermatitis Photos',
 'Bullous Disease Photos',
 'Cellulitis Impetigo and other Bacterial Infections',
 'Eczema Photos',
 'Exanthems and Drug Eruptions',
 'Hair Loss Photos Alopecia and other Hair Diseases',
 'Herpes HPV and other STDs Photos',
 'Light Diseases and Disorders of Pigmentation',
 'Lupus and other Connective Tissue diseases',
 'Melanoma Skin Cancer Nevi and Moles',
 'Nail Fungus and other Nail Disease',
 'Poison Ivy Photos and other Contact Dermatitis',
 'Psoriasis pictures Lichen Planus and related diseases',
 'Scabies Lyme Disease and other Infestations and Bites',
 'Seborrheic Keratoses and other Benign Tumors',
 'Systemic Disease',
 'Tinea Ringworm Candidiasis and other Fungal Infections',
 'Urticaria Hives',
 'Vascular Tumors',
 'Vasculitis Photos',
 'Warts Molluscum and other Viral Infections']


with open('valid_dataset.txt') as f:
    valid_files = f.read().splitlines()
    
train_paths = list(os.path.join(train_data_path, valid_files[i]) for i in range(len(valid_files)))
valid_paths = list(map(lambda st: str.replace(st, 'train', 'valid'), train_paths))
valid_classes_paths = list(os.path.join(train_data_path, classes[i]) for i in range(len(classes)))

for i in range (len(valid_classes_paths)):
    os.mkdir(valid_classes_paths[i])

for i in range (len(valid_paths)):
    src = Path(train_paths[i])
    dst = Path(valid_paths[i])
    src.rename(dst)
