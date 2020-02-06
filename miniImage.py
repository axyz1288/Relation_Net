#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class miniImage(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, way=5):
        """
        Args:
            mat_file (string): Path to the mat file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotation_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.way = way
        self.sample_class()
        
    def __len__(self):
        return self.class_idx[1]
        
    def __getitem__(self, idx):
        # sample class
        i = np.random.choice(self.way_idx)
        # read img
        file_name = self.annotation_csv['filename'][idx+i]
        class_name = self.annotation_csv['label'][idx+i]
        file_path = self.root_dir + '/' + class_name + '/' + file_name
        img = Image.open(file_path)
        # transform
        if self.transform:
            img = self.transform(img)
        label = self.class_dict[class_name]
        sample = {'data':img, 'label':label}
        return sample
    
    def sample_class(self):
        # find class index
        self.class_label, self.class_idx = np.unique(self.annotation_csv['label'], return_index=True)
        self.class_dict = dict(
            zip(self.class_label, range(1, len(self.class_label)+1))
        )
        
        # choose way
        self.way_idx = np.random.choice(self.class_idx, self.way)