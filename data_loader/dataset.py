# -*- coding: utf-8 -*-

import os
import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data_loader.data_processor import DataProcessor
import h5py
import numpy as np

class PyTorchDataset(Dataset):
    def __init__(self, txt, config, transform=None, loader = None,
                 target_transform=None,  is_train_set=True):
        self.data = h5py.File(txt,'r')
        self.sen2 = self.data['sen2'][:,:,:,1:4]
        self.label = np.argmax(self.data['label'],axis=1)
        self.config = config
        self.DataProcessor = DataProcessor(self.config)
        self.transform = transform
        self.target_transform = target_transform
        self.is_train_set = is_train_set


    def __getitem__(self, index):
        label = self.label.__getitem__(index)
        image = self.sen2.__getitem__(index)
        image = self.self_defined_loader(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


    def __len__(self):
        return self.sen2.shape[0]


    def self_defined_loader(self, filename):
        image = self.DataProcessor.image_loader(filename)
        image = self.DataProcessor.image_resize(image)
        if self.is_train_set and self.config['data_aug']:
            image = self.DataProcessor.data_aug(image)
        image = self.DataProcessor.input_norm(image)
        return image


def get_data_loader(config):
    """
    
    :param config: 
    :return: 
    """
    train_data_file = config['train_data_file']
    test_data_file = config['val_data_file']
    batch_size = config['batch_size']
    num_workers =config['dataloader_workers']
    shuffle = config['shuffle']

    if not os.path.isfile(train_data_file):
        raise ValueError('train_data_file is not existed')
    if not os.path.isfile(test_data_file):
        raise ValueError('val_data_file is not existed')

    train_data = PyTorchDataset(txt=train_data_file,config=config,
                           transform=transforms.ToTensor(), is_train_set=True)
    test_data = PyTorchDataset(txt=test_data_file,config=config,
                                transform=transforms.ToTensor(), is_train_set=False)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader



