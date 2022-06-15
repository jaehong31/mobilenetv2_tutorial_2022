# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import CIFAR100
from utils.conf import base_path
from augmentations import get_aug
from torch.utils.data import DataLoader

class get_CIFAR100:
    NAME = 'cifar100'
    def __init__(self, args):
        self.args = args
        
    def get_data_loaders(self):
        transform = get_aug(is_train=True)
        test_transform = get_aug(is_train=False)

        train_dataset = CIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)

        test_dataset = CIFAR100(base_path() + 'CIFAR100', train=False,
                                download=True, transform=test_transform)

        train_loader = DataLoader(train_dataset,
                              batch_size=self.args.train.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset,      
                                batch_size=self.args.train.batch_size, shuffle=False, num_workers=4)    
        return train_loader, test_loader


    def get_transform(self, is_train=True):
        transform = get_aug(is_train=is_train, transform_single=True, to_pil_image=True)
        return transform
