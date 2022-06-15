# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.get_cifar10 import get_CIFAR10
from datasets.get_cifar100 import get_CIFAR100
from argparse import Namespace

NAMES = {
    get_CIFAR10.NAME: get_CIFAR10,
    get_CIFAR100.NAME: get_CIFAR100,
}    

def get_dataset(args: Namespace):
    """
    Creates and returns the dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the dataset
    """
    assert args.dataset_kwargs['dataset'] in NAMES.keys()
    return NAMES[args.dataset_kwargs['dataset']](args)
