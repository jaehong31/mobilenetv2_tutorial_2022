import argparse
import os
import torch

import numpy as np
import torch
import random
import pdb

import re
import yaml

import shutil

from datetime import datetime

class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):
        raise AttributeError(f'Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!')


def set_deterministic(seed):
    # seed by default is None
    if seed != None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
def get_args():
    #parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser('mobilenetv2 training script', add_help=False)
    # basics
    parser.add_argument('--tag', type=str, default='', help='tag of experiment')

    # distributed training
    parser.add_argument('-c', '--config-file', default='configs/base_cifar10.yaml', type=str, metavar="FILE", help="path to yaml file")
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--log_dir', type=str, default='../logs')
    parser.add_argument('--ckpt_dir', type=str, default='../cache/')
    parser.add_argument('--device', type=str, default='cuda'  if torch.cuda.is_available() else 'cpu')
    
    parser.add_argument('--backbone', type=str, default='mobilenetv2', 
                                choices = [ 'mobilenetv2',
                                            ])
    
    parser.add_argument('--cl_model', type=str, default='BASE',
                                choices = [ 'BASE', 
                                           ])
    args = parser.parse_args()
    
    #config = get_config(args)
    #set_distributed(config)
    
    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]
    
    if hasattr(args.train, 'weight_decay'):
        args.train.weight_decay = float(args.train.weight_decay)
    if hasattr(args.train, 'momentum'):
        args.train.momentum = float(args.train.momentum)        
        
    if args.tag != '':
        args.name += '_'+args.tag
    
    args.log_dir = os.path.join(args.log_dir, 'in-progress_'+datetime.now().strftime('%m%d%H%M%S_')+args.name)
    if args.logger.datetime:
        args.ckpt_dir = os.path.join(args.ckpt_dir, datetime.now().strftime('%m%d%H%M%S_')+args.name)
    else:
        args.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
    
    os.makedirs(args.log_dir, exist_ok=True)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    shutil.copy2(args.config_file, args.log_dir)
    set_deterministic(args.seed)

    vars(args)['aug_kwargs'] = {
        'image_size': args.dataset.image_size
    }
    vars(args)['dataset_kwargs'] = {
        'dataset':args.dataset.name,
        'data_dir': args.data_dir,
        'download': True,
        'debug_subset_size': None,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }
    print('##########################################################################################################################################')
    print('args-name', args.name)
    print('args-log_dir', args.log_dir)
    print('args-ckpt_dir', args.ckpt_dir)
    print('##########################################################################################################################################')
    return args
