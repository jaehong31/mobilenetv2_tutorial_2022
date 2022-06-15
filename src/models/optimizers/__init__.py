import torch
from .lr_scheduler import LR_Scheduler, MultiStepLRScheduler

def get_optimizer(name, model, lr, momentum, weight_decay):
    parameters = [{
        'name': 'base',
        'params': model.parameters(),
        'lr': lr
    }]
    if name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer