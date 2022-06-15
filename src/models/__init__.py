import os
import importlib
import torch
from .backbones import mobilenetv2

def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

def get_model(args, len_train_loader, logger):
    loss = torch.nn.CrossEntropyLoss()    
    print('backbone info: %s'%(args.model.backbone))
    backbone = eval(f"{args.model.backbone}")(int(args.dataset.num_classes))
    
    names = {}
    for model in get_all_models():
        mod = importlib.import_module('models.' + model)
        class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
        names[model] = getattr(mod, class_name)

    return names[args.model.method.lower()](backbone, loss, args, len_train_loader, logger)
