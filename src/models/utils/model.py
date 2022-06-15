import torch.nn as nn
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from ..optimizers import get_optimizer, LR_Scheduler, MultiStepLRScheduler

class Model(nn.Module):
    """
    Training model.
    """
    NAME = None
    def __init__(self, backbone: nn.Module, loss: nn.Module,
            args: Namespace, len_train_lodaer, logger):
        super(Model, self).__init__()

        self.net = backbone
        self.logger = logger
        self.loss = loss
        self.args = args
        self.len_train_lodaer = len_train_lodaer
        
        self.device = get_device()

    def forward(self, x: torch.Tensor):
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net.forward(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor):
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

    def set_optimizer_and_lr_scheduler(self):        
        wamup_lr = self.args.train.warmup_lr
        base_lr = self.args.train.base_lr
        final_lr = self.args.train.min_lr

        self.logger.info(f"optimizer_type: {self.args.train.optimizer.name}, \
                        warmup_lr: {wamup_lr}, \
                        base_lr: {base_lr}, \
                        final_lr: {final_lr}")
               
        self.opt = get_optimizer(
            self.args.train.optimizer.name, self.net,
            lr=base_lr,
            momentum=self.args.train.optimizer.momentum,
            weight_decay=self.args.train.optimizer.weight_decay
        )                        
        self.logger.info('OPTIMIZER is defined.')
        
        if self.args.train.lr_schedule.type == 'cosine':                    
            lr_scheduler = LR_Scheduler(
                optimizer=self.opt,
                warmup_epochs=self.args.train.warmup_epochs,
                num_epochs=self.args.train.num_epochs,
                warmup_lr=wamup_lr,                
                base_lr=base_lr,
                final_lr=final_lr,
                iter_per_epoch=self.len_train_lodaer
            )
        elif self.args.train.lr_schedule.type == 'multistep':
            warmup_steps = int(self.len_train_lodaer * self.args.train.warmup_epochs)
            decay_steps = int(self.len_train_lodaer * (self.args.train.num_epochs - self.args.train.warmup_epochs))
            
            assert False not in [self.args.train.warmup_epochs < milestone_epoch for milestone_epoch in self.args.train.lr_schedule.multisteps]            
            multi_steps = [(i-self.args.train.warmup_epochs) * self.len_train_lodaer for i in self.args.train.lr_schedule.multisteps]
            
            lr_scheduler = MultiStepLRScheduler(
                self.opt,
                milestones=multi_steps,
                gamma=self.args.train.lr_schedule.gamma,
                task_init_lr=base_lr,
                warmup_lr=wamup_lr,
                warmup_t=warmup_steps,
                decay_steps=decay_steps
            )
        else:
            raise NotImplementedError()
            
        return lr_scheduler

    def set_task(self):
        self.lr_scheduler = self.set_optimizer_and_lr_scheduler()
        self.net = self.net.to('cuda')
        
    def end_task(self):
        pass