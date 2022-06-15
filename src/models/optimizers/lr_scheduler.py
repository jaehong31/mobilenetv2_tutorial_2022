import numpy as np
from timm.scheduler.scheduler import Scheduler
from bisect import bisect_right

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        self.optimizer = optimizer
        self.current_lr = 0
        self.iter = 0
        
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr-final_lr) * (1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr
        return lr

    def step_update(self, index):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[index]
        self.iter = index
        self.current_lr = lr
        return lr

    def reset(self):
        self.iter = 0
        self.current_lr = 0

    def get_lr(self):
        return self.current_lr

    def set_lr(self, lr):
        self.current_lr = lr

class MultiStepLRScheduler(object):
    """
    # num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    # decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    multi_steps = [i * n_iter_per_epoch for i in config.TRAIN.LR_SCHEDULER.MULTISTEPS]

    lr_scheduler = MultiStepLRScheduler(
            optimizer,
            milestones=multi_steps,
            gamma=config.TRAIN.LR_SCHEDULER.GAMMA,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    """
    def __init__(self, optimizer, milestones, gamma=0.1, task_init_lr=0, warmup_lr=0, warmup_t=0, decay_steps=0):
        self.optimizer = optimizer
        warmup_lr_schedule = np.linspace(warmup_lr, task_init_lr, warmup_t)
        multistep_lr_schedule = [task_init_lr * (gamma ** bisect_right(milestones, t)) for t in range(decay_steps)]        
        self.lr_schedule = np.concatenate((warmup_lr_schedule, multistep_lr_schedule))
        
    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr
        return lr

    def step_update(self, index):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[index]
        self.iter = index
        self.current_lr = lr
        return lr

    def reset(self):
        self.iter = 0
        self.current_lr = 0

    def get_lr(self):
        return self.current_lr

    def set_lr(self, lr):
        self.current_lr = lr