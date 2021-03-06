from models.utils.model import Model
import torch

class L1Norm(Model):
    NAME = 'l1norm'
    def __init__(self, backbone, loss, args, len_train_loader, logger):
        super(L1Norm, self).__init__(backbone, loss, args, len_train_loader, logger)
        self.l1 = self.args.hyperparameters.L1NORM

    def penalty(self):
        l1norm = [torch.norm(weight, 1) for name, weight in self.net.named_parameters() if 'weight' in name and ('conv' in name or 'fc' in name)]       
        return float(self.l1.hyp) * torch.mean(torch.stack(l1norm))

    def observe(self, inputs, labels):
        self.opt.zero_grad()
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
           
        pred = self.net(inputs)
        loss = self.loss(pred, labels)
        penalty_loss = self.penalty()
        final_loss = loss + penalty_loss
        data_dict = {'loss': final_loss.item()}        
        data_dict['penalty'] = penalty_loss.item()
               
        self.opt.zero_grad()
        final_loss.backward()
        self.opt.step()
        self.lr_scheduler.step()
        
        data_dict.update({'lr': self.lr_scheduler.get_lr()})
        return data_dict