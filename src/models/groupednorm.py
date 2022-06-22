from models.utils.model import Model
import torch

class GroupedNorm(Model):
    NAME = 'groupednorm'
    def __init__(self, backbone, loss, args, len_train_loader, logger):
        super(GroupedNorm, self).__init__(backbone, loss, args, len_train_loader, logger)
        self.gn = self.args.hyperparameters.GROUPEDNORM

    def penalty(self):
        get_weights = [weight for name, weight in self.net.named_parameters() if 'weight' in name and ('conv' in name or 'fc' in name)]       
        gsnorm = [torch.norm(torch.norm(w, dim=0), 1) if len(w.shape) == 2 else torch.norm(torch.norm(w.view(w.shape[0],-1), dim=1), 1) for w in get_weights]
        return float(self.gn.hyp) * torch.mean(torch.stack(gsnorm))


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