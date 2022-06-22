from models.utils.model import Model
import torch

class GroupedNorm(Model):
    NAME = 'groupednorm'
    def __init__(self, backbone, loss, args, len_train_loader, logger):
        super(GroupedNorm, self).__init__(backbone, loss, args, len_train_loader, logger)
        self.gn = self.args.hyperparameters.STRUCTUREDNORM

    def penalty(self):
        """_summary_        
        return structured_norm loss with hyperparameters
        """
        gsnorm = 0
        l1norm = [torch.norm(weight, 1) for name, weight in self.net.named_parameters() if 'weight' in name and ('conv' in name or 'fc' in name)]       
        import pdb; pdb.set_trace()
        print('2')
        return self.l1.hyp * torch.mean(gsnorm)


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