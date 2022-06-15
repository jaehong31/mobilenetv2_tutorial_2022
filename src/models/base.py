from models.utils.model import Model

class Base(Model):
    NAME = 'base'
    def __init__(self, backbone, loss, args, len_train_loader, logger):
        super(Base, self).__init__(backbone, loss, args, len_train_loader, logger)

    def observe(self, inputs, labels):
        self.opt.zero_grad()
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
           
        pred = self.net(inputs)
        loss = self.loss(pred, labels)
        data_dict = {'loss': loss.item()}
        data_dict['penalty'] = 0.0
               
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.lr_scheduler.step()
        
        data_dict.update({'lr': self.lr_scheduler.get_lr()})
        return data_dict