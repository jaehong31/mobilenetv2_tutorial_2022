from arguments import get_args
from models import get_model
from datasets import get_dataset
import time, datetime
import torch.distributed as dist
from logger import create_logger
import numpy as np
import torch    
import copy
    
@torch.no_grad()
def evaluate(model, test_loader, logger, loss=torch.nn.CrossEntropyLoss()):    
    tmp_model = sparsify_weights(args, model, logger)

    corrects, totals = 0, 0
    for images, labels in test_loader:
        preds = tmp_model(images.to(args.device))
        test_loss = loss(preds, labels.to(args.device))
        
        preds = preds.argmax(dim=1)
        correct = (preds == labels.to(args.device)).sum().item()               
        
        corrects += correct
        totals += preds.shape[0]
    
    del tmp_model
    torch.cuda.empty_cache()
    logger.info(f'Accuracy: {(corrects/totals)*100:.2f} % ({corrects}/{totals}), Test Loss: {test_loss:.4f}')


def sparsify_weights(args, model, logger):
    # TODO 
    # 1. if pruning methods are applied, update the weights to be sparse given threshold (args.hyperparameters.XXXX.thr)
    # 2. print weight sparsity (%) using logger.info() (excluding batchnorm params and biases)
    if args.model.method == 'BASE':
      return model
    else:
      nonzeros, totals = 0, 0
      tmp_model = copy.deepcopy(model)
      tmp_dicts = tmp_model.state_dict()
      model_keys = tmp_dicts.keys()
      wkeys = [key for key in model_keys if 'weight' in key and ('conv' in key or 'fc' in key)]

      if args.model.method == 'L1NORM':  
        thr = float(args.hyperparameters.L1NORM.thr)
        for k in wkeys:
          tmp_dicts[k] = torch.where(torch.abs(tmp_dicts[k]) > thr, tmp_dicts[k], torch.zeros_like(tmp_dicts[k]))
          totals += np.prod(tmp_dicts[k].shape)
          nonzeros += torch.count_nonzero(tmp_dicts[k]).item()
      
      elif args.model.method == 'GROUPEDNORM':
        thr = float(args.hyperparameters.GROUPEDNORM.thr)
        for k in wkeys:
          kshape = tmp_dicts[k].shape
        
          # fc == classifier
          if len(tmp_dicts[k].shape) == 2:
            cond = torch.mean(torch.abs(tmp_dicts[k]), dim=0) > thr
            cond = cond.view(1,-1).expand(kshape[0], -1)
          # convs
          else:
            cond = torch.mean(torch.abs(tmp_dicts[k]), dim=[1,2,3]) > thr
            cond = cond.view(-1,1,1,1).expand(-1,kshape[1],kshape[2],kshape[3])

          tmp_dicts[k] = torch.where(cond, tmp_dicts[k], torch.zeros_like(tmp_dicts[k]))          
          totals += np.prod(tmp_dicts[k].shape)
          nonzeros += torch.count_nonzero(tmp_dicts[k]).item()
      
      else:
        NotImplementedError()
      tmp_model.load_state_dict(tmp_dicts)      
      logger.info(f'sparsity {(1-nonzeros/totals) * 100:.2f} % ({totals-nonzeros}/{totals}), {args.model.method} thr: {thr}')
      return tmp_model  
  
def main(args):
    dataset = get_dataset(args)
    train_loader, test_loader = dataset.get_data_loaders() 
    len_train_loader = len(train_loader)
    model = get_model(args, len_train_loader, logger)      
    
    if hasattr(model, 'set_task'):
        logger.info(f'set task')      
        model.set_task()
      
    start_time = time.time()
    for epoch in range(0, args.train.num_epochs):
        model.train()
        tr_losses, tr_p_losses = 0., 0.
        
        # training phase
        start = time.time()
        for idx, (images, labels) in enumerate(train_loader):
            data_dict = model.observe(images, labels)
            tr_losses += data_dict['loss']
            tr_p_losses += data_dict['penalty']                    
        epoch_time = time.time() - start
        
        if (epoch + 1) % args.eval.interval_epochs == 0:
            evaluate(model, test_loader, logger)
    
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
        logger.info('LR: {}, \
                            TR_LOSS: {}, TR_P_LOSS: {}'.format(
                                np.round(data_dict['lr'],6), 
                                np.round(tr_losses/len_train_loader, 4), 
                                np.round(tr_p_losses/len_train_loader, 4)))
    
    if hasattr(model, 'end_task'):
        logger.info(f'end task')      
        model.set_task()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'TOTAL TRAINING TIME {total_time_str}')  
    
    
if __name__ == "__main__":
    args = get_args()
    logger = create_logger(output_dir=args.log_dir, name=f"{args.tag}")   
    args.train.base_lr = float(args.train.base_lr) * args.train.batch_size / 512 
    args.train.warmup_lr = float(args.train.warmup_lr) * args.train.batch_size / 512 
    args.train.min_lr = float(args.train.min_lr) * args.train.batch_size / 512 
    main(args=args)    