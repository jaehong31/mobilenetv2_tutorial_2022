from arguments import get_args
from models import get_model
from datasets import get_dataset
import time, datetime
import torch.distributed as dist
from logger import create_logger
import numpy as np
import torch    
    
@torch.no_grad()
def evaluate(model, test_loader, logger, loss=torch.nn.CrossEntropyLoss()):
    corrects, totals = 0, 0
    
    import pdb; pdb.set_trace()
    get_weights = [weight for name, weight in model.net.named_parameters() if 'weight' in name and ('conv' in name or 'fc' in name)]
    
    for images, labels in test_loader:
        preds = model(images.to(args.device))
        test_loss = loss(preds, labels.to(args.device))
        
        preds = preds.argmax(dim=1)
        correct = (preds == labels.to(args.device)).sum().item()

        corrects += correct
        totals += preds.shape[0]
    logger.info(f'Accuracy: {(corrects/totals)*100:.2f} % ({corrects}/{totals}), Test Loss: {test_loss:.4f}')
        
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
        start = time.time()
        model.train()

        tr_losses = 0.
        tr_p_losses = 0.
        # training phase
        for idx, (images, labels) in enumerate(train_loader):
            data_dict = model.observe(images, labels)
            tr_losses += data_dict['loss']
            tr_p_losses += data_dict['penalty']                
    
        if (epoch + 1) % args.eval.interval_epochs == 0:
            evaluate(model, test_loader, logger)
    
        epoch_time = time.time() - start
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