import argparse
import torch.backends.cudnn as cudnn
from data.get_data_set import get_data
from termcolor import colored
from utils.engine import train_model, evaluation
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DistributedSampler
import utils2.misc as misc
from configs import get_args_parser
import time
import datetime
from utils2.misc import NativeScalerWithGradNormCount as NativeScaler
from model.get_model import model_generation


def main():
    # distribution
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True

    train_set, val_set, ignore_index = get_data(args)
    best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iou': 0}

    model = model_generation(args)
    model.to(device).train()
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
        sampler_train = DistributedSampler(train_set)
        sampler_val = DistributedSampler(val_set, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=sampler_val, num_workers=args.num_workers, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index).cuda()
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    print(optimizer)
    loss_scaler = NativeScaler()

    print(f"Start training for {args.num_epoch} epochs")
    start_time = time.time()

    for epoch in range(args.num_epoch):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        print(colored('Epoch %d/%d' % (epoch + 1, args.num_epoch), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        print("start training: ")
        train_model(args, epoch, model, train_loader, criterion, optimizer, loss_scaler, device)

        print("start evaluation: ")
        evaluation(args, best_record, epoch, model, model_without_ddp, val_loader, criterion, device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main()