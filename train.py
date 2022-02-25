from torch.utils.tensorboard import SummaryWriter
import argparse
from configs import get_args_parser
import os
from data import cityscapes
from termcolor import colored
from utils.engine import train_model, evaluation
from torch.utils.data import DataLoader
import torch
from model.psp_net import PSPNet
from data.loader_tools import get_joint_transformations, get_standard_transformations
from torch.utils.data import DistributedSampler
import utils.distribute as dist
from pathlib import Path


def main():
    dist.init_distributed_mode(args)
    device = torch.device(args.device)
    model = PSPNet(num_classes=cityscapes.num_classes)
    model.to(device).train()
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iou': 0}
    writer = SummaryWriter(os.path.join(args.save_summary, 'exp', "summery"))

    criterion = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=cityscapes.ignore_label).cuda()

    optimizer = torch.optim.SGD([
        {'params': [param for name, param in model_without_ddp.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args.lr},
        {'params': [param for name, param in model_without_ddp.named_parameters() if name[-4:] != 'bias'],
         'lr': args.lr, 'weight_decay': args.weight_decay}
    ], momentum=args.momentum, nesterov=True)

    joint_transformations = get_joint_transformations(args)
    standard_transformations = get_standard_transformations()
    train_set = cityscapes.CityScapes(args, 'fine', 'train', joint_transform=joint_transformations,
                                      standard_transform=standard_transformations)
    val_set = cityscapes.CityScapes(args, 'fine', 'val', joint_transform=joint_transformations,
                                    standard_transform=standard_transformations)

    if args.distributed:
        sampler_train = DistributedSampler(train_set)
        sampler_val = DistributedSampler(val_set, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=sampler_val, num_workers=args.num_workers, shuffle=False)

    for epoch in range(args.num_epoch):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        print(colored('Epoch %d/%d' % (epoch + 1, args.num_epoch), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        print("start training: ")
        train_model(args, epoch, model, train_loader, criterion, optimizer, writer, device)

        print("start evaluation: ")
        evaluation(args, best_record, epoch, model, model_without_ddp, val_loader, criterion, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main()