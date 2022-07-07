import argparse
import torch
import utils2.misc as misc
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DistributedSampler
from data import cityscapes
from data.loader_tools import get_joint_transformations, get_standard_transformations
from model.mae_model import mae_vit
from utils2.misc import NativeScalerWithGradNormCount as NativeScaler
import timm.optim.optim_factory as optim_factory
import os
import json
import datetime
import math
import sys
from typing import Iterable
import utils2.lr_sched as lr_sched
from utils2.pos_embed import interpolate_pos_embed
import time


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--use_pre', default=True, type=bool)

    # Model parameters
    parser.add_argument("--crop_size", type=int, default=[640, 640],
                        help="crop size for training and inference slice.")
    parser.add_argument("--stride_rate", type=float, default=0.5, help="stride ratio.")

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # VIT settings
    parser.add_argument("--patch_size", type=int, default=16, help="define the patch size.")
    parser.add_argument("--encoder_embed_dim", type=int, default=768, help="dimension for encoder.")
    parser.add_argument("--decoder_embed_dim", type=int, default=512, help="dimension for decoder.")
    parser.add_argument("--encoder_depth", type=int, default=12, help="depth for encoder.")
    parser.add_argument("--decoder_depth", type=int, default=8, help="depth for decoder.")
    parser.add_argument("--encoder_num_head", type=int, default=12, help="head number for encoder.")
    parser.add_argument("--decoder_num_head", type=int, default=16, help="head number for decoder.")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--root', default="/home/wangbowen/DATA/cityscapes", type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='save_model',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='save_model',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main():
    # distribution
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True
    model = mae_vit(args)
    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    joint_transformations = get_joint_transformations(args)
    standard_transformations = get_standard_transformations()
    train_set = cityscapes.CityScapes(args, 'fine', 'train', joint_transform=joint_transformations,
                                      standard_transform=standard_transformations)

    if args.use_pre:
        # use the pre-trained parameter from mae paper
        checkpoint = torch.load("save_model/mae_pretrain_vit_base.pth", map_location='cpu')
        checkpoint_model = checkpoint['model']
        interpolate_pos_embed(model, checkpoint_model)
        model_without_ddp.load_state_dict(checkpoint_model, strict=False)
        print("load pre-trained model")

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = DistributedSampler(train_set)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=args.num_workers, pin_memory=True)

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 16

    print("base lr: %.2e" % (args.lr * 16 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    print(f"Start training for {args.num_epochs} epochs")
    start_time = time.time()

    for epoch in range(args.num_epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        if args.output_dir and ((epoch + 1) % 10 == 0 or epoch + 1 == args.num_epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples["images"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        #     """ We use epoch_1000x as the x-axis in tensorboard.
        #     This calibrates different curves when batch size changes.
        #     """
        #     epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        #     log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
        #     log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model mae pre-training', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main()