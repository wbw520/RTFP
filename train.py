from torch.utils.tensorboard import SummaryWriter
import argparse
from configs import get_args_parser
import os
from data import cityscapes
from model.losses import CrossEntropyLoss2d
from utils.base_tools import check_mkdir
from torch.utils.data import DataLoader
from model.psp_net import PSPNet
from data.loader_tools import get_joint_transformations, get_standard_transformations


def main():
    args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iou': 0}
    writer = SummaryWriter(os.path.join(args.ckpt_path, 'exp', args.exp_name))

    model = PSPNet(num_classes=cityscapes.num_classes)
    model.cuda().train()

    joint_transformations = get_joint_transformations(args)
    standard_transformations = get_standard_transformations()
    train_set = cityscapes.CityScapes(args, 'fine', 'train', joint_transform=joint_transformations, standard_transform=standard_transformations)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_set = cityscapes.CityScapes(args, 'fine', 'val', joint_transform=joint_transformations, standard_transform=standard_transformations)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=cityscapes.ignore_label).cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main()