import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="PSP-Net Network", add_help=False)

    # train settings
    parser.add_argument("--dataset", type=str, default="cityscapes")
    parser.add_argument("--model_name", type=str, default="PSPNet")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--root", type=str, default="/home/wangbowen/DATA/cityscapes",
                        help="Path to the directory containing the image list.")
    parser.add_argument("--crop_size", type=int, default=[640, 640],
                        help="crop size for training and inference slice.")
    parser.add_argument("--stride_rate", type=float, default=0.5, help="stride ratio.")
    parser.add_argument("--num_epoch", type=int, default=60, help="Number of training steps.")
    parser.add_argument("--num_classes", type=int, default=19, help="Number of class for dataset.")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # optimizer settings
    parser.add_argument("--lr", type=float, default=0.0001, help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="learning rate decay.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay.")

    # VIT settings
    parser.add_argument("--patch_size", type=int, default=8, help="define the patch size.")
    parser.add_argument("--encoder_embed_dim", type=int, default=768, help="dimension for encoder.")
    parser.add_argument("--decoder_embed_dim", type=int, default=512, help="dimension for decoder.")
    parser.add_argument("--encoder_depth", type=int, default=12, help="depth for encoder.")
    parser.add_argument("--decoder_depth", type=int, default=2, help="depth for decoder.")
    parser.add_argument("--encoder_num_head", type=int, default=12, help="head number for encoder.")
    parser.add_argument("--decoder_num_head", type=int, default=8, help="head number for decoder.")

    # other settings
    parser.add_argument("--save_summary", type=str, default="save_model")
    parser.add_argument("--print_freq", type=str, default=5, help="print frequency.")
    parser.add_argument('--output_dir', default='save_model/', help='path where to save, empty for no saving')

    # # distributed training parameters
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--device", type=str, default='cuda',
                        help="choose gpu device.")
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser