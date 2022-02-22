import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="PSP-Net Network", add_help=False)

    # train settings
    parser.add_argument("--model_name", type=str, default="PSPNet")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default="/home/wangbowen/",
                        help="Path to the directory containing the image list.")
    parser.add_argument("--crop_size", type=int, default=[768, 768],
                        help="crop size for training and inference slice.")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="this kind of pixel will not used for both train and evaluation")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num_epoch", type=int, default=40,
                        help="Number of training steps.")

    # device settings
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--device", type=str, default='cuda',
                        help="choose gpu device.")
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser