import argparse
import torch.backends.cudnn as cudnn
import torch
from data.get_data_set import get_data
import utils2.misc as misc
from configs import get_args_parser
from utils.engine import evaluation_none_training
from torch.utils.data import DataLoader
from model.get_model import model_generation
import os


def main():
    # distribution
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True
    train_set, test_set, ignore_index = get_data(args)
    model = model_generation(args)
    model.to(device)

    if args.model_name == "Segmenter":
        save_name = args.model_name + "_" + args.encoder
    else:
        save_name = args.model_name

    checkpoint = torch.load(args.output_dir + args.dataset + "_" + save_name + ".pt", map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    print("load trained model finished.")

    sampler_val = torch.utils.data.SequentialSampler(test_set)
    val_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=sampler_val, num_workers=args.num_workers, shuffle=False)
    evaluation_none_training(args, model, val_loader, device)


if __name__ == '__main__':
    os.makedirs('demo/', exist_ok=True)
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main()