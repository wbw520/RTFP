import argparse
import torch.backends.cudnn as cudnn
import torch
from PIL import Image
from data.loader_tools import get_standard_transformations
import utils2.misc as misc
from configs import get_args_parser
from model.get_model import model_generation
from utils.engine import inference_sliding
from data.cityscapes import ColorTransition
from data.facade import PolygonTrans
import matplotlib.pyplot as plt
import numpy as np
import os


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


def show_single(image, location=None, save=False):
    # show single image
    image = np.array(image, dtype=np.uint8)
    plt.imshow(image)

    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    # if save:
    #     plt.savefig("demo/" + img_name, bbox_inches='tight', pad_inches=0)
    plt.show()


def main():
    # distribution
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True
    model = model_generation(args)
    model.to(device)
    checkpoint = torch.load(args.output_dir + args.dataset + "_" + args.model_name + ".pt", map_location="cuda:1")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    standard_transformations = get_standard_transformations()
    img = Image.open(img_path).convert('RGB')
    img = img.resize((args.setting_size[1], args.setting_size[0]), Image.BILINEAR)
    img = standard_transformations(img).to(device, dtype=torch.float32)
    pred, full_pred = inference_sliding(args, model, img.unsqueeze(0))
    color_img = PolygonTrans().id2trainId(torch.squeeze(pred, dim=0).cpu().detach().numpy())
    show_single(color_img, save=True)


if __name__ == '__main__':
    os.makedirs('demo/', exist_ok=True)
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.num_classes = 10
    img_path = "/home/wangbowen/DATA/Facade/zhao_translated_data/images/IMG_1282.jpg"
    main()