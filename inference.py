import argparse
import torch.backends.cudnn as cudnn
import torch
from data.get_data_set import get_data
from PIL import Image
from data.loader_tools import get_standard_transformations
import utils2.misc as misc
from utils.base_tools import get_name
from configs import get_args_parser
from model.get_model import model_generation
from utils.engine import inference_sliding
from data.facade import PolygonTrans
import matplotlib.pyplot as plt
import numpy as np
import os


def show_single(image, location=None, save=False, name=None):
    # show single image
    image = np.array(image, dtype=np.uint8)
    plt.imshow(image)

    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    if save:
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.show()


def main():
    # distribution
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True
    train_set, val_set, ignore_index = get_data(args)
    model = model_generation(args)
    model.to(device)

    if args.model_name == "Segmenter":
        save_name = args.model_name + "_" + args.encoder
    else:
        save_name = args.model_name

    checkpoint = torch.load(args.output_dir + args.dataset + "_" + save_name + ".pt", map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    standard_transformations = get_standard_transformations()
    img = Image.open(img_path).convert('RGB')

    img = img.resize((args.setting_size[1], args.setting_size[0]), Image.BILINEAR)
    img = standard_transformations(img).to(device, dtype=torch.float32)
    pred, full_pred = inference_sliding(args, model, img.unsqueeze(0))
    color_img = PolygonTrans().id2trainId(torch.squeeze(pred, dim=0).cpu().detach().numpy(), select=2)
    print(color_img.shape)
    show_single(color_img, save=True, name="color_mask2.png")


if __name__ == '__main__':
    os.makedirs('demo/', exist_ok=True)
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    img_path = "/home/wangbowen/DATA/Facade/translated_data/images/32052284_477d66a5ae_o.png"
    main()