import sys
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import mae_model as models_mae

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clamp((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')


def prepare_model(chkpt_dir, arch='mae_vit_large_patch8'):
    # build mode
    model = models_mae.__dict__[arch](img_size=640)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()


# load an image
img = Image.open("/home/wangbowen/DATA/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000362_000019_leftImg8bit.png")
img = crop_center(img, 768, 768)
img = img.resize((640, 640))
img = np.array(img) / 255.

# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std

# plt.rcParams['figure.figsize'] = [5, 5]
# show_image(torch.tensor(img))
# plt.show()

# This is an MAE model trained with pixels as targets for visualization (ViT-Large, training mask ratio=0.75)

model_mae_gan = prepare_model('save_model/8_640_mae_pre_checkpoint-179.pth', 'mae_vit_large_patch8')
print('Model loaded.')

# torch.manual_seed(2)
print('MAE with extra GAN loss:')
run_one_image(img, model_mae_gan)