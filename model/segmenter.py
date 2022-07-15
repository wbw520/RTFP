import model.vit_model as vit
from model.segmenter_decoder import sg_vit_mask_decoder
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import re
from timm.models.helpers import load_checkpoint


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, im):
        H, W = im.size(2), im.size(3)

        x = self.encoder(im)
        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:]
        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)


def set_decoder_parameter(name):
    if "small" in name:
        encoder_dim = 384
    elif "base" in name:
        encoder_dim = 768
    elif "large" in name:
        encoder_dim = 1024
    elif "huge" in name:
        encoder_dim = 1280
    else:
        raise "type of encoder is not defined."

    patch_size = re.findall("\d+", name)

    return encoder_dim, int(patch_size[0])


def create_segmenter(args):
    encoder = vit.__dict__[args.encoder](img_size=args.crop_size[0])
    if "mae" not in args.pre_model:
        print("load pre-model trained by ImageNet")
        load_checkpoint(encoder, args.output_dir + args.pre_model)
    else:
        print("load pre-model trained by MAE")
        check_points = torch.load(os.path.join("save_model", args.pre_model), map_location="cpu")
        check_point = check_points["model"]
        state_dict = ["decoder", "mask_token"]
        record = []
        for k, v in check_point.items():
            if state_dict[0] in k or state_dict[1] in k:
                record.append(k)
        for item in record:
            del check_point[item]
        encoder.load_state_dict(check_point, strict=True)

    print("load pre-trained weight from: ", args.pre_model)

    encoder_embed_dim, patch_size = set_decoder_parameter(args.encoder)
    decoder = sg_vit_mask_decoder(patch_size=patch_size, encoder_embed_dim=encoder_embed_dim,
                                  decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_head, n_cls=args.num_classes)
    model = Segmenter(encoder, decoder)

    return model