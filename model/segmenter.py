from model.vit_model import vit_encoder
from model.segmenter_decoder import sg_vit_mask_decoder
import torch.nn as nn
import torch
import os
import torch.nn.functional as F


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


def create_segmenter(args):
    encoder = vit_encoder(img_size=args.crop_size[0], patch_size=args.patch_size, embed_dim=args.encoder_embed_dim, depth=args.encoder_depth, num_heads=args.encoder_num_head)
    check_point = torch.load(os.path.join("save_model", args.pre_model), map_location="cuda:0")
    state_dict = ["decoder", "mask_token"]
    record = []
    for k, v in check_point.items():
        if state_dict[0] in k or state_dict[1] in k:
            record.append(k)
    for item in record:
        del check_point[item]
    encoder.load_state_dict(check_point, strict=True)
    print("load pre-trained weight from: ", args.pre_model)

    decoder = sg_vit_mask_decoder(patch_size=args.patch_size, encoder_embed_dim=args.encoder_embed_dim,
                                  decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_head, n_cls=args.num_classes)
    model = Segmenter(encoder, decoder)

    return model