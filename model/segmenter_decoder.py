import torch.nn as nn
import torch
from timm.models.vision_transformer import Block
from einops import rearrange
from timm.models.layers import trunc_normal_


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MaskTransformer(nn.Module):
    def __init__(
        self, patch_size=16, encoder_embed_dim=1024, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., n_cls=0, drop_path_rate=0.
    ):
        super().__init__()
        self.d_encoder = encoder_embed_dim
        self.patch_size = patch_size
        self.n_depth = decoder_depth
        self.n_cls = n_cls
        self.d_model = decoder_embed_dim
        self.mlp_ratio = mlp_ratio
        self.scale = decoder_embed_dim ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
        self.blocks = nn.ModuleList(
            [Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, drop_path=dpr[i]) for i in range(decoder_depth)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, decoder_embed_dim))
        self.proj_dec = nn.Linear(encoder_embed_dim, decoder_embed_dim)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(decoder_embed_dim, decoder_embed_dim))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(decoder_embed_dim, decoder_embed_dim))

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


def sg_vit_mask_decoder(**kwargs):
    model = MaskTransformer(mlp_ratio=4., **kwargs)
    return model