import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class FeatureUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res_block = ResidualBlock(in_ch, out_ch)

    def forward(self, x_low, x_high):
        x_low = self.upsample(x_low)
        diffY = x_high.size()[2] - x_low.size()[2]
        diffX = x_high.size()[3] - x_low.size()[3]
        x_low = F.pad(x_low, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_high, x_low], dim=1)
        return self.res_block(x)


class PromptInjector(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        self.injector = nn.Sequential(
            nn.Linear(dim, 32), nn.GELU(),
            nn.Linear(32, dim), nn.GELU()
        )

    def forward(self, x):
        return self.block(x + self.injector(x))


class ContextualPyramidBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.paths = nn.ModuleList()
        path_cfgs = [
            [((1, 1), 1)],
            [((1, 3), 1), ((3, 1), 1), ((3, 3), 3)],
            [((1, 5), 1), ((5, 1), 1), ((3, 3), 5)],
            [((1, 7), 1), ((7, 1), 1), ((3, 3), 7)]
        ]
        for cfg in path_cfgs:
            layers = [ResidualBlock(in_ch, out_ch)]
            for k, d in cfg:
                pad = d if k == (3, 3) else tuple(k_i // 2 for k_i in k)
                layers.append(ResidualBlock(out_ch, out_ch))
            self.paths.append(nn.Sequential(*layers))

        self.concat = nn.Conv2d(out_ch * 4, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        self.se = SEBlock(out_ch)

    def forward(self, x):
        feats = [p(x) for p in self.paths]
        x_cat = torch.cat(feats, dim=1)
        fused = self.concat(x_cat) + self.shortcut(x)
        return F.relu(self.se(fused))


class Network(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        cfg_file = "sam2_hiera_l.yaml"
        base = build_sam(cfg_file, checkpoint_path) if checkpoint_path else build_sam(cfg_file)

        for attr in ["sam_mask_decoder", "sam_prompt_encoder", "memory_encoder",
                     "memory_attention", "mask_downsample", "obj_ptr_tpos_proj",
                     "obj_ptr_proj", "image_encoder.neck"]:
            self._remove_nested_attr(base, attr)

        self.encoder = base.image_encoder.trunk

        for i, block in enumerate(self.encoder.blocks):
            if i >= len(self.encoder.blocks) - 4:
                for param in block.parameters():
                    param.requires_grad = True

        self.encoder.blocks = nn.Sequential(*[PromptInjector(b) for b in self.encoder.blocks])

        self.refine_blocks = nn.ModuleList([
            ContextualPyramidBlock(144, 64),
            ContextualPyramidBlock(288, 64),
            ContextualPyramidBlock(576, 64),
            ContextualPyramidBlock(1152, 64)
        ])

        self.decoder = nn.ModuleList([
            FeatureUp(128, 64), FeatureUp(128, 64), FeatureUp(128, 64)
        ])

        self.heads = nn.ModuleDict({
            'main': nn.Conv2d(64, 1, 1),
            'aux1': nn.Conv2d(64, 1, 1),
            'aux2': nn.Conv2d(64, 1, 1)
        })

    def _remove_nested_attr(self, obj, path):
        for p in path.split('.')[:-1]:
            obj = getattr(obj, p)
        delattr(obj, path.split('.')[-1])

    def forward(self, x):
        feats = self.encoder(x)
        refined = [r(f) for r, f in zip(self.refine_blocks, feats)]

        x = self.decoder[0](refined[3], refined[2])
        aux1 = F.interpolate(self.heads['aux1'](x), scale_factor=16, mode='bilinear')

        x = self.decoder[1](x, refined[1])
        aux2 = F.interpolate(self.heads['aux2'](x), scale_factor=8, mode='bilinear')

        x = self.decoder[2](x, refined[0])
        out = F.interpolate(self.heads['main'](x), scale_factor=4, mode='bilinear')

        return out, aux1, aux2
