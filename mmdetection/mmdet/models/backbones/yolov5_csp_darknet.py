import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmengine.model import BaseModule as BaseBackbone

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act_cfg=dict(type='SiLU', inplace=True)):
        super().__init__()
        self.conv = ConvModule(
            in_channels * 4, out_channels, kernel_size, stride, padding=kernel_size//2,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=act_cfg)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)

class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, add_identity=True, use_depthwise=False, act_cfg=dict(type='SiLU', inplace=True)):
        super().__init__()
        mid_channels = out_channels // 2
        self.main_conv = ConvModule(in_channels, mid_channels, 1, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=act_cfg)
        self.short_conv = ConvModule(in_channels, mid_channels, 1, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=act_cfg)
        self.final_conv = ConvModule(mid_channels * 2, out_channels, 1, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=act_cfg)

        self.blocks = nn.Sequential(*[
            ConvModule(mid_channels, mid_channels, 3, padding=1, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=act_cfg)
            for _ in range(num_blocks)
        ])
        self.add_identity = add_identity

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)

@MODELS.register_module()
class YOLOv5CSPDarknet(BaseBackbone):
    arch_settings = {
        'P5': [[64, 128, 3], [128, 256, 6], [256, 512, 9], [512, 1024, 3]],
    }

    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 act_cfg=dict(type='SiLU', inplace=True),
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 norm_eval=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        arch_setting = self.arch_settings[arch]
        self.arch_setting = arch_setting
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        self.stem = Focus(3, int(64 * widen_factor), 3, act_cfg=act_cfg)

        self.layers = ['stem']
        for i, (in_channels, out_channels, num_blocks) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            stage.append(conv_layer)
            csp_layer = CSPLayer(out_channels, out_channels, num_blocks=num_blocks, add_identity=False, use_depthwise=use_depthwise, act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            m = getattr(self, f'stage{i+1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()