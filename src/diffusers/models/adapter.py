# modify from https://github.com/TencentARC/T2I-Adapter/blob/main/ldm/modules/encoders/adapter.py
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import Downsample2D
from .modeling_utils import ModelMixin, Sideloads
from ..configuration_utils import ConfigMixin, register_to_config


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize//2
        if in_c != out_c or sk==False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk==False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample2D(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None: # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class Adapter(ModelMixin, ConfigMixin):
    DEFAULT_TARGET = [
        "down_blocks.0.attentions.1",
        "down_blocks.1.attentions.1",
        "down_blocks.2.attentions.1",
        "down_blocks.3.resnets.1",
    ]
    
    @register_to_config
    def __init__(
            self, 
            block_out_channels=[320, 640, 1280, 1280],
            num_res_blocks=3,
            channels_in=64,
            kerenl_size=3,
            res_block_skip=False, 
            use_conv=False,
            target_layers=DEFAULT_TARGET,
            input_scale_factor=8,
        ):
        super(Adapter, self).__init__()
        
        self.num_downsample_blocks = len(block_out_channels)
        self.unshuffle = nn.PixelUnshuffle(input_scale_factor)
        self.block_out_channels = block_out_channels
        self.target_layers = target_layers
        self.num_res_blocks = num_res_blocks
        self.body = []
        
        for i in range(self.num_downsample_blocks):
            for j in range(num_res_blocks):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(
                            block_out_channels[i-1],
                            block_out_channels[i],
                            down=True,
                            ksize=kerenl_size,
                            sk=res_block_skip,
                            use_conv=use_conv
                        )
                    )
                else:
                    self.body.append(
                        ResnetBlock(
                            block_out_channels[i],
                            block_out_channels[i],
                            down=False,
                            ksize=kerenl_size,
                            sk=res_block_skip,
                            use_conv=use_conv
                        )
                    )
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(channels_in, block_out_channels[0], 3, 1, 1)

    def forward(self, x: torch.Tensor):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.conv_in(x)
        for i in range(self.num_downsample_blocks):
            for j in range(self.num_res_blocks):
                idx = i * self.num_res_blocks + j
                x = self.body[idx](x)
            features.append(x)

        return Sideloads({
            layer_name: h 
            for layer_name, h in zip(self.target_layers, features)
        })


class MultiAdapter(ModelMixin, ConfigMixin):
    
    @register_to_config
    def __init__(
            self,
            num_adapter=2,
            adapter_weights=None,
            block_out_channels=[320, 640, 1280, 1280],
            num_res_blocks=3,
            channels_in=64,
            kerenl_size=3,
            res_block_skip=False, 
            use_conv=False,
            target_layers=Adapter.DEFAULT_TARGET,
            input_scale_factor=8,
        ):
        super(MultiAdapter, self).__init__()

        self.adapters = nn.ModuleList([
            Adapter(
                block_out_channels=block_out_channels,
                num_res_blocks=num_res_blocks,
                channels_in=channels_in,
                kerenl_size=kerenl_size,
                res_block_skip=res_block_skip,
                use_conv=use_conv,
                target_layers=target_layers,
                input_scale_factor=input_scale_factor,
            ) for _ in range(num_adapter)
        ])
        if adapter_weights is None:
            self.adapter_weights = [1 / num_adapter] * num_adapter
        else:
            self.adapter_weights = adapter_weights
    
    def forward(self, xs: Iterable[torch.Tensor]):
        accume_state = 0
        for x, w, adapter in zip(xs, self.adapter_weights, self.adapters):
            accume_state += w * adapter(x)
        return accume_state
