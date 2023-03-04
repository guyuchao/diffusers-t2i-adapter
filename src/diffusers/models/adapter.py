from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from .modeling_utils import ModelMixin, Sideloads
from .resnet import Downsample2D


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk is False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk is False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down is True:
            self.down_opt = Downsample2D(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down is True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
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
        block_out_channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 3,
        channels_in: int = 64,
        kerenl_size: int = 3,
        res_block_skip: bool = False,
        use_conv: bool = False,
        target_layers: List[str] = DEFAULT_TARGET,
        input_scale_factor: int = 8,
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
                            block_out_channels[i - 1],
                            block_out_channels[i],
                            down=True,
                            ksize=kerenl_size,
                            sk=res_block_skip,
                            use_conv=use_conv,
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
                            use_conv=use_conv,
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

        return Sideloads({layer_name: h for layer_name, h in zip(self.target_layers, features)})


class MultiAdapter(ModelMixin, ConfigMixin):
    ignore_for_config = ['adapters']
    default_adapter_kwargs = {
        "block_out_channels": [320, 640, 1280, 1280],
        "num_res_blocks": 3,
        "channels_in": 64,
        "kerenl_size": 3,
        "res_block_skip": False,
        "use_conv": False,
        "target_layers": Adapter.DEFAULT_TARGET,
        "input_scale_factor": 8,
    }

    @register_to_config
    def __init__(
        self,
        num_adapter: int = 2,
        adapters_kwargs: List[Dict[str, Any]] = [default_adapter_kwargs] * 2,
        adapters: Optional[List[Adapter]] = None,
        adapter_weights: Optional[List[float]] = None,
    ):
        super(MultiAdapter, self).__init__()

        self.num_adapter = num_adapter
        if adapters is None:
            self.adapters = nn.ModuleList(
                [
                    Adapter(**kwargs)
                    for kwargs in adapters_kwargs
                ]
            )
        else:
            self._check_adapter_config(adapters_kwargs, adapters)
            self.adapters = nn.ModuleList(adapters)
        if adapter_weights is None:
            self.adapter_weights = nn.Parameter(torch.tensor([1 / num_adapter] * num_adapter))
        else:
            self.adapter_weights = nn.Parameter(torch.tensor(adapter_weights))
    
    def _check_adapter_config(self, adapters_kwargs: List[Dict[str, Any]], adapters: List[Adapter]):
        for i, (init_kwargs, adapter) in enumerate(zip(adapters_kwargs, adapters)):
            config = adapter.config
            for k, v in init_kwargs.items():
                if v != config[k]:
                    raise ValueError(
                        f"keyword argument \"{k}\" from adapters_kwargs of {i}'th adapter dont match the Adapter instance's config!"
                        f"  {v} != {config[k]}"
                    )
    
    @classmethod
    def from_adapters(cls, adapters: List[Adapter], adapter_weights: Optional[List[float]] = None):
        
        def get_public_kwargs(kwargs):
            return {
                k: v for k, v in kwargs.items()
                if not k.startswith('_')
            }
        
        adapters_kwargs = [get_public_kwargs(adapter.config) for adapter in adapters]
        multi_adapter = cls(
            num_adapter=len(adapters),
            adapters_kwargs=adapters_kwargs,
            adapters=adapters,
            adapter_weights=adapter_weights,
        )
        return multi_adapter

    def forward(self, xs: torch.Tensor):
        if xs.shape[1] % self.num_adapter != 0:
            raise ValueError(
                f"Expecting multi-adapter's input have number of channel that cab be evenly divisible "
                f"by num_adapter: {xs.shape[1]} % {self.num_adapter} != 0"
            )
        x_list = torch.chunk(xs, self.num_adapter, dim=1)
        accume_state = None
        for x, w, adapter in zip(x_list, self.adapter_weights, self.adapters):
            sideload = adapter(x)
            if accume_state is None:
                accume_state = Sideloads({layer_name: h * w for layer_name, h in sideload.items()})
            else:
                for layer_name in sideload.keys():
                    accume_state[layer_name] += w * sideload[layer_name]
        return accume_state
