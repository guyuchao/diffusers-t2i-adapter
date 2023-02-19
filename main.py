import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import *
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionAdapterPipeline, Adapter


@torch.no_grad()
def get_color_masks(image: torch.Tensor) -> Dict[Tuple[int], torch.Tensor]:
    h, w, c = image.shape
    assert c == 3
    
    img_2d = image.view((-1, 3))
    colors, freqs = torch.unique(img_2d, return_counts=True, dim=0)
    colors = colors[freqs >= h]
    color2mask = {}
    for color in colors:
        mask = (image == color).float().max(dim=-1).values
        color = color.cpu().numpy().tolist()
        color2mask[tuple(color)] = mask
    return color2mask


mask = Image.open("motor.png")

prompt = [
    "A black Honda motorcycle parked in front of a garage",
    "A red-blue Honda motorcycle parked in front of a garage",
]


# model_name = "CompVis/stable-diffusion-v1-4"
model_name = "RzZ/sd-v1-4-adapter"
pipe = StableDiffusionAdapterPipeline.from_pretrained(model_name, torch_dtype=torch.float16)

# HACK: side load adapter module
# adapter_ckpt = "/home/ron/Downloads/t2iadapter_seg_sd14v1.pth"
# pipe.adapter = Adapter(
#     cin=int(3*64), 
#     channels=[320, 640, 1280, 1280][:4], 
#     nums_rb=2, 
#     ksize=1, 
#     sk=True, 
#     use_conv=False
# ).to('cuda')
# pipe.adapter.load_state_dict(torch.load(adapter_ckpt))

pipe.to("cuda")

images = pipe(prompt, [mask, mask]).images

for image in images:
    plt.imshow(image)
    plt.show()