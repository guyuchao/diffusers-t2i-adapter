import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import *
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionAdapterPipeline, Adapter



def test_adapter():
    adapter_ckpt = "/home/ron/Downloads/t2iadapter_seg_sd14v1.pth"
    adapter = Adapter(
        channels=[320, 640, 1280, 1280][:4],
        channels_in=int(3 * 64), 
        num_res_blocks=2, 
        kerenl_size=1, 
        res_block_skip=True, 
        use_conv=False
    )
    weight = torch.load(adapter_ckpt)
    mapping = {}
    for k in weight.keys():
        print(k)
        if 'down_opt.op' in k:
            mapping[k] = k.replace('down_opt.op', 'down_opt.conv')
    print('mapping: ', mapping)
    for old, new in mapping.items():
        weight[new] = weight.pop(old)

    adapter.load_state_dict(weight)


def test_pipeline(device='cpu'):

    mask = Image.open("motor.png")
    prompt = [
        "A black Honda motorcycle parked in front of a garage",
        "A red-blue Honda motorcycle parked in front of a garage",
        "A green Honda motorcycle parked in a desert",
    ]

    # model_name = "CompVis/stable-diffusion-v1-4"
    model_name = "RzZ/sd-v1-4-adapter"
    revision = "keypose"

    if device =='cuda':
        pipe = StableDiffusionAdapterPipeline.from_pretrained(model_name, revision=revision, torch_dtype=torch.float16)
        pipe.to("cuda")
    else:
        pipe = StableDiffusionAdapterPipeline.from_pretrained(model_name, revision=revision, torch_dtype=torch.float32)


    images = pipe(prompt, [mask] * len(prompt)).images

    plt.subplot(2, 2, 1)
    plt.imshow(mask)

    for i, image in enumerate(images):
        plt.subplot(2, 2, 2 + i)
        plt.imshow(image)
        plt.title(prompt[i], fontsize=24)
    plt.show()


if __name__ == "__main__":
    # test_adapter()
    test_pipeline(device='cpu')