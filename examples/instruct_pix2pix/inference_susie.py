"""
Script to test the SuSIE unet zero-shot inference.
"""

import time
import os
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel

DIR = "/home/amli/research/diffusers/examples/instruct_pix2pix"

# pre-downloaded weights from SuSIE unet
unet = UNet2DConditionModel.from_pretrained(
    "./examples/instruct_pix2pix/susie-unet-pt", torch_dtype=torch.float16
)

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
seed = 42
generator = torch.Generator("cuda").manual_seed(seed)

# img_path = "/data/robotsmith/task03_flatten/wm_vlm_dataset/test/00004.png"
img_path = "/data/robotsmith/task03_flatten/wm_vlm_dataset/test/00003_output.png"
image = Image.open(img_path).convert("RGB")

prompt = "flatten the dough to a height smaller than 0.03"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 7

start_time = time.time()
edited_image = pipeline(
    prompt,
    image=image,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]
time_taken = time.time() - start_time
print(f"Time taken: {time_taken} seconds")
edited_image.save(
    os.path.join(
        DIR,
        "outputs",
        f"susie-no-trace_{seed}_{num_inference_steps}_{image_guidance_scale}_{guidance_scale}.png",
    )
)
