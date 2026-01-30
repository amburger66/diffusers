import time
import os
import PIL
from PIL import Image
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

DIR = "/home/amli/research/diffusers/examples/instruct_pix2pix"

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "/home/amli/research/diffusers/examples/instruct_pix2pix/robotsmith-flatten-wm",
    torch_dtype=torch.float16,
).to("cuda")
generator = torch.Generator("cuda").manual_seed(42)

# image = load_image(
#     # "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png"
#     "https://huggingface.co/datasets/amburger66/robotsmith-examples/resolve/main/00026.png"
# )
img_path = "/home/amli/Desktop/input2.png"
image = Image.open(img_path).convert("RGB")

prompt = "flatten the dough to a height smaller than 0.03"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

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
edited_image.save(os.path.join(DIR, "output2.png"))
