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
    # "./examples/instruct_pix2pix/robotsmith-flatten-vlm-susie",
    "/data/robotsmith/models/ip2p/task03/task03_12_1",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
# Load LoRA weights
pipeline.load_lora_weights(
    "/data/robotsmith/models/ip2p/task03/task03_12_1_lora_max",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="default",
)

seed = 42
generator = torch.Generator("cuda").manual_seed(seed)

image = load_image(
    # "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png"
    "https://huggingface.co/datasets/amburger66/robotsmith-examples/resolve/main/00025.png"
)
# img_path = "/home/amli/Desktop/00006.png"
# img_path = "/data/robotsmith/task03_flatten/wm_vlm_dataset/test/00004.png"
# image = Image.open(img_path).convert("RGB")

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
        f"task03_12_1_lora_max_{seed}_{num_inference_steps}_{image_guidance_scale}_{guidance_scale}.png",
    )
)
