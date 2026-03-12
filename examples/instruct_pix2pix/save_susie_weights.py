import os
import torch
from huggingface_hub import hf_hub_download
from flax.serialization import from_bytes

from diffusers import UNet2DConditionModel
from diffusers.models.modeling_pytorch_flax_utils import (
    load_flax_weights_in_pytorch_model,
)

MODEL_ID = "kvablack/susie"
SUBFOLDER = "unet"
OUT_DIR = "./susie-unet-pt"

# 1) Download the Flax checkpoint (msgpack)
msgpack_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="diffusion_flax_model.msgpack",
    subfolder=SUBFOLDER,
)

# 2) Load the UNet config from the repo and instantiate a Torch UNet with the *same architecture*
config = UNet2DConditionModel.load_config(MODEL_ID, subfolder=SUBFOLDER)
unet = UNet2DConditionModel.from_config(config)

# 3) Load Flax params from msgpack (no torch.load involved)
with open(msgpack_path, "rb") as f:
    flax_state = from_bytes(None, f.read())

# 4) Convert + load weights into the Torch UNet
unet = load_flax_weights_in_pytorch_model(unet, flax_state)

# 5) Save as PyTorch safetensors for reuse in your main torch-only env
os.makedirs(OUT_DIR, exist_ok=True)
unet.save_pretrained(OUT_DIR, safe_serialization=True)

print("Saved converted UNet to:", OUT_DIR)
