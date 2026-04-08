# Training examples

Example command for fine-tuning from initialization with **InstructPix2Pix** weights and **SuSIE** unet weights (empirically this seems to result in a better model than just using IP2P weights).

Things to double check:
- dataset_name
- val_image_url (may need to upload a new image)
- validation_prompt
- output_dir (important to not overwrite other models!)
```
conda activate diffusers-ip2p
accelerate launch train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=timbrooks/instruct-pix2pix \
    --pretrained_unet_model_name_or_path="./susie-unet-pt" \
    --dataset_name=/data/robotsmith/task03_flatten/wm_vlm_dataset \
    --val_image_url=https://huggingface.co/datasets/amburger66/robotsmith-examples/resolve/main/00004.png \
    --validation_prompt="flatten the dough to a height smaller than 0.03" \
    --validation_epochs 10 \
    --output_dir=robotsmith-flatten-vlm-susie \
    --num_train_epochs=1000 \
    --max_train_steps=2000 \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 \
    --random_flip \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=1 \
    --learning_rate=5e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42
```

Example command for fine-tuning from initialization with **InstructPix2Pix** weights:
```
conda activate diffusers-ip2p
accelerate launch train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=timbrooks/instruct-pix2pix \
    --dataset_name=/data/robotsmith/task03_flatten/wm_vlm_dataset \
    --val_image_url=https://huggingface.co/datasets/amburger66/robotsmith-examples/resolve/main/00004.png \
    --validation_prompt="flatten the dough to a height smaller than 0.03" \
    --validation_epochs 50 \
    --output_dir=robotsmith-flatten-vlm-wm \
    --num_train_epochs=1000 \
    --max_train_steps=2000 \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 \
    --random_flip \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=1 \
    --learning_rate=5e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42
```

## LoRA PEFT

See train_lora.sh.