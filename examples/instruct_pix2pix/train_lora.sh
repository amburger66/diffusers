#!/bin/bash
set -eux
export NCCL_P2P_DISABLE=1  # To avoid some hanging in DDP/NCCL synchronization, but slows down training

accelerate launch train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=timbrooks/instruct-pix2pix \
    --pretrained_unet_model_name_or_path="/data/robotsmith/models/ip2p/task03/task03_12_1" \
    --preprocessed_train_data_dir="/data/robotsmith/datasets/task03_flatten_12_long_2/train_ds_v1" \
    --val_image_url=https://huggingface.co/datasets/amburger66/robotsmith-examples/resolve/main/00025.png \
    --validation_prompt="flatten the dough to a height smaller than 0.03" \
    --validation_epochs 6 \
    --output_dir="/data/robotsmith/models/ip2p/task03/task03_12_1_lora_max" \
    --max_train_steps=1000 \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --checkpointing_steps=200 \
    --checkpoints_total_limit=5 \
    --learning_rate=5e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=0 \
    --train_lora \
    --rank=8 \
    --lora_alpha=8 \
    --lora_dropout=0.0