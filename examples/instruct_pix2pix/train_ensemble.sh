#!/bin/bash
set -eux

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # To avoid some hanging in DDP/NCCL synchronization, but slows down training

for i in {0..4}; do
    accelerate launch --multi_gpu train_instruct_pix2pix.py \
        --pretrained_model_name_or_path=timbrooks/instruct-pix2pix \
        --pretrained_unet_model_name_or_path="./susie-unet-pt" \
        --preprocessed_train_data_dir=/data/robotsmith/datasets/task03_flatten_12/train_ds_v1 \
        --val_image_url=https://huggingface.co/datasets/amburger66/robotsmith-examples/resolve/main/00004.png \
        --validation_prompt="flatten the dough to a height smaller than 0.03" \
        --validation_epochs 6 \
        --output_dir="/data/robotsmith/models/ip2p/task03/task03_12_$i" \
        --max_train_steps=2000 \
        --enable_xformers_memory_efficient_attention \
        --resolution=256 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --gradient_checkpointing \
        --checkpointing_steps=preprocess_dataset.py \
        --checkpoints_total_limit=5 \
        --learning_rate=5e-05 \
        --max_grad_norm=1 \
        --lr_warmup_steps=0 \
        --conditioning_dropout_prob=0.05 \
        --mixed_precision=fp16 \
        --seed=$i
done