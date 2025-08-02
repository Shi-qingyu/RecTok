#!/bin/bash

# Training configuration
batch_size=256  # global batch size = batch_size x num_nodes x 4 = 1024
epochs=50
lr=3e-3
eval_freq=5

# Get current timestamp for unique experiment name
timestamp=$(date +%Y%m%d_%H%M)
exp_name="dinov2_linear_prob_${timestamp}"

echo "Starting DINOv2 Linear Probing Training..."
echo "Experiment: $exp_name"
echo "Batch size: $batch_size"
echo "Learning rate: $lr"
echo "Epochs: $epochs"

# Run training with torchrun for distributed training
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29501 \
    linear_prob_dinov2.py \
    --batch_size $batch_size \
    --lr $lr \
    --epochs $epochs \
    --eval_freq $eval_freq \
    --data_path ./data/imagenet/train \
    --output_dir ./work_dirs/dinov2_linear_prob \
    --project "dinov2_linear_probing" \
    --exp_name "$exp_name" \
    --num_workers 10 \
    --pin_mem \
    --weight_decay 0.0 \
    --print_freq 50 \
    --use_wandb \
    --img_size 224  # DINOv2 default image size 