#!/bin/bash

# Training configuration
batch_size=512  # global batch size = batch_size x num_nodes x 8 = 1024
epochs=50
lr=3e-3
eval_freq=5

# Get current timestamp for unique experiment name
timestamp=$(date +%Y%m%d_%H%M)
exp_name="clip_vit_base_${timestamp}"

echo "Starting CLIP Linear Probing Training..."
echo "Experiment: $exp_name"
echo "Batch size: $batch_size"
echo "Learning rate: $lr"
echo "Epochs: $epochs"

# Run training with torchrun for distributed training
export WANDB_MODE=offline
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29501 \
    linear_prob_clip.py \
    --batch_size $batch_size \
    --lr $lr \
    --epochs $epochs \
    --eval_freq $eval_freq \
    --data_path ./data/imagenet/train \
    --output_dir ./work_dirs/clip_linear_prob \
    --project "linear_probing" \
    --exp_name "$exp_name" \
    --num_workers 10 \
    --pin_mem \
    --weight_decay 0.0 \
    --print_freq 50 \
    --use_wandb

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Find the metrics file
    latest_exp=$(find ./work_dirs/clip_linear_prob/ClipLinearProb -name "clip_vit_base_*" -type d | sort | tail -1)
    metrics_file="$latest_exp/training_metrics.json"
    
    if [ -f "$metrics_file" ]; then
        echo "Creating training plots..."
        python plot_training_curves.py --metrics_file "$metrics_file"
        echo "Plots saved to: $latest_exp"
    else
        echo "Warning: Metrics file not found at $metrics_file"
    fi
else
    echo "Training failed!"
    exit 1
fi