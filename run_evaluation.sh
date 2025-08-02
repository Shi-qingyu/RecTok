#!/bin/bash

# Evaluation configuration
eval_bsz=256
num_images=50000

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path> [num_gpus]"
    echo "Example: $0 released_model/detok-BB-gamm3.0-m0.7.pth 4"
    exit 1
fi

checkpoint_path="$1"
num_gpus="${2:-8}"  # Default to 4 GPUs if not specified

# Get current timestamp for unique experiment name
timestamp=$(date +%Y%m%d_%H%M)
exp_name="eval_${timestamp}"

echo "Starting DeTok Model Evaluation..."
echo "Checkpoint: $checkpoint_path"
echo "Experiment: $exp_name"
echo "Batch size per GPU: $eval_bsz"
echo "Number of images: $num_images"
echo "Number of GPUs: $num_gpus"

# Check if checkpoint exists
if [ ! -f "$checkpoint_path" ]; then
    echo "Error: Checkpoint file not found: $checkpoint_path"
    exit 1
fi

# Run distributed evaluation
torchrun --nproc_per_node=$num_gpus --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29502 \
    main_evaluation.py \
    --load_from "$checkpoint_path" \
    --data_path ./data/imagenet/val \
    --eval_bsz $eval_bsz \
    --num_images $num_images \
    --output_dir ./work_dirs/eval \
    --project "DeTok_evaluation" \
    --exp_name "$exp_name" \
    --num_workers 10 \
    --pin_mem

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to: ./work_dirs/eval/DeTok_evaluation/$exp_name"
else
    echo "Evaluation failed!"
    exit 1
fi 