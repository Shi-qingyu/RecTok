#!/bin/bash

# Set required environment variables for torchrun
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

project=tokenizer_training
exp_name=detokBB-g3.0-m-1.0-200ep
batch_size=32  # global batch size = batch_size x num_nodes x 8 = 1024
num_nodes=1   # adjust for your multi-node setup

torchrun --nproc_per_node=8 --nnodes=$num_nodes --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    main_reconstruction.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size --model detok_BB \
    --gamma 3.0 --mask_ratio -1.0 \
    --online_eval \
    --epochs 200 --discriminator_start_epoch 100 \
    --data_path ./data/imagenet/train