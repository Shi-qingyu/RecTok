batch_size=64  # global batch size = batch_size x num_nodes x 8 = 1024

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 \
    linear_prob.py \
    --checkpoint_path "released_model/detok-BB-gamm3.0-m0.7.pth" \
    --batch_size $batch_size \
    --lr 1e-1 \
    --epochs 200 \
    --eval_freq 10 \
    --data_path ./data/imagenet/train \
    --output_dir ./work_dirs/linear_prob