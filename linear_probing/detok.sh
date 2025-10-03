batch_size=512

checkpoint_path=$1
token_channels=${2:-16}

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}

export PYTHONPATH=.
torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 \
    linear_probing/detok.py \
    --model detok_BB \
    --last_layer_feature \
    --pretrained_model_name_or_path "" \
    --token_channels ${token_channels} \
    --checkpoint_path ${checkpoint_path} \
    --batch_size $batch_size \
    --epochs 1 \
    --print_freq 50 \
    --eval_freq 1 \
    --data_path ./data/imagenet/train \
    --output_dir ./work_dirs/linear_prob