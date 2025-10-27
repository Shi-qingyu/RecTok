batch_size=512

checkpoint_path=work_dirs/tokenizer_training/detokBB-ch768-p16-g3.0lognorm-m-0.10.7random-auxdinov3transformernoisyalign-10-20/checkpoints/epoch_0199.pth
token_channels=768
pretrained_model_name_or_path=""
num_register_tokens=0

# add variable
# add variable
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
export NPROC_PER_NODE=8
export NNODES=1
export NODE_RANK=0

export PYTHONPATH=.
torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    linear_probing/detok.py \
    --model detok_BB \
    --last_layer_feature \
    --num_register_tokens ${num_register_tokens} \
    --pretrained_model_name_or_path ${pretrained_model_name_or_path} \
    --token_channels ${token_channels} \
    --checkpoint_path ${checkpoint_path} \
    --batch_size ${batch_size} \
    --epochs 1 \
    --print_freq 50 \
    --eval_freq 1 \
    --data_path ./data/imagenet/train \
    --output_dir ./work_dirs/linear_prob