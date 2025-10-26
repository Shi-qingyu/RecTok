batch_size=512

checkpoint_path=work_dirs/tokenizer_training/detokBB-ch768-p16-g3.0lognorm-m0.00.0fix-auxdinov3transformernoisyalign-10-20/checkpoints/epoch_0199.pth
token_channels=768
pretrained_model_name_or_path=""
num_register_tokens=0

# add variable
export MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
export PORT=(${ARNOLD_WORKER_0_PORT//,/ })
export NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
export NNODES=${ARNOLD_WORKER_NUM}
export NODE_RANK=${ARNOLD_ID}

export PYTHONPATH=.
torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${PORT} \
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