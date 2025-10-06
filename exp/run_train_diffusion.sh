#!/bin/bash

tokenizer_project=tokenizer_training
tokenizer_exp_name=detokBB-ch16-g3.0-m0.7-auxdinov2
num_register_tokens=0

force_one_d_seq=0
exp_name=sit_b-${tokenizer_exp_name}

project=gen_model_training
batch_size=128  # nnodes * ngpus * batch_size = 1024
epochs=100

# add variable
export MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT=10122
NPROC_PER_NODE=8
WORLD_SIZE=1
RANK=0


echo "[INFO] nnodes=${WORLD_SIZE}, node_rank=${RANK}, nproc_per_node=${NPROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"
global_batch=$(( batch_size * WORLD_SIZE * NPROC_PER_NODE ))
echo "[INFO] per-GPU batch=${batch_size}, global batch=${global_batch}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${WORLD_SIZE:-1}" \
    --node_rank="${RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29501}" \
    main_diffusion.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size --epochs $epochs --use_aligned_schedule \
    --pretrained_model_name_or_path "" \
    --num_register_tokens $num_register_tokens \
    --tokenizer detok_BB --use_ema_tokenizer --collect_tokenizer_stats \
    --stats_key $tokenizer_exp_name --stats_cache_path work_dirs/stats.pkl \
    --load_tokenizer_from work_dirs/$tokenizer_project/$tokenizer_exp_name/checkpoints/epoch_0199.pth \
    --model SiT_base \
    --force_one_d_seq $force_one_d_seq \
    --num_sampling_steps 250 --cfg 1.3 \
    --cfg_list 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 \
    --online_eval --eval_freq 10 \
    --vis_freq 50 --eval_bsz 256 \
    --data_path ./data/imagenet/train