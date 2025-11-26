tokenizer_project=tokenizer_training
tokenizer=sd3vae
pretrained_model_name_or_path=""
tokenizer_exp_name=sd3vae
num_register_tokens=0
token_channels=16
tokenizer_patch_size=8

force_one_d_seq=0
exp_name=sit_b-${tokenizer_exp_name}

project=gen_model_training
model=SiT_base
batch_size=64
epochs=100

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}

echo "[INFO] nnodes=${WORLD_SIZE}, node_rank=${RANK}, nproc_per_node=${NPROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"
global_batch=$(( batch_size * WORLD_SIZE * NPROC_PER_NODE ))
echo "[INFO] per-GPU batch=${batch_size}, global batch=${global_batch}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${WORLD_SIZE:-1}" \
    --node_rank="${RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29500}" \
    main_diffusion.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size --epochs $epochs --use_aligned_schedule \
    --pretrained_model_name_or_path "${pretrained_model_name_or_path}" \
    --num_register_tokens $num_register_tokens \
    --token_channels $token_channels \
    --tokenizer_patch_size $tokenizer_patch_size \
    --tokenizer $tokenizer \
    --model $model \
    --force_one_d_seq $force_one_d_seq \
    --num_sampling_steps 250 --cfg 1.6 \
    --cfg_list 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 \
    --online_eval --eval_freq 10 \
    --vis_freq 50 --eval_bsz 256 \
    --data_path ./data/imagenet/train