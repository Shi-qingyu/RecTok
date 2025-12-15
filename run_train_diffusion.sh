tokenizer_project=tokenizer_training
tokenizer=rectok_BB
token_channels=128
tokenizer_exp_name=${1}

project=gen_model_training
exp_name=ditddt_xl-${tokenizer_exp_name}
force_one_d_seq=0
model=DiTDDT_xl
batch_size=64
epochs=800

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}

echo "[INFO] nnodes=${NNODES}, node_rank=${NODE_RANK}, nproc_per_node=${NPROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"
global_batch=$(( batch_size * WORLD_SIZE * NPROC_PER_NODE ))
echo "[INFO] per-GPU batch=${batch_size}, global batch=${global_batch}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-12345}" \
    main_diffusion.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size --epochs $epochs \
    --token_channels $token_channels \
    --tokenizer $tokenizer --use_ema_tokenizer --collect_tokenizer_stats \
    --stats_key $tokenizer_exp_name --stats_cache_path work_dirs/stats.pkl \
    --load_tokenizer_from work_dirs/tokenizer_training/$tokenizer_exp_name/checkpoints/latest.pth \
    --model $model \
    --force_one_d_seq $force_one_d_seq \
    --lr 2e-4 \
    --min_lr 2e-5 \
    --lr_sched "linear" \
    --grad_clip  1.0 \
    --weight_decay 0.0 \
    --ema_rate 0.9995 \
    --ditdh_sched \
    --milestone_interval 80 \
    --warmup_start_epoch 40 \
    --warmup_end_epoch 800 \
    --num_sampling_steps 50 --cfg 1.0 \
    --cfg_list 1.0 \
    --online_eval --eval_freq 10 \
    --vis_freq 50 --eval_bsz 256 \
    --data_path ./data/imagenet/train \
    --enable_wandb \
    --entity "YOUR_WANDB_ENTITY"