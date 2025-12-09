tokenizer_project=tokenizer_training
tokenizer=rectok_BB
tokenizer_exp_name=rectokBB-ch128-p16-g1.0shift-m-0.10.6random-auxdinov3btransformertinynoisy1.0align
num_register_tokens=0
token_channels=128

force_one_d_seq=0
exp_name=ditddt_xl-${tokenizer_exp_name}

project=gen_model_training
model=DiTDDT_xl
batch_size=128
epochs=80

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
    --batch_size $batch_size --epochs $epochs \
    --pretrained_model_name_or_path "" \
    --num_register_tokens $num_register_tokens \
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
    --data_path ./data/imagenet/train