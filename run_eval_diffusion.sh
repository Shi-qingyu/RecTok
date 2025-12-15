project=gen_model_training
tokenizer=rectok_BB
img_size=256
token_channels=128
force_one_d_seq=0
exp_name=RecTok_eval

load_tokenizer_from="$1"
load_ditdh_from="$2"
load_auto_guidance_from="$3"

MASTER_ADDR=${ARNOLD_WORKER_0_HOST:-127.0.0.1}
MASTER_PORT=(${ARNOLD_WORKER_0_PORT//,/ })
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${ARNOLD_WORKER_NUM:-1}
NODE_RANK=${ARNOLD_ID:-0}

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}

echo "[INFO] nnodes=${NNODES}, node_rank=${NODE_RANK}, nproc_per_node=${NPROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"
global_batch=$(( batch_size * NNODES * NPROC_PER_NODE ))
echo "[INFO] per-GPU batch=${batch_size}, global batch=${global_batch}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main_diffusion.py \
    --project $project --exp_name $exp_name \
    --img_size $img_size \
    --token_channels $token_channels \
    --tokenizer $tokenizer \
    --use_ema_tokenizer \
    --collect_tokenizer_stats \
    --stats_key rectok_official_evaluation --stats_cache_path work_dirs/stats.pkl \
    --load_tokenizer_from $load_tokenizer_from \
    --load_from $load_ditdh_from \
    --model DiTDDT_xl \
    --force_one_d_seq $force_one_d_seq \
    --num_sampling_steps 150 \
    --cfg_list 1.29 1.0 \
    --evaluate \
    --eval_bsz 256 \
    --num_images 50000 \
    --data_path ./data/imagenet/train \
    --use_auto_guidance \
    --load_auto_guidance_from $load_auto_guidance_from \
    --auto_guidance_model DiTDDT_s