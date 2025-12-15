project=tokenizer_training
data_path=./data/imagenet/train
model=rectok_BB
img_size=256
token_channels=128
patch_size=16
exp_name="RecTok_eval"

load_from="$1"

MASTER_ADDR=${ARNOLD_WORKER_0_HOST:-127.0.0.1}
MASTER_PORT=(${ARNOLD_WORKER_0_PORT//,/ })
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${ARNOLD_WORKER_NUM:-1}
NODE_RANK=${ARNOLD_ID:-0}

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}

echo "[INFO] nnodes=${NNODES}, node_rank=${NODE_RANK}, nproc_per_node=${NPROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main_reconstruction.py \
    --project "${project}" --exp_name "${exp_name}" \
    --model "${model}" \
    --token_channels "${token_channels}" \
    --img_size "${img_size}" \
    --patch_size "${patch_size}" \
    --load_from "${load_from}" \
    --evaluate \
    --eval_bsz 256 \
    --data_path "${data_path}"