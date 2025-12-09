project=tokenizer_training
batch_size=64
data_path=./data/imagenet/train
model=detok_BB
img_size=512
token_channels=128
patch_size=16
pretrained_model_name_or_path=""
num_register_tokens=0
exp_name="$1"
load_from="work_dirs/tokenizer_training/${exp_name}/checkpoints/epoch_0199.pth"

MASTER_ADDR=${ARNOLD_WORKER_0_HOST:-127.0.0.1}
MASTER_PORT=(${ARNOLD_WORKER_0_PORT//,/ })
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${ARNOLD_WORKER_NUM:-1}
NODE_RANK=${ARNOLD_ID:-0}

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
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
    main_reconstruction.py \
    --project "${project}" --exp_name "${exp_name}" \
    --batch_size "${batch_size}" --model "${model}" \
    --token_channels "${token_channels}" \
    --img_size "${img_size}" \
    --patch_size "${patch_size}" \
    --pretrained_model_name_or_path "${pretrained_model_name_or_path}" \
    --num_register_tokens "${num_register_tokens}" \
    --load_from "${load_from}" \
    --evaluate \
    --eval_bsz 256 \
    --data_path "${data_path}"