project=tokenizer_training
batch_size=64
data_path=./data/imagenet/train

model=rectok_BB
token_channels=128
img_size=256
patch_size=16
foundation_model_type="dinov3"
sem_dec_type="transformer"
sem_input_type="noisy"
sem_target="rec+align"
sem_loss_type="cosine"
reconstruction_weight=1.0
perceptual_weight=1.0
discriminator_weight=0.5
kl_loss_weight=1e-6
sem_loss_weight=1.0

epochs=200
discriminator_start_epoch=100
gamma=1.0
noise_schedule="shift"  # lognorm, shift, uniform
mask_ratio=0.4
mask_ratio_min=-0.1
mask_ratio_type="random"
vit_sem_model_size="tiny"

exp_name="${model}-img${img_size}-ch${token_channels}-p${patch_size}-g${gamma}${noise_schedule}-m${mask_ratio_min}${mask_ratio}${mask_ratio_type}"
exp_name="${exp_name}-sem${foundation_model_type}${sem_dec_type}${vit_sem_model_size}${sem_input_type}${sem_loss_weight}${sem_target}"

MASTER_ADDR=${ARNOLD_WORKER_0_HOST:-127.0.0.1}
MASTER_PORT=(${ARNOLD_WORKER_0_PORT//,/ })
MASTER_PORT=${MASTER_PORT:-12345}
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
    --project "${project}" --exp_name "${exp_name}" --auto_resume \
    --batch_size "${batch_size}" --model "${model}" \
    --token_channels "${token_channels}" \
    --img_size "${img_size}" \
    --patch_size "${patch_size}" \
    --foundation_model_type "${foundation_model_type}" \
    --sem_dec_type "${sem_dec_type}" \
    --sem_input_type "${sem_input_type}" \
    --sem_target "${sem_target}" \
    --sem_loss_type "${sem_loss_type}" \
    --gamma "${gamma}" \
    --noise_schedule "${noise_schedule}" \
    --mask_ratio "${mask_ratio}" \
    --mask_ratio_min "${mask_ratio_min}" \
    --mask_ratio_type "${mask_ratio_type}" \
    --vit_sem_model_size "${vit_sem_model_size}" \
    --reconstruction_weight "${reconstruction_weight}" \
    --perceptual_weight "${perceptual_weight}" \
    --discriminator_weight "${discriminator_weight}" \
    --kl_loss_weight "${kl_loss_weight}" \
    --sem_loss_weight "${sem_loss_weight}" \
    --online_eval \
    --eval_freq 10 \
    --fid_stats_path "data/fid_stats/val_fid_statistics_file_${img_size}.npz" \
    --milestone_interval 100 \
    --epochs "${epochs}" \
    --discriminator_start_epoch "${discriminator_start_epoch}" \
    --data_path "${data_path}" \
    --enable_wandb \
    --entity "YOUR_WANDB_ENTITY"