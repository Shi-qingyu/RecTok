project=tokenizer_training
batch_size=64 # per-GPU batch size, global batch = batch_size * num_gpus * num_nodes = 1024
data_path=./data/imagenet/train

model=rectok_BB
token_channels=128
img_size=256
patch_size=16
foundation_model_type=""
reconstruction_weight=1.0
perceptual_weight=1.0
discriminator_weight=0.5
kl_loss_weight=0.0
sem_loss_weight=0.0

epochs=100
discriminator_start_epoch=0
gamma=0.0
noise_schedule="shift"  # lognorm, shift, uniform
mask_ratio=0.0
mask_ratio_min=0.0
mask_ratio_type="fix"

exp_name="${1}"
load_from="work_dirs/tokenizer_training/$exp_name/checkpoints/epoch_0199.pth"

exp_name="${exp_name}-finetune_decoder"

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
  --blr 5e-5 --warmup_rate 0.05 \
  --train_decoder_only \
  --load_from $load_from \
  --batch_size "${batch_size}" --model "${model}" \
  --token_channels "${token_channels}" \
  --img_size "${img_size}" \
  --patch_size "${patch_size}" \
  --foundation_model_type "${foundation_model_type}" \
  --gamma "${gamma}" \
  --noise_schedule "${noise_schedule}" \
  --mask_ratio "${mask_ratio}" \
  --mask_ratio_min "${mask_ratio_min}" \
  --mask_ratio_type "${mask_ratio_type}" \
  --reconstruction_weight "${reconstruction_weight}" \
  --perceptual_weight "${perceptual_weight}" \
  --discriminator_weight "${discriminator_weight}" \
  --kl_loss_weight "${kl_loss_weight}" \
  --sem_loss_weight "${sem_loss_weight}" \
  --online_eval \
  --eval_freq 10 \
  --fid_stats_path "data/fid_stats/val_fid_statistics_file_256.npz" \
  --milestone_interval 100 \
  --epochs "${epochs}" \
  --discriminator_start_epoch "${discriminator_start_epoch}" \
  --data_path "${data_path}" \
  --enable_wandb \
  --entity "qingyushi"