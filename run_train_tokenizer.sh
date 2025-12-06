project=tokenizer_training
batch_size=32
data_path=./data/imagenet/train

model=detok_BB
token_channels=128
img_size=512
patch_size=16
pretrained_model_name_or_path=""
num_register_tokens=0
aux_model_type="dinov3"
aux_dec_type="transformer"
aux_input_type="noisy"
aux_target="align"
aux_loss_type="cosine"
channel_drop=0.0
reconstruction_weight=1.0
perceptual_weight=1.0
discriminator_weight=0.5
kl_loss_weight=1e-6
aux_loss_weight=1.0

epochs=200
discriminator_start_epoch=100
gamma=1.0
noise_schedule="shift"  # lognorm, shift, uniform
mask_ratio=0.4
mask_ratio_min=-0.1
mask_ratio_type="random"
vit_aux_model_size="tiny"

exp_name="detokBB${pretrained_model_name_or_path}-img${img_size}-ch${token_channels}-p${patch_size}-g${gamma}${noise_schedule}-m${mask_ratio_min}${mask_ratio}${mask_ratio_type}"
exp_name="${exp_name}-aux${aux_model_type}${aux_dec_type}${vit_aux_model_size}${aux_input_type}${aux_loss_weight}${aux_target}"

MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT=(${ARNOLD_WORKER_0_PORT//,/ })
NNODES=${ARNOLD_WORKER_NUM}
NODE_RANK=${ARNOLD_ID}

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}

echo "[INFO] nnodes=${NNODES}, node_rank=${NODE_RANK}, nproc_per_node=${NPROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"
global_batch=$(( batch_size * NNODES * NPROC_PER_NODE ))
echo "[INFO] per-GPU batch=${batch_size}, global batch=${global_batch}"

torchrun \
  --nnodes="${NNODES:-1}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29501}" \
  main_reconstruction.py \
  --project "${project}" --exp_name "${exp_name}" --auto_resume \
  --batch_size "${batch_size}" --model "${model}" \
  --token_channels "${token_channels}" \
  --img_size "${img_size}" \
  --patch_size "${patch_size}" \
  --pretrained_model_name_or_path "${pretrained_model_name_or_path}" \
  --num_register_tokens "${num_register_tokens}" \
  --aux_model_type "${aux_model_type}" \
  --aux_dec_type "${aux_dec_type}" \
  --aux_input_type "${aux_input_type}" \
  --aux_target "${aux_target}" \
  --aux_loss_type "${aux_loss_type}" \
  --gamma "${gamma}" \
  --noise_schedule "${noise_schedule}" \
  --channel_drop "${channel_drop}" \
  --mask_ratio "${mask_ratio}" \
  --mask_ratio_min "${mask_ratio_min}" \
  --mask_ratio_type "${mask_ratio_type}" \
  --vit_aux_model_size "${vit_aux_model_size}" \
  --reconstruction_weight "${reconstruction_weight}" \
  --perceptual_weight "${perceptual_weight}" \
  --discriminator_weight "${discriminator_weight}" \
  --kl_loss_weight "${kl_loss_weight}" \
  --aux_loss_weight "${aux_loss_weight}" \
  --online_eval \
  --eval_freq 10 \
  --milestone_interval 100 \
  --epochs "${epochs}" --discriminator_start_epoch "${discriminator_start_epoch}" \
  --data_path "${data_path}"