project=tokenizer_training
batch_size=64
data_path=./data/imagenet/train
model=detok_BB
token_channels=16
patch_size=16
pretrained_model_name_or_path=""
num_register_tokens=0
aux_model_type="dinov3"
aux_dec_type="transformer"
aux_input_type="noisy"
aux_target="align"
reconstruction_weight=1.0
perceptual_weight=1.0
discriminator_weight=0.5
kl_loss_weight=1e-6
aux_loss_weight=1.0

epochs=50
discriminator_start_epoch=1
gamma=3.0
mask_ratio=0.0
mask_ratio_min=0.0
mask_ratio_type="fix"
vit_aux_model_size="tiny"

exp_name="detokBB${pretrained_model_name_or_path}-reg${num_register_tokens}-ch${token_channels}-p${patch_size}-g${gamma}-m${mask_ratio_min}${mask_ratio}${mask_ratio_type}-aux${aux_model_type}${aux_dec_type}${aux_input_type}${aux_target}diffcls"


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
  --nnodes="${WORLD_SIZE:-1}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29501}" \
  main_reconstruction.py \
  --project "${project}" --exp_name "${exp_name}" --auto_resume \
  --batch_size "${batch_size}" --model "${model}" \
  --token_channels "${token_channels}" \
  --patch_size "${patch_size}" \
  --pretrained_model_name_or_path "${pretrained_model_name_or_path}" \
  --num_register_tokens "${num_register_tokens}" \
  --aux_model_type "${aux_model_type}" \
  --aux_dec_type "${aux_dec_type}" \
  --aux_input_type "${aux_input_type}" \
  --aux_target "${aux_target}" \
  --gamma "${gamma}" \
  --mask_ratio "${mask_ratio}" \
  --mask_ratio_min "${mask_ratio_min}" \
  --mask_ratio_type "${mask_ratio_type}" \
  --vit_aux_model_size "${vit_aux_model_size}" \
  --aux_cls_token \
  --diff_cls_token \
  --reconstruction_weight "${reconstruction_weight}" \
  --perceptual_weight "${perceptual_weight}" \
  --discriminator_weight "${discriminator_weight}" \
  --kl_loss_weight "${kl_loss_weight}" \
  --aux_loss_weight "${aux_loss_weight}" \
  --online_eval \
  --epochs "${epochs}" --discriminator_start_epoch "${discriminator_start_epoch}" \
  --data_path "${data_path}"