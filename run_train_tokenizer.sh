project=tokenizer_training
batch_size=64
data_path=./data/imagenet/train
model=detok_BB
token_channels=16
pretrained_model_name_or_path="postmask"
aux_model_type="dinov2"
aux_dec_type="transformer"
aux_target="reconstruction"

epochs=200
discriminator_start_epoch=100
gamma=3.0
mask_ratio=0.5
mask_ratio_type="fix"

exp_name="detokBB${pretrained_model_name_or_path}-ch${token_channels}-g${gamma}-m${mask_ratio}${mask_ratio_type}-aux${aux_model_type}${aux_dec_type}${aux_target}"

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}

torchrun \
  --nnodes="${WORLD_SIZE:-1}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  main_reconstruction.py \
  --project "${project}" --exp_name "${exp_name}" --auto_resume \
  --batch_size "${batch_size}" --model "${model}" \
  --token_channels "${token_channels}" \
  --pretrained_model_name_or_path "${pretrained_model_name_or_path}" \
  --aux_model_type "${aux_model_type}" \
  --aux_dec_type "${aux_dec_type}" \
  --aux_target "${aux_target}" \
  --gamma "${gamma}" --mask_ratio "${mask_ratio}" --mask_ratio_type "${mask_ratio_type}" \
  --online_eval \
  --epochs "${epochs}" --discriminator_start_epoch "${discriminator_start_epoch}" \
  --data_path "${data_path}"