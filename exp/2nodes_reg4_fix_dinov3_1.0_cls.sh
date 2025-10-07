# add the requirement env
sudo apt-get install ffmpeg libsm6 libxext6 tmux htop  -y

export http_proxy=bj-rd-proxy.byted.org:3128  https_proxy=bj-rd-proxy.byted.org:3128  no_proxy=code.byted.org

cd /mnt/bn/zilongdata-us/xiangtai/SemanticTok/

pip install -r requirements.txt


project=tokenizer_training
batch_size=64
data_path=./data/imagenet/train
model=detok_BB
token_channels=16
patch_size=16
pretrained_model_name_or_path=""
num_register_tokens=4
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

exp_name="detokBB${pretrained_model_name_or_path}-reg${num_register_tokens}-ch${token_channels}-p${patch_size}-g${gamma}-m${mask_ratio_min}${mask_ratio}${mask_ratio_type}-aux${aux_model_type}${aux_dec_type}${aux_input_type}${aux_target}${aux_loss_weight}cls"

# add variable
export MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
export PORT=(${ARNOLD_WORKER_0_PORT//,/ })
export NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
export NNODES=${ARNOLD_WORKER_NUM}
export NODE_RANK=${ARNOLD_ID}


echo "[INFO] per-GPU batch=${batch_size}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${PORT}" \
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
  --reconstruction_weight "${reconstruction_weight}" \
  --perceptual_weight "${perceptual_weight}" \
  --discriminator_weight "${discriminator_weight}" \
  --kl_loss_weight "${kl_loss_weight}" \
  --aux_loss_weight "${aux_loss_weight}" \
  --online_eval \
  --eval_freq 50 \
  --epochs "${epochs}" --discriminator_start_epoch "${discriminator_start_epoch}" \
  --data_path "${data_path}"
