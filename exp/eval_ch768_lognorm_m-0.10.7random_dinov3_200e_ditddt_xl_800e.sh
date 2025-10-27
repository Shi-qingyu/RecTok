# add the requirement env
sudo apt-get install ffmpeg libsm6 libxext6 tmux htop  -y

export NCCL_WATCHDOG_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

cd /mnt/bn/zilongdata-us/xiangtai/SemanticTok/

# Use uv instead of pip
pip install -r requirements.txt

project=tokenizer_training
batch_size=32
data_path=./data/imagenet/train

model=detok_BB
token_channels=768
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

epochs=200
discriminator_start_epoch=100
gamma=3.0
mask_ratio=0.7
mask_ratio_min=-0.1
mask_ratio_type="random"
vit_aux_model_size="tiny"

exp_name="detokBB${pretrained_model_name_or_path}-ch${token_channels}-p${patch_size}-g${gamma}lognorm-m${mask_ratio_min}${mask_ratio}${mask_ratio_type}-aux${aux_model_type}${aux_dec_type}${aux_input_type}${aux_target}-10-20"

# add variable
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
export NPROC_PER_NODE=8
export NNODES=1
export NODE_RANK=0


echo "[INFO] per-GPU batch=${batch_size}"

# Use uv run to execute the training script
torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
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
  --use_log_normal_noise \
  --mask_ratio "${mask_ratio}" \
  --mask_ratio_min "${mask_ratio_min}" \
  --mask_ratio_type "${mask_ratio_type}" \
  --vit_aux_model_size "${vit_aux_model_size}" \
  --reconstruction_weight "${reconstruction_weight}" \
  --perceptual_weight "${perceptual_weight}" \
  --discriminator_weight "${discriminator_weight}" \
  --kl_loss_weight "${kl_loss_weight}" \
  --aux_loss_weight "${aux_loss_weight}" \
  --epochs "${epochs}" --discriminator_start_epoch "${discriminator_start_epoch}" \
  --evaluate \
  --keep_eval_folder \
  --data_path "${data_path}"


tokenizer_project=tokenizer_training
tokenizer=detok_BB
tokenizer_exp_name=${exp_name}
num_register_tokens=0

force_one_d_seq=0
exp_name=ditddt_xl-${tokenizer_exp_name}

project=gen_model_training
model=DiTDDT_xl
batch_size=32  # nnodes * ngpus * batch_size = 1024
epochs=800


echo "[INFO] per-GPU batch=${batch_size}"

# Use uv run to execute the diffusion training script
torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${PORT}" \
    main_diffusion.py \
    --project $project --exp_name $exp_name \
    --batch_size $batch_size --epochs $epochs \
    --pretrained_model_name_or_path "" \
    --num_register_tokens $num_register_tokens \
    --token_channels $token_channels \
    --tokenizer $tokenizer --use_ema_tokenizer --collect_tokenizer_stats \
    --stats_key $tokenizer_exp_name --stats_cache_path work_dirs/stats.pkl \
    --load_tokenizer_from work_dirs/tokenizer_training/$tokenizer_exp_name/checkpoints/epoch_0199.pth \
    --load_from work_dirs/gen_model_training/$exp_name/checkpoints/latest.pth \
    --model $model \
    --force_one_d_seq $force_one_d_seq \
    --lr 2e-4 \
    --min_lr 2e-5 \
    --lr_sched "linear" \
    --grad_clip 1.0 \
    --weight_decay 0.0 \
    --ema_rate 0.9995 \
    --ditdh_sched \
    --warmup_start_epoch 40 \
    --warmup_end_epoch 800 \
    --num_sampling_steps 50 \
    --cfg_list 1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.35 1.4 \
    --evaluate \
    --keep_eval_folder \
    --vis_freq 50 --eval_bsz 256 \
    --data_path ./data/imagenet/train