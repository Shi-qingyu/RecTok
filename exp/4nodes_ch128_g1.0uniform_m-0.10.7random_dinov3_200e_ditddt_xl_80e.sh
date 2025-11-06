# add the requirement env
sudo apt-get install ffmpeg libsm6 libxext6 tmux htop  -y

export NCCL_WATCHDOG_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

cd /mnt/bn/zilongdata-us/xiangtai/SemanticTok/

pip install -r requirements.txt

project=tokenizer_training
batch_size=32
data_path=./data/imagenet/train

model=detok_BB
token_channels=128
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
gamma=1.0
mask_ratio=0.7
mask_ratio_min=-0.1
mask_ratio_type="random"
vit_aux_model_size="tiny"
noise_schedule="uniform"  # lognorm, shift, uniform

exp_name="detokBB${pretrained_model_name_or_path}-ch${token_channels}-p${patch_size}-g${gamma}${noise_schedule}-m${mask_ratio_min}${mask_ratio}${mask_ratio_type}"
exp_name="${exp_name}-aux${aux_model_type}${aux_dec_type}${vit_aux_model_size}${aux_input_type}${aux_target}-11-06"

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
  --noise_schedule "${noise_schedule}" \
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
epochs=80

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
    main_diffusion.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size --epochs $epochs \
    --pretrained_model_name_or_path "" \
    --num_register_tokens $num_register_tokens \
    --token_channels $token_channels \
    --tokenizer $tokenizer --use_ema_tokenizer --collect_tokenizer_stats \
    --stats_key $tokenizer_exp_name --stats_cache_path work_dirs/stats.pkl \
    --load_tokenizer_from work_dirs/tokenizer_training/$tokenizer_exp_name/checkpoints/epoch_0199.pth \
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
    --milestone_interval 80 \
    --num_sampling_steps 50 --cfg 1.6 \
    --cfg_list 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 \
    --keep_eval_folder \
    --vis_freq 50 --eval_bsz 256 \
    --data_path ./data/imagenet/train
