## Preparation


### Installation

Create and activate conda environment:
```bash
conda create -n detok python=3.10 -y && conda activate detok
pip install -r requirements.txt
```

### Dataset
Download ImageNet1K through:
```bash
bash prepare_data.sh
```

### Data Organization

Your data directory should be organized as follows:
```
data/
├── fid_stats/          # FID statistics
│   ├── adm_in256_stats.npz
│   └── val_fid_statistics_file.npz
├── imagenet/           # ImageNet dataset (or symlink)
│   ├── train/
│   │   ├──n01440764
│   │   └──n01443537
│   └── val/
│       ├──n01440764
│       └──n01443537
├── imagenet-val-prc/   # Precision-recall data
├── train.txt           # Training file list
└── val.txt             # Validation file list
```

## Training

### 1. Tokenizer Training

Train DeTok tokenizer with denoising:
```bash
project=tokenizer_training
exp_name=detokBB-tokch-16-g3.0-m0.0-aux1.0-200ep
batch_size=64  # global batch size = batch_size x num_nodes x 8 = 1024
num_nodes=2    # adjust for your multi-node setup

torchrun --nproc_per_node=8 --nnodes=$num_nodes --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    main_reconstruction.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size \
    --model detok_BB --token_channels 16 \
    --gamma 3.0 --mask_ratio 0.0 \
    --use_aux_decoder --aux_loss_weight 1.0 \
    --online_eval \
    --epochs 200 --discriminator_start_epoch 100 \
    --data_path ./data/imagenet/train
```

Decoder fine-tuning:
```bash
project=tokenizer_training
exp_name=detokBB-g3.0-m0.7-200ep-decoder_ft-100ep
batch_size=64
num_nodes=2
pretrained_tok=work_dirs/tokenizer_training/detokBB-g3.0-m0.7-200ep/checkpoints/latest.pth

torchrun --nproc_per_node=8 --nnodes=$num_nodes --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    main_reconstruction.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size --model detok_BB \
    --load_from $pretrained_tok \
    --online_eval --train_decoder_only \
    --perceptual_weight 0.1 \
    --gamma 0.0 --mask_ratio 0.0 \
    --blr 5e-5 --warmup_rate 0.05 \
    --epochs 100 --discriminator_start_epoch 0 \
    --data_path ./data/imagenet/train
```

Latent head fine-tuning:
```bash
project=tokenizer_training
exp_name=detokBB-ch32-g3.0-m-0.10.7random-auxdinov2transformeralign-head_ft
batch_size=128
num_nodes=1
pretrained_tok=work_dirs/tokenizer_training/detokBB-ch32-g3.0-m-0.10.7random-auxdinov2transformeralign/checkpoints/epoch_0199.pth

torchrun --nproc_per_node=8 --nnodes=$num_nodes --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    main_reconstruction.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size --model detok_BB \
    --load_from $pretrained_tok \
    --online_eval --train_latent_head_only \
    --perceptual_weight 0.1 \
    --gamma 0.0 --mask_ratio 0.0 \
    --blr 5e-5 --warmup_rate 0.05 \
    --epochs 100 --discriminator_start_epoch 0 \
    --data_path ./data/imagenet/train
```

### 2. Generative Model Training

Train SiT-base (100 epochs):
```bash
tokenizer_project=tokenizer_training
tokenizer_exp_name=detokBB-tokch-16-g3.0-m0.0-aux1.0-200ep  # should be same as exp_name in the tokenizer training script
project=gen_model_training
exp_name=sit_base-${tokenizer_exp_name}
batch_size=64
num_nodes=2
epochs=100

torchrun --nproc_per_node=8 --nnodes=$num_nodes --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    main_diffusion.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size --epochs $epochs --use_aligned_schedule \
    --tokenizer detok_BB --token_channels 16 \
    --use_ema_tokenizer --collect_tokenizer_stats \
    --stats_key $tokenizer_exp_name --stats_cache_path work_dirs/stats.pkl --overwrite_stats \
    --load_tokenizer_from work_dirs/$tokenizer_project/$tokenizer_exp_name/checkpoints/latest.pth \
    --model SiT_base \
    --num_sampling_steps 250 --cfg 1.6 \
    --cfg_list 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 \
    --vis_freq 50 --eval_bsz 256 \
    --data_path ./data/imagenet/train
```