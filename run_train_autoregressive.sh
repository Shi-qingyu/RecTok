tokenizer_project=tokenizer_training
tokenizer_exp_name=detokBB-g3.0-m0.7-200ep
project=gen_model_training
exp_name=mar_base-${tokenizer_exp_name}
batch_size=32  # global batch size = batch_size x num_nodes x 8 = 1024 
num_nodes=4
epochs=100

# Set default values for distributed training parameters
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

torchrun --nproc_per_node=8 --nnodes=$num_nodes --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    main_diffusion.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size --epochs $epochs --use_aligned_schedule \
    --tokenizer detok_BB --use_ema_tokenizer --collect_tokenizer_stats \
    --stats_key $tokenizer_exp_name --stats_cache_path work_dirs/stats.pkl \
    --load_tokenizer_from work_dirs/$tokenizer_project/$tokenizer_exp_name/checkpoints/latest.pth \
    --model MAR_base --no_dropout_in_mlp \
    --diffloss_d 6 --diffloss_w 1024 \
    --num_sampling_steps 100 --cfg 4.0 \
    --cfg_list 3.0 3.5 3.7 3.8 3.9 4.0 4.1 4.3 4.5 \ --vis_freq 50 --eval_bsz 256 \
    --data_path ./data/imagenet/train