import argparse
import sys
import logging
import os
import time
import json

import torch
from torch import nn
import torchvision.transforms as transforms

from utils.loader import ListDataset, center_crop_arr
import utils.distributed as dist
import utils.misc as misc
from utils.logger import MetricLogger, SmoothedValue


from transformers import CLIPProcessor, CLIPVisionModel

logger = logging.getLogger("Linear Probing")


class LinearProbing(nn.Module):
    def __init__(self, clip_model, num_classes: int):
        super().__init__()
        self.model = clip_model
        token_channels = clip_model.config.hidden_size  # 768 for CLIP-base
        self.linear = nn.Linear(token_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # CLIP vision model expects pixel values
            outputs = self.model(pixel_values=x)
            # Use the pooler output (global image representation)
            z = outputs.pooler_output  # [B, hidden_size]
        logits = self.linear(z)
        return logits


def create_clip_backbone(device):
    """Create CLIP vision backbone using transformers library"""
    model_name = "openai/clip-vit-base-patch16"
    
    # Load processor and model
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPVisionModel.from_pretrained(model_name)
    
    # Freeze the model
    clip_model.eval()
    clip_model.requires_grad_(False)
    
    return clip_model.to(device), processor


def create_clip_transform(processor):
    """Create transform using CLIP processor"""
    def clip_transform(image):
        # Convert PIL image to tensor and apply CLIP preprocessing
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)  # Remove batch dimension
    
    return clip_transform


def setup(args: argparse.Namespace):
    """setup distributed training, logging, and experiment configuration"""
    dist.enable_distributed()
    
    if args.exp_name is None:
        args.exp_name = f"linear_prob_{time.strftime('%Y%m%d_%H%M')}"

    base_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    args.log_dir = base_dir
    
    global_rank, world_size = dist.get_global_rank(), dist.get_world_size()
    args.world_size = world_size
    args.global_bsz = args.batch_size * world_size
    
    misc.fix_random_seeds(args.seed + global_rank)
    
    if global_rank == 0:
        os.makedirs(base_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(base_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    logger.info(f"Distributed setup: rank {global_rank}/{world_size}")
    return global_rank == 0


def create_val_dataloader(args, custom_transform=None):
    """Create validation dataloader that returns labels for classification"""
    if custom_transform is None:
        transform_val = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    else:
        transform_val = custom_transform
    
    dataset_val = ListDataset(
        args.data_path.replace("train", "val"),
        data_list="data/val.txt",
        transform=transform_val,
        loader_name="img_loader",
        return_label=True,  # Enable labels for classification
        should_flip=False,
    )
    
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val,
        num_replicas=dist.get_world_size(),
        rank=dist.get_global_rank(),
        shuffle=False,
    )

    logger.info(f"Val dataset size: {len(dataset_val)}")

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.eval_bsz,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_loader_val


def create_train_dataloader(
    args, 
    should_flip=True, 
    batch_size=-1, 
    return_path=False, 
    drop_last=True, 
    custom_transform=None
):
    if custom_transform is None:
        transform_train = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    else:
        transform_train = custom_transform

    input_transform = transform_train if not args.use_cached_tokens else None
    dataset_train = ListDataset(
        args.data_path,
        data_list="data/train.txt",
        transform=input_transform,
        loader_name="img_loader" if not args.use_cached_tokens else "npz_loader",
        return_label=True,
        return_path=return_path,
        should_flip=should_flip,
    )
    logger.info(f"Train dataset size: {len(dataset_train)}")

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=dist.get_world_size(),
        rank=dist.get_global_rank(),
        shuffle=True,
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size if batch_size < 0 else batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
    )
    return data_loader_train


def evaluate(model, data_loader, device):
    """Evaluate the model on validation set"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            img = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            logits = model(img)
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels).sum().item()
            
            total_correct += correct
            total_samples += labels.size(0)
    
    # Aggregate across all processes
    total_correct_tensor = torch.tensor(total_correct, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    
    if dist.get_world_size() > 1:
        torch.distributed.all_reduce(total_correct_tensor)
        torch.distributed.all_reduce(total_samples_tensor)
    
    accuracy = total_correct_tensor.item() / total_samples_tensor.item()
    return accuracy


def main(args: argparse.Namespace) -> int:
    is_main_process = setup(args)
    # Initialize wandb if enabled
    if is_main_process and hasattr(args, 'use_wandb') and args.use_wandb:
        import wandb
        wandb.init(
            project=args.project,
            entity=args.entity,
            name=args.exp_name,
            config=args.__dict__,
            settings=wandb.Settings(init_timeout=120)
        )

    device = torch.device(f"cuda:{dist.get_global_rank()}")
    clip_model, processor = create_clip_backbone(device)
    linear_prob = LinearProbing(clip_model, args.num_classes).to(device)
    
    clip_transform = create_clip_transform(processor)
    data_loader_train = create_train_dataloader(args, custom_transform=clip_transform)
    data_loader_val = create_val_dataloader(args, custom_transform=clip_transform)

    
    # Move model to appropriate device
    device = torch.device(f"cuda:{dist.get_global_rank()}")
    linear_prob = linear_prob.to(device)
    
    # Wrap model with DistributedDataParallel
    if args.world_size > 1:
        linear_prob = torch.nn.parallel.DistributedDataParallel(
            linear_prob, 
            device_ids=[dist.get_global_rank()],
            find_unused_parameters=False
        )

    trainable_params = [p for p in linear_prob.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params, 
        lr=args.lr if args.lr else args.blr,
        weight_decay=args.weight_decay
    )

    loss_fn = nn.CrossEntropyLoss()
    
    # Setup metric logger
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=50, fmt='{value:.4f}'))

    # Track metrics for plotting
    train_metrics = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rate': []
    }

    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
            
        linear_prob.train()
        for step, batch in enumerate(data_loader_train):
            img = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # Debug: print shapes and ranges on first step
            if step == 0 and epoch == 0 and is_main_process:
                logger.info(f"Input image shape: {img.shape}")
                logger.info(f"Input image range: [{img.min().item():.3f}, {img.max().item():.3f}]")
                logger.info(f"Labels shape: {labels.shape}")
                logger.info(f"Labels range: [{labels.min().item()}, {labels.max().item()}]")

            logits = linear_prob(img)   # [b, num_classes]
            loss = loss_fn(logits, labels)
            
            # Debug: print logits stats on first step
            if step == 0 and epoch == 0 and is_main_process:
                logger.info(f"Logits shape: {logits.shape}")
                logger.info(f"Logits mean: {logits.mean().item():.3f}, std: {logits.std().item():.3f}")
                logger.info(f"Initial loss: {loss.item():.4f}")
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Aggregate metrics across processes
            loss_reduced = dist.all_reduce_mean(loss)
            acc_reduced = dist.all_reduce_mean(acc)
            
            metric_logger.update(
                loss=loss_reduced,
                acc=acc_reduced,
                lr=optimizer.param_groups[0]["lr"]
            )
            
            if is_main_process and step % args.print_freq == 0:
                logger.info(
                    f"Epoch: [{epoch}/{args.epochs}] Step: [{step}/{len(data_loader_train)}] "
                    f"Loss: {loss_reduced:.4f} Acc: {acc_reduced:.4f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
                
                # Log to wandb
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({
                        "train/loss": loss_reduced,
                        "train/accuracy": acc_reduced,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "step": step + epoch * len(data_loader_train)
                    })
        
        # Validation and metric tracking
        val_acc = 0.0
        if epoch % args.eval_freq == 0:
            val_acc = evaluate(linear_prob, data_loader_val, device)
            if is_main_process:
                logger.info(f"Epoch {epoch} - Validation accuracy: {val_acc:.4f}")
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({
                        "val/accuracy": val_acc,
                        "epoch": epoch
                    })

    # Cleanup wandb
    if is_main_process and hasattr(args, 'use_wandb') and args.use_wandb:
        wandb.finish()
    
    return 0


def get_args_parser():
    parser = argparse.ArgumentParser("Reconstruction model training", add_help=False)

    # basic training parameters
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size per GPU for training")

    # model parameters
    parser.add_argument("--model", default="detok_BB", type=str)
    parser.add_argument("--token_channels", default=16, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--patch_size", default=16, type=int)

    parser.add_argument("--mask_ratio", default=0.0, type=float)
    parser.add_argument("--gamma", default=0.0, type=float, help="noise standard deviation for training")
    parser.add_argument("--use_additive_noise", action="store_true")

    parser.add_argument("--checkpoint_path", default=None, type=str)

    # logging parameters
    parser.add_argument("--output_dir", default="./work_dirs/linear_prob")
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=10)


    # evaluation parameters
    parser.add_argument("--num_images", default=50000, type=int, help="Number of images to evaluate on")
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--fid_stats_path", type=str, default="data/fid_stats/val_fid_statistics_file.npz")
    parser.add_argument("--keep_eval_folder", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval_bsz", type=int, default=256)

    # optimization parameters
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=3e-3)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_sched", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_rate", type=float, default=0.25)
    parser.add_argument("--ema_rate", default=0.999, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--grad_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for AdamW optimizer")

    # dataset parameters
    parser.add_argument("--use_cached_tokens", action="store_true")
    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--class_of_interest", default=[207, 360, 387, 974, 88, 979, 417, 279], type=int, nargs="+")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # system parameters
    parser.add_argument("--seed", default=1, type=int)

    # wandb parameters
    parser.add_argument("--project", default="linear_probing", type=str)
    parser.add_argument("--entity", default="220bd3c9b2335d6ac97c90d30fb9b6880f0b7008", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--use_wandb", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exit_code = main(args)
    sys.exit(exit_code)